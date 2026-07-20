import pickle
import tempfile
import threading
from types import SimpleNamespace
from unittest import SkipTest, TestCase

import numpy as np
import torch

from ardy.skeleton import CoreSkeleton27
from ardy.viz.core_skin import CoreSkin
from scripts.interactive_demo.accelerator import (
    create_playback_skeleton,
    move_playback_tensors,
    move_to_playback_device,
    playback_device,
    serialized_accelerator,
    serialized_text_encoder,
)


class _ObservableRLock:
    def __init__(self):
        self._lock = threading.RLock()
        self._count_lock = threading.Lock()
        self.attempt_count = 0
        self.second_attempted = threading.Event()

    def __enter__(self):
        with self._count_lock:
            self.attempt_count += 1
            if self.attempt_count == 2:
                self.second_attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


class _DepthRLock:
    def __init__(self):
        self._lock = threading.RLock()
        self.depth = 0

    def __enter__(self):
        self._lock.acquire()
        self.depth += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.depth -= 1
        self._lock.release()


class DemoAcceleratorTests(TestCase):
    def test_mps_methods_share_one_reentrant_lock(self):
        class Demo:
            device = "mps"

            def __init__(self):
                self._accelerator_lock = threading.RLock()

            @serialized_accelerator
            def outer(self):
                return self.inner()

            @serialized_accelerator
            def inner(self):
                return "done"

        self.assertEqual(Demo().outer(), "done")

    def test_mps_methods_are_serialized_across_threads(self):
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()

        class Demo:
            device = "mps"

            def __init__(self):
                self._accelerator_lock = _ObservableRLock()

            @serialized_accelerator
            def hold(self):
                first_entered.set()
                self.assert_event(release_first)

            @serialized_accelerator
            def enter_second(self):
                second_entered.set()

            @staticmethod
            def assert_event(event):
                if not event.wait(timeout=2):
                    raise AssertionError("test release event was not signalled")

        demo = Demo()
        first = threading.Thread(target=demo.hold)
        second = threading.Thread(target=demo.enter_second)
        first.start()
        self.assertTrue(first_entered.wait(timeout=2))
        second.start()
        self.assertTrue(demo._accelerator_lock.second_attempted.wait(timeout=2))
        self.assertFalse(second_entered.is_set())

        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertTrue(second_entered.is_set())

    def test_cpu_and_cuda_do_not_enter_the_mps_lock(self):
        class ExplodingLock:
            def __enter__(self):
                raise AssertionError("non-MPS code acquired the MPS lock")

            def __exit__(self, exc_type, exc_value, traceback):
                return None

        class Demo:
            def __init__(self, device):
                self.device = device
                self._accelerator_lock = ExplodingLock()

            @serialized_accelerator
            def run(self):
                return "unchanged"

        self.assertEqual(Demo("cpu").run(), "unchanged")
        self.assertEqual(Demo("cuda:1").run(), "unchanged")

    def test_text_encoder_is_serialized_even_when_model_uses_cpu(self):
        class CountingLock:
            entries = 0

            def __enter__(self):
                self.entries += 1

            def __exit__(self, exc_type, exc_value, traceback):
                return None

        class Demo:
            device = "cpu"

            def __init__(self):
                self._accelerator_lock = CountingLock()

            @serialized_text_encoder
            def encode(self):
                return "encoded"

        demo = Demo()
        self.assertEqual(demo.encode(), "encoded")
        self.assertEqual(demo._accelerator_lock.entries, 1)

    def test_mps_playback_state_is_detached_on_cpu(self):
        source = torch.tensor([1.0], requires_grad=True)
        state = move_to_playback_device(source, "mps")

        self.assertEqual(state.device.type, "cpu")
        self.assertFalse(state.requires_grad)
        self.assertEqual(state.item(), 1.0)
        self.assertEqual(playback_device("mps"), "cpu")

    def test_generation_playback_group_covers_every_published_buffer(self):
        tensors = {
            "motion_tensor": torch.tensor([1.0], requires_grad=True),
            "joints_pos": torch.tensor([2.0], requires_grad=True),
            "joints_rot": torch.tensor([3.0], requires_grad=True),
            "foot_contacts": torch.tensor([4.0], requires_grad=True),
            "root_velocities": torch.tensor([5.0], requires_grad=True),
        }
        state = move_playback_tensors("mps", **tensors)

        self.assertEqual(state.keys(), tensors.keys())
        for name, tensor in state.items():
            with self.subTest(name=name):
                self.assertEqual(tensor.device.type, "cpu")
                self.assertFalse(tensor.requires_grad)

    def test_cpu_and_cuda_playback_state_is_unchanged(self):
        source = torch.tensor([1.0], requires_grad=True)

        self.assertIs(move_to_playback_device(source, "cpu"), source)
        self.assertIs(move_to_playback_device(source, "cuda:1"), source)
        self.assertEqual(playback_device("cpu"), "cpu")
        self.assertEqual(playback_device("cuda:1"), "cuda:1")

    def test_mps_uses_an_independent_cpu_visualization_skeleton(self):
        inference_skeleton = CoreSkeleton27()
        viz_skeleton = create_playback_skeleton(inference_skeleton, "mps")

        self.assertIsNot(viz_skeleton, inference_skeleton)
        self.assertIsInstance(viz_skeleton, CoreSkeleton27)
        self.assertEqual(viz_skeleton.neutral_joints.device.type, "cpu")
        self.assertTrue(torch.equal(viz_skeleton.neutral_joints, inference_skeleton.neutral_joints))

        skin = CoreSkin(viz_skeleton)
        joints_pos = viz_skeleton.neutral_joints.to(dtype=torch.float32).unsqueeze(0)
        joints_rot = torch.eye(3).repeat(1, viz_skeleton.nbjoints, 1, 1)
        self.assertEqual(skin.skin(joints_rot, joints_pos, rot_is_global=True).device.type, "cpu")

    def test_non_mps_reuses_the_inference_skeleton(self):
        inference_skeleton = CoreSkeleton27()

        self.assertIs(create_playback_skeleton(inference_skeleton, "cpu"), inference_skeleton)
        self.assertIs(create_playback_skeleton(inference_skeleton, "cuda"), inference_skeleton)

    def test_native_mps_tensor_and_skeleton_are_offloaded_for_playback(self):
        if not torch.backends.mps.is_available():
            raise SkipTest("MPS is unavailable")

        source = torch.tensor([1.0, 2.0], device="mps", requires_grad=True)
        state = move_to_playback_device(source, "mps")
        self.assertEqual(state.device.type, "cpu")
        self.assertFalse(state.requires_grad)
        self.assertTrue(torch.equal(state, torch.tensor([1.0, 2.0])))

        inference_skeleton = CoreSkeleton27().to("mps")
        viz_skeleton = create_playback_skeleton(inference_skeleton, "mps")
        self.assertEqual(viz_skeleton.neutral_joints.device.type, "cpu")
        self.assertEqual(viz_skeleton.neutral_joints.dtype, torch.float64)
        self.assertTrue(
            torch.allclose(
                viz_skeleton.neutral_joints,
                inference_skeleton.neutral_joints.cpu().to(dtype=torch.float64),
                atol=1e-6,
                rtol=0,
            )
        )

    def test_accelerator_entry_points_are_marked(self):
        try:
            from scripts.interactive_demo.generation import GenerationMixin
            from scripts.interactive_demo.loading import ModelLoadingMixin
            from scripts.interactive_demo.motion_io import MotionIOMixin
            from scripts.interactive_demo.session_io import SessionIOMixin
        except ModuleNotFoundError as exc:
            raise SkipTest(f"interactive-demo extras unavailable: {exc}") from exc

        methods = [
            GenerationMixin.restart,
            GenerationMixin.restart_from_now,
            GenerationMixin._generate_step,
            ModelLoadingMixin.load_model,
            ModelLoadingMixin.load_model_and_restart,
            ModelLoadingMixin._prewarm_text_encoder,
            ModelLoadingMixin._update_text_embedding,
            ModelLoadingMixin._move_text_encoder,
            MotionIOMixin.load_motion_from_file,
            MotionIOMixin._load_sequence_locked,
            SessionIOMixin.export_session,
            SessionIOMixin.load_session,
        ]
        for method in methods:
            with self.subTest(method=method.__qualname__):
                self.assertTrue(getattr(method, "__accelerator_serialized__", False))

    def test_mps_model_replacement_is_one_locked_transaction(self):
        try:
            from scripts.interactive_demo.loading import ModelLoadingMixin
        except ModuleNotFoundError as exc:
            raise SkipTest(f"interactive-demo extras unavailable: {exc}") from exc

        class Demo(ModelLoadingMixin):
            device = "mps"

            def __init__(self):
                self._accelerator_lock = _DepthRLock()
                self.calls = []

            def load_model(self, client_id, model_name, progress=None):
                self.calls.append(("model", self._accelerator_lock.depth))
                return object()

            def _update_text_embedding(self, client_id, text_prompt):
                self.calls.append(("embedding", self._accelerator_lock.depth))
                return True

            def restart(self, client_id):
                self.calls.append(("restart", self._accelerator_lock.depth))

        demo = Demo()
        self.assertIsNotNone(demo.load_model_and_restart(1, "model", "prompt"))
        self.assertEqual(
            demo.calls,
            [("model", 1), ("embedding", 1), ("restart", 1)],
        )

    def test_session_import_keeps_mps_playback_buffers_on_cpu(self):
        try:
            from scripts.interactive_demo.session_io import SessionIOMixin
        except ModuleNotFoundError as exc:
            raise SkipTest(f"interactive-demo extras unavailable: {exc}") from exc

        class Client:
            def add_notification(self, **_kwargs):
                return None

        gui = SimpleNamespace(
            gui_frame_idx_input=SimpleNamespace(max=-1),
            gui_enable_auto_replan_checkbox=SimpleNamespace(value=True),
        )
        session = SimpleNamespace(
            client=Client(),
            motion_rep=None,
            constraints={},
            timeline_data=None,
            gui_elements=gui,
        )

        class Demo(SessionIOMixin):
            device = "mps"
            playback_device = "cpu"

            def __init__(self):
                self._accelerator_lock = threading.RLock()
                self.client_sessions = {1: session}
                self.frame = None

            def client_active(self, client_id):
                return client_id in self.client_sessions

            def set_frame(self, client_id, frame_idx):
                self.frame = (client_id, frame_idx)

        arrays = {
            "joints_pos": np.zeros((1, 2, 27, 3), dtype=np.float32),
            "joints_rot": np.zeros((1, 2, 27, 3, 3), dtype=np.float32),
            "root_velocities": np.zeros((1, 2, 3), dtype=np.float32),
            "motion_tensor": np.zeros((1, 2, 8), dtype=np.float32),
            "foot_contacts": np.zeros((1, 2, 4), dtype=np.float32),
        }
        payload = {
            "version": "1.0",
            "timestamp": "test",
            "max_frame_idx": 1,
            "motion": arrays,
            "constraints": {},
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl") as handle:
            pickle.dump(payload, handle)
            handle.flush()
            self.assertTrue(Demo().load_session(1, handle.name))

        for name in arrays:
            with self.subTest(name=name):
                self.assertEqual(getattr(session, name).device.type, "cpu")
