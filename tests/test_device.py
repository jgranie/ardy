from pathlib import Path
from unittest import SkipTest, TestCase, mock

import torch

from ardy.device import (
    available_device_types,
    clear_device_cache,
    device_type,
    select_device,
    supports_tensorrt,
)
from ardy.skeleton import CoreSkeleton27


class DeviceSelectionTests(TestCase):
    @mock.patch("ardy.device.mps_is_available", return_value=True)
    @mock.patch("ardy.device.torch.cuda.is_available", return_value=True)
    def test_auto_prefers_cuda_over_mps(self, _cuda, _mps):
        self.assertEqual(select_device(), "cuda")

    @mock.patch("ardy.device.torch.cuda.is_available", return_value=True)
    def test_explicit_cuda_index_is_preserved(self, _cuda):
        self.assertEqual(select_device("cuda:1"), "cuda:1")

    @mock.patch("ardy.device.mps_is_available", return_value=True)
    @mock.patch("ardy.device.torch.cuda.is_available", return_value=False)
    def test_auto_uses_mps_before_cpu(self, _cuda, _mps):
        self.assertEqual(select_device("auto"), "mps")

    @mock.patch("ardy.device.mps_is_available", return_value=False)
    @mock.patch("ardy.device.torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, _cuda, _mps):
        self.assertEqual(select_device(), "cpu")

    @mock.patch("ardy.device.mps_is_available", return_value=True)
    @mock.patch("ardy.device.torch.cuda.is_available", return_value=False)
    def test_empty_device_string_uses_auto(self, _cuda, _mps):
        self.assertEqual(select_device(""), "mps")

    @mock.patch("ardy.device.mps_is_available", return_value=False)
    @mock.patch("ardy.device.torch.cuda.is_available", return_value=False)
    def test_explicit_unavailable_mps_fails_early(self, _cuda, _mps):
        with self.assertRaisesRegex(RuntimeError, "MPS is unavailable"):
            select_device("mps")

    @mock.patch("ardy.device.mps_is_available", return_value=False)
    @mock.patch("ardy.device.torch.cuda.is_available", return_value=False)
    def test_explicit_unavailable_cuda_fails_early(self, _cuda, _mps):
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            select_device("cuda")

    def test_unsupported_device_type_fails_early(self):
        with self.assertRaisesRegex(ValueError, "Unsupported ARDY device"):
            select_device("xpu")

    @mock.patch("ardy.device.mps_is_available", return_value=True)
    @mock.patch("ardy.device.torch.cuda.is_available", return_value=True)
    def test_available_devices_prefer_cuda_then_mps_then_cpu(self, _cuda, _mps):
        self.assertEqual(available_device_types(), ["cuda", "mps", "cpu"])

    @mock.patch("ardy.device.mps_is_available", return_value=True)
    @mock.patch("ardy.device.torch.cuda.is_available", return_value=False)
    def test_available_devices_include_mps_and_cpu(self, _cuda, _mps):
        self.assertEqual(available_device_types(), ["mps", "cpu"])

    def test_device_type_and_tensorrt_support(self):
        self.assertEqual(device_type(torch.device("mps")), "mps")
        self.assertTrue(supports_tensorrt("cuda:1"))
        self.assertFalse(supports_tensorrt("mps"))

    @mock.patch("ardy.device.torch.cuda.reset_peak_memory_stats")
    @mock.patch("ardy.device.torch.cuda.empty_cache")
    @mock.patch("ardy.device.torch.cuda.synchronize")
    @mock.patch("ardy.device.torch.cuda.is_available", return_value=True)
    @mock.patch("ardy.device.device_type", return_value="cuda")
    def test_clear_device_cache_clears_cuda_allocator(
        self, _device_type, _cuda_available, synchronize, empty_cache, reset_peak_memory_stats
    ):
        clear_device_cache("cuda:0")
        synchronize.assert_called_once_with()
        empty_cache.assert_called_once_with()
        reset_peak_memory_stats.assert_called_once_with()

    @mock.patch("ardy.device.torch.mps.empty_cache")
    @mock.patch("ardy.device.torch.mps.synchronize")
    @mock.patch("ardy.device.mps_is_available", return_value=True)
    @mock.patch("ardy.device.device_type", return_value="mps")
    def test_clear_device_cache_clears_mps_allocator(
        self, _device_type, _mps_available, synchronize, empty_cache
    ):
        clear_device_cache("mps")
        synchronize.assert_called_once_with()
        empty_cache.assert_called_once_with()


class SkeletonDtypeTests(TestCase):
    def test_cpu_skeleton_preserves_serialized_dtype_and_values(self):
        skeleton = CoreSkeleton27()
        serialized_joints = torch.load(Path(skeleton.folder) / "joints.p").squeeze()

        self.assertEqual(skeleton.neutral_joints.dtype, serialized_joints.dtype)
        self.assertTrue(torch.equal(skeleton.neutral_joints, serialized_joints))

    def test_non_mps_apply_preserves_float64_assets(self):
        for target in ("cpu", "cuda"):
            with self.subTest(target=target):
                skeleton = CoreSkeleton27()
                original_joints = skeleton.neutral_joints.clone()
                skeleton._apply(torch.clone)

                self.assertEqual(skeleton.neutral_joints.dtype, torch.float64)
                self.assertTrue(torch.equal(skeleton.neutral_joints, original_joints))

    def test_parent_apply_casts_only_float64_buffers_for_mps(self):
        container = torch.nn.Module()
        skeleton = CoreSkeleton27()
        skeleton.register_buffer("float32_buffer", torch.ones(1, dtype=torch.float32), persistent=False)
        container.add_module("skeleton", skeleton)

        def fake_mps_apply(tensor):
            if tensor.dtype == torch.float64:
                raise TypeError("MPS framework doesn't support float64")
            return tensor.clone()

        container._apply(fake_mps_apply)

        self.assertEqual(skeleton.neutral_joints.dtype, torch.float32)
        self.assertEqual(skeleton.float32_buffer.dtype, torch.float32)
        self.assertEqual(skeleton.joint_parents.dtype, torch.int64)

    def test_mps_fallback_preserves_explicit_destination_dtype(self):
        skeleton = CoreSkeleton27()

        def fake_mps_float16_apply(tensor):
            if tensor.dtype == torch.float64:
                raise TypeError("MPS framework doesn't support float64")
            if tensor.is_floating_point():
                return tensor.to(dtype=torch.float16)
            return tensor.clone()

        skeleton._apply(fake_mps_float16_apply)

        self.assertEqual(skeleton.neutral_joints.dtype, torch.float16)
        self.assertEqual(skeleton.joint_parents.dtype, torch.int64)

    def test_unrelated_apply_error_is_not_masked(self):
        skeleton = CoreSkeleton27()

        def failing_apply(_tensor):
            raise RuntimeError("unrelated conversion failure")

        with self.assertRaisesRegex(RuntimeError, "unrelated conversion failure"):
            skeleton._apply(failing_apply)

    def test_explicit_cpu_dtype_conversion_is_honored(self):
        skeleton = CoreSkeleton27().to(device="cpu", dtype=torch.float16)
        self.assertEqual(skeleton.neutral_joints.dtype, torch.float16)
        self.assertEqual(skeleton.joint_parents.dtype, torch.int64)


class ActualMpsSmokeTests(TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.backends.mps.is_available():
            raise SkipTest("MPS is unavailable")

    def test_auto_selects_mps_and_runs_bfloat16(self):
        self.assertEqual(select_device(), "mps")
        layer = torch.nn.Linear(8, 4, device="mps", dtype=torch.bfloat16)
        output = layer(torch.randn(2, 8, device="mps", dtype=torch.bfloat16))
        torch.mps.synchronize()
        self.assertEqual(output.device.type, "mps")
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_core_skeleton_buffers_can_move_to_mps(self):
        skeleton = CoreSkeleton27().to("mps")
        self.assertEqual(skeleton.neutral_joints.device.type, "mps")
        self.assertEqual(skeleton.neutral_joints.dtype, torch.float32)
