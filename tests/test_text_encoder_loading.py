from unittest import TestCase, mock

import torch

from ardy.model.load_model import load_text_encoder


class _DummyTextEncoder:
    def __init__(self):
        self.device = None
        self.dtype = None

    def to(self, *, device=None, dtype=None):
        self.device = device
        self.dtype = dtype
        return self


class TextEncoderLoadingTests(TestCase):
    def test_unknown_mode_is_rejected_before_instantiation(self):
        with self.assertRaisesRegex(ValueError, "Unknown text-encoder mode"):
            load_text_encoder(mode="qwen")

    @mock.patch("ardy.model.load_model.select_device", return_value="mps")
    @mock.patch("ardy.model.load_model._select_text_encoder_conf")
    def test_resolved_device_and_dtype_are_applied(self, select_conf, select_device):
        encoder = _DummyTextEncoder()
        select_conf.return_value = ({}, encoder)

        result = load_text_encoder(mode="local", device="mps")

        self.assertIs(result, encoder)
        self.assertEqual(encoder.device, "mps")
        self.assertIs(encoder.dtype, torch.bfloat16)
        select_device.assert_called_once_with("mps")
        select_conf.assert_called_once()
