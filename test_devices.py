"""Device selection and propagation without requiring GPU hardware."""

import unittest
from unittest.mock import patch

from devices import resolve_device


class DeviceTests(unittest.TestCase):
    def test_auto_selects_available_accelerator(self):
        for mps, cuda, expected in [(True, False, 'mps'), (False, True, 'cuda'), (False, False, 'cpu')]:
            with self.subTest(expected=expected), \
                    patch('torch.backends.mps.is_available', return_value=mps), \
                    patch('torch.cuda.is_available', return_value=cuda):
                self.assertEqual(resolve_device(), expected)
                self.assertEqual(resolve_device('cpu'), 'cpu')

    @patch('torch.backends.mps.is_available', return_value=False)
    def test_explicit_mps_does_not_silently_use_cpu(self, available):
        with self.assertRaisesRegex(ValueError, 'MPS is unavailable'):
            resolve_device('mps')

    @patch('torch.cuda.is_available', return_value=False)
    def test_explicit_cuda_requires_available_gpu(self, available):
        with self.assertRaisesRegex(ValueError, 'CUDA is unavailable'):
            resolve_device('cuda')
