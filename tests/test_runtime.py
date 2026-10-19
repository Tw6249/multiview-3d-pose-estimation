from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pose_sim.runtime import create_body


class RuntimeTests(unittest.TestCase):
    def test_requested_cuda_unavailable_fails_before_model_loading(self):
        ort = SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'], set_default_logger_severity=Mock())
        with patch.dict('sys.modules', {'onnxruntime': ort}):
            with self.assertRaisesRegex(RuntimeError, 'CUDA provider unavailable'):
                create_body(device='cuda')

    def test_cuda_session_silent_cpu_fallback_is_rejected(self):
        ort = SimpleNamespace(get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                              preload_dlls=Mock(), set_default_logger_severity=Mock())
        model = SimpleNamespace(session=SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider']))
        body = Mock(return_value=SimpleNamespace(det_model=model, pose_model=model))
        with patch.dict('sys.modules', {'onnxruntime': ort, 'rtmlib': SimpleNamespace(Body=body)}):
            with self.assertRaisesRegex(RuntimeError, 'refusing silent CPU fallback'):
                create_body(device='cuda')


if __name__ == '__main__':
    unittest.main()
