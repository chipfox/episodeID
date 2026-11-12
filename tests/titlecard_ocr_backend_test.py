"""OCR backend selection: CUDA first, then Intel OpenVINO GPU, then CPU.

The OCR cache is keyed by the backend; a GPU read must never be reused as a CPU
read (or vice versa) after a settings change or a GPU failure.
"""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_ocr_backend"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import titlecard  # noqa: E402

real_settings = titlecard.get_settings
real_cuda = titlecard._ocr_gpu_available
real_openvino = titlecard._openvino_gpu_available


def choose(**overrides):
    titlecard._accel.clear()
    settings = {"titlecard_ocr_device": "auto", "titlecard_openvino_device": "GPU"}
    settings.update(overrides)
    titlecard.get_settings = lambda: settings
    return titlecard._ocr_choice()


try:
    titlecard._ocr_gpu_available = lambda: True
    titlecard._openvino_gpu_available = lambda: False
    assert choose() == ("cuda", "NVIDIA GPU (CUDA)"), "CUDA wins when both are present"
    assert titlecard._ocr_sig().startswith("rapidocr-2|cuda|")

    titlecard._ocr_gpu_available = lambda: False
    titlecard._openvino_gpu_available = lambda: True
    backend, label = choose(titlecard_openvino_device="GPU.0")
    assert backend == "openvino" and "GPU.0" in label, (backend, label)
    assert titlecard._ocr_sig().startswith("rapidocr-2|openvino|")

    assert choose(titlecard_ocr_device="cpu") == ("cpu", "CPU")
    titlecard._openvino_gpu_available = lambda: False
    assert choose() == ("cpu", "CPU"), "no GPU falls back to CPU"
    assert choose(titlecard_ocr_device="openvino") == ("cpu", "CPU"), \
        "an unavailable OpenVINO GPU falls back to CPU"

    titlecard._openvino_gpu_available = lambda: True
    titlecard._accel["force_cpu"] = True
    titlecard.get_settings = lambda: {"titlecard_ocr_device": "auto", "titlecard_openvino_device": "GPU"}
    assert titlecard._ocr_choice() == ("cpu", "CPU"), "a previous GPU failure forces CPU"
finally:
    titlecard.get_settings = real_settings
    titlecard._ocr_gpu_available = real_cuda
    titlecard._openvino_gpu_available = real_openvino
    titlecard._accel.clear()

print("titlecard OCR backend: CUDA → OpenVINO GPU → CPU, cache signature tracks backend")
