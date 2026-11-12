# OpenVINO Title-Card OCR on the Intel iGPU — Implementation Plan

> **For Hermes:** use `subagent-driven-development` to execute the spike first, then the implementation tasks. Do not enable GPU OCR by default until the parity and benchmark gates pass.

**Goal:** Move title-card OCR from the i9-12900H CPU to the Iris Xe 96 EU iGPU via OpenVINO, while keeping CPU as a deterministic fallback and preserving OCR results.

> **Status (2026-11-12): implemented on `feat/openvino-ocr`.** Spike A
> (`onnxruntime-openvino` 1.24.1 + OpenVINO 2026.4.1, provider patch with
> `precision=FP32`) is the chosen path. On the Iris Xe it measured **78 ms/frame
> steady-state vs 1,344 ms CPU (~17×)** with identical OCR text on the five test
> frames. The image includes the pinned Intel compute runtime. No Arc A310 move is
> needed; title cards remain disabled globally until the normal test/enable flow.

**Architecture:** The compose stack now passes `/dev/dri` into the EpisodeID container and VAAPI decode is confirmed. OCR still runs on CPU because `rapidocr_onnxruntime` only knows CPU/CUDA/DirectML and the image has no OpenVINO runtime. Implement a small OCR backend selector in `app/titlecard.py`: CUDA when present, then OpenVINO GPU, then CPU. Use a short spike to choose between `onnxruntime-openvino` with a provider patch and RapidOCR 3.x's OpenVINO engine (which currently hardcodes CPU and needs a device patch). Cache the OCR backend/device in `OCR_SIG` so CPU and GPU reads cannot be mixed silently.

**Tech stack:** Python 3.12 slim image, OpenVINO, Intel OpenCL/Level Zero runtime, RapidOCR, ffmpeg VAAPI, Docker/Portainer stack 190, Iris Xe (iGPU) or Arc A310 later.

---

## 1. Current state (verified live)

- Stack 190 compose now has:
  - `devices: /dev/dri:/dev/dri`
  - `group_add: ["104"]` (turnkey2 `render` GID; container runs as root anyway)
- Container sees `/dev/dri/renderD128`.
- `ffmpeg -hwaccel vaapi -hwaccel_device /dev/dri/renderD128` decodes a Beavis file successfully.
- `titlecard.acceleration_status()` now reports `{'decode': 'vaapi', ...}`.
- OCR status is still `None`/CPU: `onnxruntime` providers are only `AzureExecutionProvider` and `CPUExecutionProvider`.
- Host: i9-12900H, Iris Xe 96 EU; iGPU is shared with Plex/Frigate/Tdarr/Immich. `dmesg` has repeated HEVC GPU hangs from another consumer.
- `titlecard_ocr_device` is `auto`; code's auto path only checks CUDA.
- `titlecards` are currently disabled globally (`titlecard_enabled=false`).

## 2. Why this is not just “install openvino”

- `rapidocr_onnxruntime` 1.4.4's `OrtInferSession` builds a provider list of CPU, CUDA, and DirectML only.
- RapidOCR 2/3's native OpenVINO engine currently compiles every model with `device_name="CPU"` — installing OpenVINO alone would still run on CPU.
- Therefore the work is: a GPU-capable OCR backend plus a device-selection patch, system OpenCL/Level Zero libraries, and `/dev/dri` (already done).

## 3. Backend spike (do before the main implementation)

Create a throwaway image/venv based on the app image and test the same cached frame or a short Beavis clip:

### Spike A — `onnxruntime-openvino` with the existing wrapper

```bash
pip install --no-cache-dir openvino onnxruntime-openvino==1.24.1
python - <<'PY'
import onnxruntime as ort
print(ort.__version__, ort.get_available_providers())
PY
```

Then monkeypatch `rapidocr_onnxruntime.utils.infer_engine.OrtInferSession._get_ep_list`
to prepend:

```python
("OpenVINOExecutionProvider", {"device_type": "GPU"})
```

when `"OpenVINOExecutionProvider"` is in `get_available_providers()`. Confirm:
- OCR output matches CPU output on the same 20 frames (text and boxes; allow minor numeric jitter).
- `device_type: GPU` actually appears in the session providers.
- CPU fallback still works if the provider is unavailable.
- `onnxruntime-openvino` 1.24.1 is compatible with `rapidocr_onnxruntime` 1.4.4 and Python 3.12.

If this fails or downgrades break the wrapper, use Spike B.

### Spike B — RapidOCR 3.x OpenVINO engine with a device patch

```bash
pip install --no-cache-dir "rapidocr[openvino]>=3.10"
```

Inspect `rapidocr.inference_engine.openvino.main.OpenVINOInferSession`; it currently calls
`core.compile_model(model=self.model, device_name="CPU")`. Vendor a small subclass or patch:

```python
device = (cfg.get("engine_cfg") or {}).get("device_name", "GPU")
compiled = core.compile_model(model=self.model, device_name=device)
```

Map `AUTO`/`GPU`/`GPU.0`/`GPU.1`. Benchmark and compare output with the current CPU path. This
is the more official backend but a larger upgrade from `rapidocr_onnxruntime` 1.4.4.

**Decision gate:** choose the path with CPU-output parity, a working `GPU` provider/device, and
the smaller image/dependency change. Record the choice and versions in the plan’s execution log.

## 4. Implementation tasks

### Task 1 — Add the Intel GPU runtime packages to the image

**Files:** `Dockerfile`.

Add the OpenCL/Level Zero runtime, then `openvino` and the chosen OCR backend. On Debian slim,
verify package availability first; Intel’s documented set is:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
      ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero clinfo \
    && rm -rf /var/lib/apt/lists/*
```

If `intel-opencl-icd`/`level-zero` are not in the Debian slim repositories, add Intel’s apt
repository or install the matching Intel `.deb` packages. Do not ship an image that only works
because a host package happens to be present.

Do not install both `onnxruntime` and `onnxruntime-openvino`; the chosen backend determines which
one is present. If RapidOCR 3.x is chosen, keep `rapidocr` (not `rapidocr_onnxruntime`) as the
single OCR package.

**Verify:** `docker run --rm --device /dev/dri <image> clinfo -l` lists the Intel GPU; this check
is optional at runtime but required during build qualification.

### Task 2 — Add the OCR backend selector

**Files:** `app/config.py`, `app/titlecard.py`.

Add:

```python
"titlecard_ocr_device": ("auto", "TITLECARD_OCR_DEVICE", False),  # auto | cpu | openvino
"titlecard_openvino_device": ("GPU", "TITLECARD_OPENVINO_DEVICE", False),
```

`auto` order: NVIDIA CUDA (existing path) → OpenVINO GPU → CPU. `cpu` never probes a GPU.
`openvino` forces OpenVINO and fails over to CPU with a logged reason.

Add helpers in `titlecard.py`:

```python
def _openvino_gpu_available() -> bool:
    try:
        from openvino import Core
        return "GPU" in Core().available_devices
    except Exception:
        return False

def _openvino_device() -> str:
    return str(get_settings().get("titlecard_openvino_device") or "GPU")
```

`_engine()` sets `_accel["ocr"]` to `NVIDIA GPU (CUDA)`, `Intel GPU (OpenVINO: GPU)`, or `CPU`.
`acceleration_status()` reports the OCR device. Do not report “GPU” until a session was actually
created with the provider/device.

For the `onnxruntime-openvino` path, patch the provider list before `RapidOCR` constructs its
sessions. Keep the patch local to `_engine()` and restore it only if initialization fails; do not
mutate the global package permanently across a CPU fallback.

### Task 3 — GPU failure fallback

**Files:** `app/titlecard.py`.

Mirror the CUDA fallback in `_ocr_frame()`:

- If OpenVINO GPU initialization or inference raises, set `_accel["ocr_gpu_error"]`, set
  `force_cpu`, reset `_ocr`, and retry with CPU once.
- Log the last line of the exception (`error_summary`) and never crash the scan.
- A GPU failure must not poison a later CPU retry; reset both engine and provider patch state.

### Task 4 — Cache identity

**Files:** `app/titlecard.py`.

`OCR_SIG` currently identifies the OCR engine/frame preprocessing. Add the backend and device:
`rapidocr-<version>|openvino-<version>|GPU|640x360|det-min-...`. Bump the cache tag to `v3`.
This prevents a CPU OCR cache from being silently reused after the backend change and makes
CPU-vs-GPU benchmarks independent.

Do not invalidate on every process start; the device string and versions are enough.

### Task 5 — Configuration/UI

**Files:** `app/config.py`, `app/static/app.js`.

- `titlecard_ocr_device` options: `auto`, `cpu`, `openvino`.
- `titlecard_openvino_device` options: `GPU`, `GPU.0`, `GPU.1`, `AUTO` (Arc/iGPU selection).
- On a different host, resolve `getent group render` and update the compose `group_add` GID;
  `104` is specific to turnkey2. The Portainer stack compose is authoritative — the local
  `stacks/episodeid/docker-compose.yml` is a source copy, not what `deploy.py` PUTs.
- Settings hint: “OpenVINO uses the Intel iGPU/Arc; falls back to CPU if the GPU plugin or device
  is unavailable.”
- Show `acceleration_status()`’s OCR device in the title-card settings fieldset if the UI already
  has a test button; otherwise add it to the Test output.

### Task 6 — Tests

**Files:** `tests/titlecard_ocr_backend_test.py` (new), `tests/titlecard_ocr_parity_test.py` (new).

- Fake `onnxruntime.get_available_providers()` / fake `openvino.Core.available_devices`:
  - auto + CUDA provider → CUDA selected
  - auto + OpenVINO GPU only → OpenVINO selected
  - auto + neither → CPU
  - explicit `openvino` + no GPU → CPU fallback with reason
  - `cpu` + GPU available → CPU selected
- Provider patch restores the original provider list after a forced CPU fallback.
- Parity: run 20 cached title-card frames through CPU and GPU backends; assert normalized text
  matches and box coordinates are within a small tolerance. Skip GPU parity when no device.
- Cache tag changes when backend/device version changes; no mutation of existing CPU cache.

### Task 7 — Benchmark and acceptance

**Files:** `tools/titlecard_ocr_bench.py` (new).

Measure, on one 25-minute Beavis file and 20 frames:
- decode only: CPU ffmpeg vs `/dev/dri` VAAPI;
- OCR only: CPU vs OpenVINO GPU;
- end-to-end title-card pass (frame extraction + OCR) with the real `_sample`/`_ocr_frame` path;
- CPU utilization and wall time per file.

Acceptance:
- OCR text parity on the 20 frames; no lost/duplicated cards vs CPU baseline.
- At least a meaningful wall-time reduction (set the target after the spike; do not guess).
- No iGPU hang or Plex/Frigate/Immich regression during the test; if the shared iGPU is unstable,
  keep CPU or use the A310.
- Golden/plan behavior unchanged when title cards are disabled.

### Task 8 — Rollout

1. Build the image with the chosen backend; default `titlecard_ocr_device=auto` **but keep
   `titlecard_enabled=false`** until the benchmark passes.
2. Run the benchmark and a single-file title-card test.
3. Enable title cards for a small multi-episode subset; compare results to the CPU baseline.
4. Only then enable the setting globally.
5. If the iGPU is contended or hangs, set `titlecard_ocr_device=cpu` and revisit the A310 path.

## 5. A310 follow-up

The same OpenVINO code path supports the Arc A310. Differences:
- 4 GB GDDR6, 124 GB/s, 96 XMX engines / 52 INT8 TOPS, 2 codec engines; likely faster than the
  Iris Xe for OCR and decode, especially under contention.
- Needs PCIe 4.0 x8 in an x16 slot, 40–75 W, and Resizable BAR strongly recommended.
- If both iGPU and A310 are present, use `titlecard_openvino_device=GPU.1` or OpenVINO `AUTO`
  with a benchmark to confirm which device is selected.
- Do not buy the A310 until the iGPU OpenVINO benchmark shows OCR remains the bottleneck.

## 6. Risks

- **RapidOCR OpenVINO backend hardcodes CPU.** The patch is mandatory; verify it in the chosen
  version rather than assuming a config flag exists.
- **`onnxruntime-openvino` version conflict.** It may downgrade `onnxruntime` from 1.31.1 to
  1.24.1; test `rapidocr_onnxruntime` and OCR parity before committing.
- **Debian slim packages.** Intel OpenCL/Level Zero may need Intel’s apt repository; the image
  must be self-contained, not dependent on host libraries.
- **Shared iGPU.** Plex/Frigate/Tdarr/Immich already use it; HEVC hangs have been observed. A GPU
  OCR failure must fall back to CPU, and a benchmark must check other containers.
- **Cache mixing.** Bump `OCR_SIG` when the backend/device changes; never reuse CPU OCR JSON as if
  it were a GPU result.
- **No `device_name=GPU` guarantee.** OpenVINO `AUTO` may choose CPU; log the actual compiled
  device, not just the requested one.

## 7. Files likely to change

- `Dockerfile`
- `app/config.py`
- `app/titlecard.py`
- `app/static/app.js`
- `tests/titlecard_ocr_backend_test.py` (new)
- `tests/titlecard_ocr_parity_test.py` (new)
- `tools/titlecard_ocr_bench.py` (new)
- `docs/plans/2026-11-12-openvino-titlecard-ocr.md` (this plan)
