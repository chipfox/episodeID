# EpisodeID — CPU image (Whisper runs on CPU with int8).
# For NVIDIA GPUs build Dockerfile.cuda instead.
# Intel/AMD integrated graphics: pass /dev/dri to the container and title-card
# video decoding uses VAAPI automatically.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MEDIA_ROOT=/media CONFIG_DIR=/config CACHE_DIR=/cache \
    HF_HOME=/cache/huggingface PUID=99 PGID=100 UMASK=002 PORT=8686

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tini libgl1 libglib2.0-0 \
      intel-media-va-driver mesa-va-drivers \
      ocl-icd-libopencl1 libze1 curl ca-certificates clinfo \
 && rm -rf /var/lib/apt/lists/*

# Intel user-space compute stack for OpenVINO GPU OCR (Iris Xe / Arc).
# Pinned to the versions validated on the Iris Xe; the OpenCL/Level Zero
# runtime cannot be installed from the Debian repositories used by this image.
ARG INTEL_COMPUTE=26.40.40196.8
ARG INTEL_IGC=2.42.6
RUN set -eux; \
    base="https://github.com/intel/compute-runtime/releases/download/${INTEL_COMPUTE}"; \
    igc="https://github.com/intel/intel-graphics-compiler/releases/download/v${INTEL_IGC}"; \
    curl -fsSL -o /tmp/libigdgmm12.deb "${base}/libigdgmm12_22.10.2_amd64.deb"; \
    curl -fsSL -o /tmp/intel-igc-core-2.deb "${igc}/intel-igc-core-2_${INTEL_IGC}%2B23070_amd64.deb"; \
    curl -fsSL -o /tmp/intel-igc-opencl-2.deb "${igc}/intel-igc-opencl-2_${INTEL_IGC}%2B23070_amd64.deb"; \
    curl -fsSL -o /tmp/libze-intel-gpu1.deb "${base}/libze-intel-gpu1_${INTEL_COMPUTE}-0_amd64.deb"; \
    curl -fsSL -o /tmp/intel-opencl-icd.deb "${base}/intel-opencl-icd_${INTEL_COMPUTE}-0_amd64.deb"; \
    curl -fsSL -o /tmp/intel-ocloc.deb "${base}/intel-ocloc_${INTEL_COMPUTE}-0_amd64.deb"; \
    dpkg -i /tmp/libigdgmm12.deb /tmp/intel-igc-core-2.deb /tmp/intel-igc-opencl-2.deb \
      /tmp/libze-intel-gpu1.deb /tmp/intel-opencl-icd.deb /tmp/intel-ocloc.deb; \
    rm -f /tmp/*.deb; ldconfig

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y onnxruntime \
 && pip install --no-cache-dir "onnxruntime-openvino==1.24.1" "openvino==2026.4.1"
# Which build this is — shown next to the version in the web UI's header (set by CI;
# placed here so changing it doesn't throw away the pip layer above).
ARG GIT_SHA=""
ARG BUILD_DATE=""
ENV EPISODEID_COMMIT=$GIT_SHA EPISODEID_BUILT=$BUILD_DATE
COPY app ./app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/config", "/cache"]
EXPOSE 8686
HEALTHCHECK --interval=60s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8686\")}/api/status',timeout=4)"
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
