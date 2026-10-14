# kcoral:rocm-dev with libhsa-runtime64 rebuilt from ROCR-Runtime plus the
# libhsakmt patch in this directory. Build from the repository root:
#   docker build -f docker/rocm-libhsakmt-patch/Dockerfile -t kcoral:rocm-dev-patched .
# ROCR_TAG must match the ROCm release the image's torch was built for.
ARG BASE=kcoral:rocm-dev
FROM ${BASE}

ARG ROCR_TAG=rocm-7.0.0

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        cmake git libdrm-dev libelf-dev libnuma-dev pkg-config rocm-llvm-dev xxd \
    && rm -rf /var/lib/apt/lists/*

COPY docker/rocm-libhsakmt-patch/0001-libhsakmt-skip-memory-policy-for-unusable-gpus.patch /tmp/rocr.patch
COPY docker/rocm-libhsakmt-patch/install_patched_rocr.sh /usr/local/bin/install_patched_rocr.sh

RUN git clone --depth 1 --branch "${ROCR_TAG}" https://github.com/ROCm/ROCR-Runtime.git /tmp/rocr \
    && cd /tmp/rocr && git apply /tmp/rocr.patch \
    && grep -q "skip GPUs the kernel" libhsakmt/src/fmm.c \
    && cmake -S /tmp/rocr -B /tmp/rocr/build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=/tmp/rocr/install \
        -DBUILD_SHARED_LIBS=ON \
    && cmake --build /tmp/rocr/build -j"$(nproc)" \
    && cmake --install /tmp/rocr/build \
    && bash /usr/local/bin/install_patched_rocr.sh /tmp/rocr/install \
    && rm -rf /tmp/rocr /tmp/rocr.patch
