ARG ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
FROM ${ISAAC_LAB_IMAGE}

ARG WORKSHOP_REPOSITORY=https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop.git
ARG WORKSHOP_COMMIT=ce807d99724cb65671abec01f908a2fcb4a6eab7
ARG LEROBOT_COMMIT=e670ac5daf9b76

LABEL org.opencontainers.image.title="Project Subliminal Stage 1 generator" \
      org.opencontainers.image.description="Pinned Isaac Lab SO-101 data-generation runtime" \
      project.subliminal.isaac-lab="2.3.2" \
      project.subliminal.workshop-commit="${WORKSHOP_COMMIT}" \
      project.subliminal.lerobot-commit="${LEROBOT_COMMIT}"

ENV ISAAC_PYTHON=/workspace/isaaclab/_isaac_sim/python.sh \
    ISAAC_SIM=/workspace/isaaclab/_isaac_sim \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    PYTHONUNBUFFERED=1 \
    OMNI_KIT_ACCEPT_EULA=YES

RUN apt-get update && apt-get install -y --no-install-recommends \
    git-lfs \
    libx11-6 \
    libxcursor1 \
    libxinerama1 \
    libxi6 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxrandr2 \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone "${WORKSHOP_REPOSITORY}" Sim-to-Real-SO-101-Workshop \
    && git -C Sim-to-Real-SO-101-Workshop checkout "${WORKSHOP_COMMIT}" \
    && git -C Sim-to-Real-SO-101-Workshop lfs pull \
    && test "$(stat -c %s Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd)" -gt 1000

RUN git clone https://github.com/huggingface/lerobot.git lerobot \
    && git -C lerobot checkout "${LEROBOT_COMMIT}" \
    && "${ISAAC_PYTHON}" -m pip install --no-deps -e /workspace/lerobot

# Preserve Isaac Lab's vendored binary stack while adding only the video encoder.
RUN printf '%s\n' \
    'packaging==23.0' \
    'numpy==1.26.0' \
    'lxml==4.9.4' \
    'torch==2.7.0' \
    'torchvision==0.22.0' \
    'imageio==2.37.0' \
    > /tmp/isaac-constraints.txt \
    && "${ISAAC_PYTHON}" -m pip install -c /tmp/isaac-constraints.txt 'av>=15.0.0,<16.0.0' \
    && "${ISAAC_PYTHON}" -m pip install --no-deps -e /workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101

COPY docker/stage1/entrypoint.sh /opt/project-subliminal/stage1-entrypoint.sh
RUN chmod 0755 /opt/project-subliminal/stage1-entrypoint.sh \
    && mkdir -p /tmp/pycache /workspace/output \
    && chmod 1777 /tmp/pycache

WORKDIR /workspace/subliminal
ENTRYPOINT ["/opt/project-subliminal/stage1-entrypoint.sh"]
CMD ["bash"]
