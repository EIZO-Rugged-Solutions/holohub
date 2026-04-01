
# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /tmp/scripts
COPY holohub /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/holohub
RUN /tmp/scripts/holohub setup && rm -rf /var/lib/apt/lists/*

RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

ARG CMAKE_BUILD_TYPE=Release

ARG GPU_TYPE
ARG CUDA_MAJOR
RUN INDEX_URL=""; \
    if [ "${GPU_TYPE}" = "igpu" ]; then \
        apt-get update && \
        apt-get install -y --reinstall --no-install-recommends \
            libcudnn9-cuda-12 \
            libcudnn9-dev-cuda-12 && \
        rm -rf /var/lib/apt/lists/* && \
        python3 -m pip list | grep -i torch | awk '{print $1}' | xargs -r python3 -m pip uninstall -y && \
        INDEX_URL="https://pypi.jetson-ai-lab.io/jp6/cu126"; \
    elif [ "${CUDA_MAJOR}" = "12" ]; then \
        INDEX_URL="https://download.pytorch.org/whl/cu124"; \
    elif [ "${CUDA_MAJOR}" = "13" ]; then \
        INDEX_URL="https://download.pytorch.org/whl/cu130"; \
    else \
        echo "Error: Unsupported GPU_TYPE (${GPU_TYPE}) or CUDA_MAJOR (${CUDA_MAJOR})" && exit 1; \
    fi \
    && python3 -m pip install --no-cache-dir setuptools \
    && python3 -m pip install --no-cache-dir torch torchvision --index-url ${INDEX_URL}

COPY applications/qwen2_5_vl_3b_local/requirements.txt /tmp/requirements.txt
RUN sed -i '/holoscan/d' /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt --no-cache-dir

# Download sample video from Pexels and trim to 6.4 seconds
# Stored outside /workspace/holohub so it survives the volume mount at runtime
RUN mkdir -p /opt/holohub_data/qwen2_5_vl_3b_local \
    && curl -S -L -o /tmp/video.mp4 \
       "https://www.pexels.com/download/video/19326803/?fps=29.97&h=720&w=1280" \
    && ffmpeg -y -i /tmp/video.mp4 -t 6.4 -c:v libx264 \
       /opt/holohub_data/qwen2_5_vl_3b_local/traffic.mp4 \
    && rm -f /tmp/video.mp4

WORKDIR /workspace/holohub
