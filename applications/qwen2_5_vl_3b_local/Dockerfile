
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

COPY applications/qwen2_5_vl_3b_local/requirements.txt /tmp/requirements.txt
RUN sed -i '/holoscan/d' /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir setuptools && \
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install -r /tmp/requirements.txt --no-cache-dir

WORKDIR /workspace/holohub
