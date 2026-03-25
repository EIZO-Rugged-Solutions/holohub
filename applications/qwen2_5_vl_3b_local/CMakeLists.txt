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

cmake_minimum_required(VERSION 3.20)
project(qwen2_5_vl_3b_local LANGUAGES NONE)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

set(QWEN3_VL_SAMPLE_DIR "${HOLOHUB_DATA_DIR}/qwen2_5_vl_3b_local")
set(QWEN3_VL_SAMPLE_MP4 "${QWEN3_VL_SAMPLE_DIR}/av_construction_stop_timestamped.mp4")
set(QWEN3_VL_SAMPLE_BASENAME "av_construction_stop_timestamped")

add_custom_command(
  OUTPUT "${QWEN3_VL_SAMPLE_DIR}/${QWEN3_VL_SAMPLE_BASENAME}.gxf_index"
         "${QWEN3_VL_SAMPLE_DIR}/${QWEN3_VL_SAMPLE_BASENAME}.gxf_entities"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${QWEN3_VL_SAMPLE_DIR}"
  COMMAND ffmpeg -i "${QWEN3_VL_SAMPLE_MP4}" -pix_fmt rgb24 -f rawvideo pipe:1 |
          python3 "${CMAKE_SOURCE_DIR}/utilities/convert_video_to_gxf_entities.py"
          --directory "${QWEN3_VL_SAMPLE_DIR}"
          --basename "${QWEN3_VL_SAMPLE_BASENAME}"
          --width 1280
          --height 752
          --framerate 30
  DEPENDS "${QWEN3_VL_SAMPLE_MP4}"
  VERBATIM
)

add_custom_target(
  qwen2_5_vl_3b_local ALL
  DEPENDS "${QWEN3_VL_SAMPLE_DIR}/${QWEN3_VL_SAMPLE_BASENAME}.gxf_index"
)
