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

import argparse
import logging
import os
import pathlib
import threading
import time
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cupy as cp
import numpy as np
import torch
import transformers
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, HolovizOp, VideoStreamReplayerOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator
from PIL import Image

logger = logging.getLogger("QWEN_VL_APP")
logging.basicConfig(level=logging.INFO)


def valid_existing_path(path: str) -> pathlib.Path:
    expanded = os.path.expanduser(path)
    candidate = pathlib.Path(expanded).absolute()
    if candidate.exists():
        return candidate
    raise argparse.ArgumentTypeError(f"No such file or directory: '{candidate}'")


def _resolve_dtype(dtype_name: str):
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported tensor dtype: {dtype_name}")
    return mapping[dtype_name]


class OverlayState:
    def __init__(self, initial_text: str):
        self._status = "info"
        self._text = initial_text
        self._started_at = None
        self._lock = threading.Lock()

    def set_info(self, text: str):
        with self._lock:
            self._status = "info"
            self._text = text
            self._started_at = None

    def set_generating(self, text: str):
        with self._lock:
            self._status = "generating"
            self._text = text
            self._started_at = time.monotonic()

    def set_response(self, text: str):
        with self._lock:
            self._status = "response"
            self._text = text
            self._started_at = None

    def set_error(self, text: str):
        with self._lock:
            self._status = "error"
            self._text = text
            self._started_at = None

    def get_display_text(self) -> str:
        with self._lock:
            if self._status == "generating" and self._started_at is not None:
                elapsed = max(0, int(time.monotonic() - self._started_at))
                return f"{self._text}\nElapsed: {elapsed}s"
            return self._text


class LiveQwen3VLOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        model_path: str,
        prompt: str,
        system_prompt: str,
        query_interval_sec: float,
        max_tokens: int,
        temperature: float,
        tensor_dtype: str,
        attn_implementation: str,
        quantization: str,
        overlay_state: OverlayState,
        **kwargs,
    ):
        self.model_path = pathlib.Path(model_path).expanduser().absolute()
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.query_interval_sec = float(query_interval_sec)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tensor_dtype = tensor_dtype
        self.attn_implementation = attn_implementation
        self.quantization = quantization
        self.overlay_state = overlay_state
        self._last_query_time = 0.0
        self._busy = threading.Event()
        self._model = None
        self._processor = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("video_stream")

    def start(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for local Qwen VL inference.")
        model_dtype = _resolve_dtype(self.tensor_dtype)
        logger.info("Loading local model from %s", self.model_path)
        self.overlay_state.set_info("Loading Qwen2.5-VL-3B model...")
        torch.cuda.empty_cache()
        model_kwargs = {
            "device_map": "auto",
            "attn_implementation": self.attn_implementation,
            "local_files_only": True,
        }
        if self.quantization == "4bit":
            model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=model_dtype,
            )
        else:
            model_kwargs["dtype"] = model_dtype
        self._model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        self._processor = transformers.AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.overlay_state.set_info("Waiting for first response...")

    def _wrap_text(self, text: str, width: int = 56) -> str:
        words = text.split()
        if not words:
            return ""
        lines = []
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) <= width:
                current = f"{current} {word}"
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return "\n".join(lines[:8])

    def _run_inference(self, frame_array: np.ndarray):
        self._busy.set()
        try:
            pil_image = Image.fromarray(frame_array)
            pil_image.thumbnail((448, 448), Image.Resampling.BICUBIC)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ],
                },
            ]
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(text=[text], images=[pil_image], return_tensors="pt")
            inputs = inputs.to(self._model.device)
            torch.cuda.empty_cache()

            logger.info("Running local generation")
            generation_kwargs = {
                "max_new_tokens": self.max_tokens,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                generation_kwargs["temperature"] = self.temperature
            generated_ids = self._model.generate(**inputs, **generation_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
            ]
            response_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            self.overlay_state.set_response(response_text)

            timestamp = datetime.now().isoformat(timespec="seconds")
            print(f"\n[{timestamp}] Prompt:\n")
            print(self.prompt)
            print(f"\n[{timestamp}] Response:\n")
            print(response_text)
        except Exception:
            self.overlay_state.set_error("Generation failed. Check container logs.")
            logger.exception("Inference thread failed")
        finally:
            self._busy.clear()

    def compute(self, op_input, op_output, context):
        try:
            in_message = op_input.receive("video_stream")
            if in_message is None:
                return

            now = time.monotonic()
            if self._busy.is_set() or (now - self._last_query_time) < self.query_interval_sec:
                return

            tensor = None
            if hasattr(in_message, "get"):
                tensor = in_message.get("image")
                if tensor is None:
                    tensor = in_message.get("")
            else:
                tensor = in_message
            if tensor is None:
                logger.warning(
                    "No tensor found in video_stream message of type %s", type(in_message)
                )
                return
            self.overlay_state.set_generating("Generating response...")
            frame_array = cp.asnumpy(cp.from_dlpack(tensor)).copy()
            self._last_query_time = now
            thread = threading.Thread(target=self._run_inference, args=(frame_array,), daemon=True)
            thread.start()
        except Exception:
            logger.exception("Compute failed")
            raise


class TextOverlayOp(Operator):
    def __init__(self, fragment, *args, overlay_state: OverlayState, **kwargs):
        self.overlay_state = overlay_state
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("outputs")
        spec.output("output_specs")

    def _wrap_text(self, text: str, width: int = 56) -> str:
        words = text.split()
        if not words:
            return ""
        lines = []
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) <= width:
                current = f"{current} {word}"
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return "\n".join(lines[:8])

    def compute(self, op_input, op_output, context):
        current_text = self.overlay_state.get_display_text()
        out_message = {"qwen_caption": np.asarray([(0.03, 0.08)], dtype=np.float32)}
        spec = HolovizOp.InputSpec("qwen_caption", "text")
        spec.text = [self._wrap_text(current_text)]
        spec.color = [1.0, 1.0, 0.0, 1.0]
        spec.priority = 10
        op_output.emit(out_message, "outputs")
        op_output.emit([spec], "output_specs")


class QwenVLVideoApp(Application):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def compose(self):
        app_config = self.kwargs("app")
        replayer_config = self.kwargs("replayer_source")
        holoviz_config = self.kwargs("holoviz")
        overlay_state = OverlayState("Initializing...")

        sample_data_dir = os.path.dirname(app_config["sample_video_path"])
        replayer = VideoStreamReplayerOp(
            self,
            name="replayer_source",
            directory=sample_data_dir,
            **replayer_config,
        )

        visualizer = HolovizOp(
            self,
            name="holoviz",
            allocator=UnboundedAllocator(self, name="holoviz_allocator"),
            enable_render_buffer_input=False,
            **holoviz_config,
        )

        formatter_cuda_stream_pool = CudaStreamPool(
            self,
            name="format_cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        format_converter = FormatConverterOp(
            self,
            name="convert_video_to_tensor",
            in_dtype="rgb888",
            out_dtype="rgb888",
            cuda_stream_pool=formatter_cuda_stream_pool,
            pool=UnboundedAllocator(self, name="format_converter_allocator"),
        )

        qwen = LiveQwen3VLOp(
            self,
            name="query",
            model_path=self.args.model_path or app_config["model_path"],
            prompt=self.args.prompt or app_config["prompt"],
            system_prompt=app_config["system_prompt"],
            query_interval_sec=float(app_config["query_interval_sec"]),
            max_tokens=int(app_config["max_tokens"]),
            temperature=float(app_config["temperature"]),
            tensor_dtype=app_config["tensor_dtype"],
            attn_implementation=app_config["attn_implementation"],
            quantization=app_config["quantization"],
            overlay_state=overlay_state,
        )

        text_overlay = TextOverlayOp(
            self,
            PeriodicCondition(self, name="overlay_tick", recess_period=0.1),
            name="text_overlay",
            overlay_state=overlay_state,
        )

        self.add_flow(replayer, format_converter, {("output", "source_video")})
        self.add_flow(format_converter, visualizer, {("tensor", "receivers")})
        self.add_flow(format_converter, qwen, {("tensor", "video_stream")})
        self.add_flow(text_overlay, visualizer, {("outputs", "receivers")})
        self.add_flow(text_overlay, visualizer, {("output_specs", "input_specs")})


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL-3B-Instruct live video reasoning sample"
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        default=os.path.join(os.path.dirname(__file__), "nvidia_nim.yaml"),
        type=valid_existing_path,
        dest="config",
        help="Application configuration file",
    )
    parser.add_argument(
        "--model-path", default=None, help="Override the configured local model directory."
    )
    parser.add_argument("--prompt", default=None, help="Override the configured prompt.")
    args = parser.parse_args()

    app = QwenVLVideoApp(args)
    app.config(str(args.config))
    app.run()


if __name__ == "__main__":
    main()
