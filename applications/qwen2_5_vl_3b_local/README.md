# Qwen2.5-VL Local Video Understanding

This application demonstrates a minimal Holoscan application that runs [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) locally with `transformers`, using a local model directory and a local MP4 file. It does not call any API at runtime and does not require internet access or API keys once the assets are staged locally.

The default sample video is a traffic clip downloaded from [Pexels](https://www.pexels.com/video/19326803/) during the Docker build and stored in the container image.

## What It Does

- Loads a local Qwen2.5-VL model from disk
- Resolves a local video input from a local file or the default sample path
- Runs local video reasoning with `transformers`
- Prints the model response

## Quick Start

1. Download the model weights (one-time, see below).
2. Build the application:

   ```bash
   ./holohub build qwen2_5_vl_3b_local
   ```

3. Run it:

   ```bash
   ./holohub run qwen2_5_vl_3b_local
   ```

The default configuration expects the model at `/workspace/holohub/data/models/Qwen/Qwen2.5-VL-3B-Instruct`. The sample video is baked into the Docker image at `/opt/holohub_data/qwen2_5_vl_3b_local/traffic.mp4` and copied into the data directory during the CMake build.

## Download The Model Once

Reference model:

- <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>

The model is published under the Apache-2.0 license. Download the weights once, then run fully offline.

**Inside the container:**

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /workspace/holohub/data/models/Qwen/Qwen2.5-VL-3B-Instruct
```

**On the host (before building):**

From the root of this repository:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir data/models/Qwen/Qwen2.5-VL-3B-Instruct
```

## Configuration

Edit [`nvidia_nim.yaml`](./nvidia_nim.yaml) to change the local model path, prompt, and sampling parameters.

Important fields:

- `app.model_path`: local model directory
- `app.prompt`: prompt sent with the video
- `app.video_fps`: video sampling rate used for video loading
- `app.video_num_frames`: preferred way to cap VRAM usage on smaller GPUs
- `app.sample_video_path`: local sample video path
- `app.tensor_dtype`: use `float16` on tighter VRAM budgets
- `app.quantization`: leave this as `none` by default for the 3B model; `4bit` is optional if you want to experiment

## Running With Another Video

Use a local file:

```bash
./holohub run qwen2_5_vl_3b_local --run-args="--video-file /path/to/your/video.mp4"
```

Override the prompt:

```bash
./holohub run qwen2_5_vl_3b_local --run-args="--prompt 'Describe the worker actions and any hazards in this clip.'"
```

Override the model path:

```bash
./holohub run qwen2_5_vl_3b_local --run-args="--model-path /workspace/holohub/data/models/Qwen/Qwen2.5-VL-3B-Instruct"
```

## References

- Hugging Face model card: <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>
