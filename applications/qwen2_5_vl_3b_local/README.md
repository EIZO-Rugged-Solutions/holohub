# Qwen2.5-VL Local Video Understanding

This application demonstrates a minimal Holoscan application that runs [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) locally with `transformers`, using a local model directory and a local MP4 file. It does not call any API at runtime and does not require internet access or API keys once the assets are staged locally.

The default sample video path matches the sample clip shown on NVIDIA's deploy page:

- `data/qwen2_5_vl_3b_local/av_construction_stop_timestamped.mp4`

The clip itself corresponds to:

- `https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/av_construction_stop_timestamped.mp4`

## What It Does

- Loads a local Qwen2.5-VL model from disk
- Resolves a local video input from a local file or the default sample path
- Runs local video reasoning with `transformers`
- Prints the model response and saves it to `qwen2_5_vl_3b_response.txt`

## Quick Start

1. Stage the model weights locally.
2. Stage the sample video locally.
3. Build the application:

   ```bash
   ./holohub build qwen2_5_vl_3b_local
   ```

4. Run it:

   ```bash
   ./holohub run qwen2_5_vl_3b_local
   ```

The default configuration expects the model at `/workspace/holohub/data/models/Qwen/Qwen2.5-VL-3B-Instruct` and the sample video at `/workspace/holohub/data/qwen2_5_vl_3b_local/av_construction_stop_timestamped.mp4`.

## One-Time Staging

This app is offline at runtime. `Qwen/Qwen2.5-VL-3B-Instruct` is published on Hugging Face under the Apache-2.0 license, so you can stage the weights once and then run fully offline.

That means:

- You need a one-time model download step.
- After the weights and video are staged locally, runtime requires no API calls, no API keys, and no internet.

Suggested staging locations:

- Model: `/workspace/holohub/data/models/Qwen/Qwen2.5-VL-3B-Instruct`
- Video: `/workspace/holohub/data/qwen2_5_vl_3b_local/av_construction_stop_timestamped.mp4`

## Download The Model Once

Reference model:

- <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>

The Hugging Face model card shows local inference with:

- `transformers.Qwen2_5_VLForConditionalGeneration`
- `transformers.AutoProcessor`
- `device_map="auto"`

Example one-time staging:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /workspace/holohub/data/models/Qwen/Qwen2.5-VL-3B-Instruct
```

If you run the download on the host instead of inside the container, use the host path for this repo:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /home/test/projects/tradeshow/vlm/holohub/data/models/Qwen/Qwen2.5-VL-3B-Instruct
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
./holohub run qwen2_5_vl_3b_local --run-args="--video-file /workspace/holohub/data/qwen2_5_vl_3b_local/av_construction_stop_timestamped.mp4"
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
