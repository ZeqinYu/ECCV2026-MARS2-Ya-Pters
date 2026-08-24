#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TaRO-8B + MARS2 VTG 赛道 2

运行：
    python3 TempoFlex.py --help

功能：TOTAL_VIDEO_TOKENS
    - 读取 VTG_QA.jsonl
    - 使用 TaRO-8B 官方 Prompt / FPS / 视频 token 设置
    - 使用 vLLM 推理
    - 边推理边保存 raw_results.jsonl
    - 支持断点续跑
    - 自动解析 <answer>...</answer> 中的时间区间
    - 输出比赛要求的 out.jsonl：
        {"id": "...", "model_prediction": "12.3-45.6"}
"""

# =============================================================================
# 0. 命令行参数（必须在导入 torch/vLLM 前解析）
# =============================================================================

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TaRO-8B inference for MARS2 Track 2: VTG"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the local TaRO-8B model directory.",
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Path to the official VTG_QA.jsonl file.",
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing videos named <id>.mp4.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which inference outputs will be written.",
    )
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="Physical GPU ID exposed to this process. Default: 0.",
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of samples to process; 0 processes all samples.",
    )
    parser.add_argument(
        "--video-ext",
        default=".mp4",
        help="Video filename extension. Default: .mp4.",
    )
    return parser.parse_args()


ARGS = parse_args()

GPU_ID = str(ARGS.gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

if ARGS.hf_home:
    os.environ["HF_HOME"] = str(
        Path(ARGS.hf_home).expanduser().resolve()
    )

# =============================================================================
# 1. Imports
# =============================================================================

import gc
import json
import math
import re
import shutil
import subprocess
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


# =============================================================================
# 2. 路径参数与固定推理参数
# =============================================================================

MODEL_PATH = str(Path(ARGS.model_path).expanduser().resolve())
INPUT_JSONL = str(Path(ARGS.input_jsonl).expanduser().resolve())
VIDEO_DIR = str(Path(ARGS.video_dir).expanduser().resolve())
EXPERIMENT_DIR = str(Path(ARGS.output_dir).expanduser().resolve())
LIMIT = ARGS.limit
VIDEO_EXT = ARGS.video_ext
if not VIDEO_EXT.startswith("."):
    VIDEO_EXT = "." + VIDEO_EXT

# TaRO 官方 demo设置：FPS和Prompt保持不变。
FPS = 5
MAX_FRAMES = 1024

# 每秒约分配 528 个视觉 token，总预算限制在 [11264, 18432]。
ADAPTIVE_VIDEO_TOKENS = True
BASE_TOTAL_VIDEO_TOKENS = 11264
MAX_TOTAL_VIDEO_TOKENS = 18432
VIDEO_TOKENS_PER_SECOND = 528
VIDEO_TOKEN_ROUND = 256

# vLLM 设置。上下文长度为最大 18432 视觉 token 和文本生成留出余量。
MAX_MODEL_LEN = 28672
MAX_NUM_BATCHED_TOKENS = 32768
GPU_MEMORY_UTILIZATION = 0.80
MAX_NEW_TOKENS = 1024

# 每完成多少条，重建一次当前 out.jsonl。
SAVE_OUT_EVERY = 10

# 解析或视频失败时，为保证最终提交行数完整，使用整段视频兜底。
USE_FULL_VIDEO_FALLBACK = True

# 官方 Prompt，保持不改。
PROMPT = """To accurately pinpoint the event "{query}" in the video, determine the precise time period of the event.

Output your thought process within the <think> </think> tags, including analysis with specific time ranges (xx.xx to xx.xx).

Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example:

<think>
I need to find the specific moment a person eats from a box.

From 0.0s to 5.2s, a person is holding a box of food and talking to the camera, but they are not eating yet.

From 5.3s to 8.1s, the person opens the box and looks inside. Still no eating action.

At 8.2s, the person picks up a piece of food from the box.

From 8.5s to 12.4s, the person puts the food in their mouth and chews while holding the box. This clearly matches "eating from a box."

From 12.5s to 15.0s, the person puts the box down and wipes their mouth. The action has ended.

Therefore, the relevant segment starts when the food approaches the mouth and ends when the chewing action concludes or the box is lowered.
</think>

<answer>
8.50s to 12.40s
</answer>"""


# =============================================================================
# 3. 输出路径
# =============================================================================

EXPERIMENT_PATH = Path(EXPERIMENT_DIR)
RAW_JSONL = EXPERIMENT_PATH / "raw_results.jsonl"
ERROR_JSONL = EXPERIMENT_PATH / "errors.jsonl"
OUT_JSONL = EXPERIMENT_PATH / "out.jsonl"
RUN_CONFIG_JSON = EXPERIMENT_PATH / "run_config.json"


# =============================================================================
# 4. 基础读写
# =============================================================================

def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"JSONL line {line_no} parse failed: {e}"
                ) from e

            if not isinstance(obj, dict):
                raise RuntimeError(
                    f"JSONL line {line_no} is not a JSON object"
                )

            rows.append(obj)

    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


# =============================================================================
# 5. 输入检查与视频时长
# =============================================================================

def validate_input(items: List[dict]) -> None:
    if not items:
        raise RuntimeError("VTG_QA.jsonl is empty")

    seen = set()

    for index, item in enumerate(items, 1):
        if "id" not in item or "question" not in item:
            raise RuntimeError(
                f"Input row {index} lacks 'id' or 'question'"
            )

        video_id = str(item["id"])

        if video_id in seen:
            raise RuntimeError(f"Duplicate input id: {video_id}")

        seen.add(video_id)


def check_environment() -> None:
    if not os.path.isdir(MODEL_PATH):
        raise NotADirectoryError(f"Model directory not found: {MODEL_PATH}")

    if not os.path.isfile(os.path.join(MODEL_PATH, "config.json")):
        raise FileNotFoundError(
            f"config.json not found in model directory: {MODEL_PATH}"
        )

    if not os.path.isfile(INPUT_JSONL):
        raise FileNotFoundError(f"Input JSONL not found: {INPUT_JSONL}")

    if not os.path.isdir(VIDEO_DIR):
        raise NotADirectoryError(f"Video directory not found: {VIDEO_DIR}")

    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffprobe not found. Install ffmpeg first, e.g. "
            "`conda install -c conda-forge ffmpeg -y`."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    if SAVE_OUT_EVERY <= 0:
        raise ValueError("SAVE_OUT_EVERY must be greater than 0")

    if LIMIT < 0:
        raise ValueError("--limit must be 0 or a positive integer")


def ffprobe_duration(video_path: str) -> float:
    commands = [
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
    ]

    for cmd in commands:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            continue

        values: List[float] = []

        for line in result.stdout.splitlines():
            line = line.strip()

            if not line or line.upper() == "N/A":
                continue

            try:
                value = float(line)
            except ValueError:
                continue

            if math.isfinite(value) and value > 0:
                values.append(value)

        if values:
            return max(values)

    raise RuntimeError(
        f"ffprobe could not obtain duration: {video_path}"
    )


# =============================================================================
# 6. TaRO 输出解析
# =============================================================================

NUMBER = r"-?\d+(?:\.\d+)?"

ANSWER_TAG_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)

PAIR_PATTERNS = [
    re.compile(
        rf"({NUMBER})\s*s?\s*(?:to|-|–|—)\s*({NUMBER})\s*s?",
        flags=re.IGNORECASE,
    ),
    re.compile(
        rf"(?:start(?:ing)?(?:\s+time)?\s*[:=]?\s*)({NUMBER}).*?"
        rf"(?:end(?:ing)?(?:\s+time)?\s*[:=]?\s*)({NUMBER})",
        flags=re.IGNORECASE | re.DOTALL,
    ),
]


def normalize_window(
    start: float,
    end: float,
    duration: float,
) -> Optional[Tuple[float, float]]:
    if not math.isfinite(start) or not math.isfinite(end):
        return None

    if start > end:
        start, end = end, start

    start = max(0.0, start)
    end = max(0.0, end)

    if duration > 0:
        start = min(start, duration)
        end = min(end, duration)

    if end <= start:
        return None

    return start, end


def parse_answer(
    output_text: str,
    duration: float,
) -> Optional[Tuple[float, float]]:
    if not isinstance(output_text, str):
        return None

    # 优先只解析 <answer> 标签，避免误取 <think> 里的中间时间。
    answer_match = ANSWER_TAG_PATTERN.search(output_text)

    if answer_match:
        candidates = [answer_match.group(1)]
    else:
        # 模型偶尔可能缺失标签，才回退到全文解析。
        candidates = [output_text]

    for candidate in candidates:
        for pattern in PAIR_PATTERNS:
            match = pattern.search(candidate)

            if not match:
                continue

            try:
                start = float(match.group(1))
                end = float(match.group(2))
            except (TypeError, ValueError):
                continue

            normalized = normalize_window(start, end, duration)

            if normalized is not None:
                return normalized

        numbers = re.findall(NUMBER, candidate)

        if len(numbers) >= 2:
            try:
                start = float(numbers[0])
                end = float(numbers[1])
            except (TypeError, ValueError):
                continue

            normalized = normalize_window(start, end, duration)

            if normalized is not None:
                return normalized

    return None


# =============================================================================
# 6.5 自适应视频token预算
# =============================================================================

def choose_total_video_tokens(duration: float) -> int:
    """根据视频时长选择总视觉 token 预算。"""
    if not ADAPTIVE_VIDEO_TOKENS:
        return BASE_TOTAL_VIDEO_TOKENS

    raw_tokens = VIDEO_TOKENS_PER_SECOND * max(0.0, float(duration))
    rounded_tokens = int(
        round(raw_tokens / VIDEO_TOKEN_ROUND) * VIDEO_TOKEN_ROUND
    )

    return max(
        BASE_TOTAL_VIDEO_TOKENS,
        min(MAX_TOTAL_VIDEO_TOKENS, rounded_tokens),
    )


# =============================================================================
# 7. 模型加载
# =============================================================================

def load_model_and_processor():
    print("=" * 80)
    print("[INFO] Loading TaRO-8B")
    print(f"[INFO] Model: {MODEL_PATH}")
    print(f"[INFO] Physical GPU: {GPU_ID}; visible device: cuda:0")
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_mm_preprocessor_cache=True,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )

    sampling_params = SamplingParams(
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        stop_token_ids=[151645, 151643],
        max_tokens=MAX_NEW_TOKENS,
        include_stop_str_in_output=False,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )

    return processor, llm, sampling_params


# =============================================================================
# 8. 单条推理
# =============================================================================

def infer_one(
    processor,
    llm,
    sampling_params,
    video_path: str,
    question: str,
    duration: float,
) -> Tuple[str, int]:
    formatted_query = PROMPT.format(query=question)

    patch_size = processor.image_processor.patch_size
    patch_area = (patch_size * 2) ** 2
    total_video_tokens = choose_total_video_tokens(duration)

    video_info = {
        "type": "video",
        "video": video_path,
        "total_pixels": total_video_tokens * patch_area,
        "min_pixels": 4 * patch_area,
        "max_frames": MAX_FRAMES,
        "fps": FPS,
    }

    messages = [
        {
            "role": "user",
            "content": [
                video_info,
                {
                    "type": "text",
                    "text": formatted_query,
                },
            ],
        }
    ]

    raw_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=patch_size,
        return_video_metadata=True,
        return_video_kwargs=True,
    )

    # qwen-vl-utils已经完成视频resize，禁止vLLM processor再次resize。
    video_kwargs = dict(video_kwargs or {})
    video_kwargs["do_resize"] = False

    if videos is None or len(videos) == 0:
        raise RuntimeError("No video tensor was produced")

    # 官方 demo 对帧索引执行同样的归零处理。
    video_data, video_metadata = videos[0]

    frame_indices = video_metadata.get("frames_indices", [])

    if frame_indices is not None:
        if torch.is_tensor(frame_indices):
            frame_indices = frame_indices.tolist()

        if len(frame_indices) > 0:
            start_idx = frame_indices[0]
            video_metadata["frames_indices"] = [
                idx - start_idx for idx in frame_indices
            ]

    videos = [(video_data, video_metadata)]

    prompt_data = {
        "prompt": raw_prompt,
        "multi_modal_data": {
            "video": videos,
        },
        "mm_processor_kwargs": video_kwargs,
    }

    outputs = llm.generate(
        prompts=[prompt_data],
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    if not outputs or not outputs[0].outputs:
        raise RuntimeError("vLLM returned no generation")

    return outputs[0].outputs[0].text.strip(), total_video_tokens


# =============================================================================
# 9. 断点续跑与提交文件
# =============================================================================

def load_existing_raw() -> Dict[str, dict]:
    by_id: Dict[str, dict] = {}

    if not RAW_JSONL.is_file():
        return by_id

    with open(RAW_JSONL, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[WARN] Ignore malformed raw line {line_no}",
                    file=sys.stderr,
                )
                continue

            if not isinstance(row, dict) or "id" not in row:
                continue

            by_id[str(row["id"])] = row

    return by_id


def build_submission(
    selected_items: List[dict],
    raw_by_id: Dict[str, dict],
    require_complete: bool,
) -> List[dict]:
    output: List[dict] = []
    missing: List[str] = []

    for item in selected_items:
        video_id = str(item["id"])
        result = raw_by_id.get(video_id)

        if result is None:
            missing.append(video_id)
            continue

        try:
            start = float(result["start"])
            end = float(result["end"])
        except (KeyError, TypeError, ValueError):
            missing.append(video_id)
            continue

        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or end <= start
        ):
            missing.append(video_id)
            continue

        output.append(
            {
                "id": video_id,
                "model_prediction": f"{start:.1f}-{end:.1f}",
            }
        )

    if require_complete and missing:
        preview = ", ".join(missing[:20])

        raise RuntimeError(
            f"{len(missing)} predictions are missing or invalid. "
            f"First IDs: {preview}"
        )

    return output


def validate_submission(
    selected_items: List[dict],
    output: List[dict],
) -> None:
    expected_ids = [
        str(item["id"])
        for item in selected_items
    ]
    actual_ids = [
        str(row.get("id"))
        for row in output
    ]

    if actual_ids != expected_ids:
        raise RuntimeError(
            "Submission IDs or order do not match the input"
        )

    if len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError(
            "Duplicate IDs found in submission"
        )

    pattern = re.compile(
        r"\d+(?:\.\d+)?-\d+(?:\.\d+)?$"
    )

    for line_no, row in enumerate(output, 1):
        if set(row.keys()) != {
            "id",
            "model_prediction",
        }:
            raise RuntimeError(
                f"Invalid fields at output line {line_no}"
            )

        if not pattern.fullmatch(
            str(row["model_prediction"])
        ):
            raise RuntimeError(
                f"Invalid prediction at line {line_no}: "
                f"{row['model_prediction']}"
            )


# =============================================================================
# 10. 主流程
# =============================================================================

def main() -> None:
    check_environment()

    EXPERIMENT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    items = load_jsonl(INPUT_JSONL)
    validate_input(items)

    selected_items = (
        items[:LIMIT]
        if LIMIT > 0
        else items
    )

    run_config = {
        "model_path": MODEL_PATH,
        "input_jsonl": INPUT_JSONL,
        "video_dir": VIDEO_DIR,
        "experiment_dir": EXPERIMENT_DIR,
        "limit": LIMIT,
        "selected_samples": len(selected_items),
        "gpu_id": GPU_ID,
        "fps": FPS,
        "adaptive_video_tokens": ADAPTIVE_VIDEO_TOKENS,
        "base_total_video_tokens": BASE_TOTAL_VIDEO_TOKENS,
        "max_total_video_tokens": MAX_TOTAL_VIDEO_TOKENS,
        "video_tokens_per_second": VIDEO_TOKENS_PER_SECOND,
        "video_token_round": VIDEO_TOKEN_ROUND,
        "max_frames": MAX_FRAMES,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "prompt": PROMPT,
    }
    atomic_write_json(RUN_CONFIG_JSON, run_config)

    raw_by_id = load_existing_raw()

    selected_ids = {
        str(item["id"])
        for item in selected_items
    }

    completed_before = sum(
        1
        for video_id in raw_by_id
        if video_id in selected_ids
    )

    print("=" * 80)
    print("[INFO] TaRO-8B + MARS2 VTG")
    print(f"[INFO] Samples: {len(selected_items)}")
    print(f"[INFO] Already completed: {completed_before}")
    print(f"[INFO] FPS: {FPS}")
    print(
        "[INFO] Adaptive video tokens: "
        f"{BASE_TOTAL_VIDEO_TOKENS}-{MAX_TOTAL_VIDEO_TOKENS}"
    )
    print(f"[INFO] Video tokens per second: {VIDEO_TOKENS_PER_SECOND}")
    print(f"[INFO] Max frames: {MAX_FRAMES}")
    print(f"[INFO] Experiment: {EXPERIMENT_PATH}")
    print(f"[INFO] Raw results: {RAW_JSONL}")
    print(f"[INFO] Final output: {OUT_JSONL}")
    print("=" * 80)

    remaining = [
        item
        for item in selected_items
        if str(item["id"]) not in raw_by_id
    ]

    if remaining:
        processor, llm, sampling_params = (
            load_model_and_processor()
        )
    else:
        processor = None
        llm = None
        sampling_params = None
        print(
            "[INFO] No remaining samples; "
            "rebuilding final out.jsonl."
        )

    newly_processed = 0

    progress = tqdm(
        remaining,
        total=len(remaining),
        desc="TaRO inference",
    )

    for item in progress:
        video_id = str(item["id"])
        question = str(item["question"]).strip()

        video_path = os.path.join(
            VIDEO_DIR,
            video_id + VIDEO_EXT,
        )

        output_text = ""
        used_video_tokens = choose_total_video_tokens(0.0)
        used_fallback = False
        error_type = None
        error_message = None

        try:
            if not os.path.isfile(video_path):
                raise FileNotFoundError(video_path)

            duration = ffprobe_duration(video_path)

            output_text, used_video_tokens = infer_one(
                processor=processor,
                llm=llm,
                sampling_params=sampling_params,
                video_path=video_path,
                question=question,
                duration=duration,
            )

            prediction = parse_answer(
                output_text,
                duration,
            )

            if prediction is None:
                raise RuntimeError(
                    "Could not parse a valid temporal window "
                    f"from model output: {output_text!r}"
                )

            start, end = prediction

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)

            try:
                duration = ffprobe_duration(video_path)
            except Exception:
                duration = 0.0

            used_video_tokens = choose_total_video_tokens(duration)

            if (
                USE_FULL_VIDEO_FALLBACK
                and duration > 0
            ):
                start = 0.0
                end = duration
                used_fallback = True
            else:
                append_jsonl(
                    ERROR_JSONL,
                    {
                        "id": video_id,
                        "question": question,
                        "video_path": video_path,
                        "output_text": output_text,
                        "error_type": error_type,
                        "error": error_message,
                        "traceback": traceback.format_exc(),
                        "used_fallback": False,
                    },
                )

                print(
                    f"\n[ERROR] id={video_id}: "
                    f"{error_type}: {error_message}",
                    file=sys.stderr,
                    flush=True,
                )

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                continue

            append_jsonl(
                ERROR_JSONL,
                {
                    "id": video_id,
                    "question": question,
                    "video_path": video_path,
                    "output_text": output_text,
                    "duration": duration,
                    "error_type": error_type,
                    "error": error_message,
                    "traceback": traceback.format_exc(),
                    "used_fallback": True,
                    "fallback_window": [
                        start,
                        end,
                    ],
                },
            )

            print(
                f"\n[WARN] id={video_id}: "
                f"{error_type}: {error_message}; "
                f"fallback={start:.1f}-{end:.1f}",
                file=sys.stderr,
                flush=True,
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        raw_row = {
            "id": video_id,
            "question": question,
            "video_path": video_path,
            "duration": round(float(duration), 3),
            "used_video_tokens": int(used_video_tokens),
            "output_text": output_text,
            "start": round(float(start), 6),
            "end": round(float(end), 6),
            "model_prediction": f"{start:.1f}-{end:.1f}",
            "used_fallback": used_fallback,
            "error_type": error_type,
            "error": error_message,
        }

        append_jsonl(RAW_JSONL, raw_row)
        raw_by_id[video_id] = raw_row
        newly_processed += 1

        fallback_count = sum(
            1
            for row in raw_by_id.values()
            if row.get("used_fallback")
        )

        progress.set_postfix(
            completed=len(raw_by_id),
            fallback=fallback_count,
        )

        if newly_processed % SAVE_OUT_EVERY == 0:
            partial_output = build_submission(
                selected_items=selected_items,
                raw_by_id=raw_by_id,
                require_complete=False,
            )
            atomic_write_jsonl(
                OUT_JSONL,
                partial_output,
            )

    final_output = build_submission(
        selected_items=selected_items,
        raw_by_id=raw_by_id,
        require_complete=True,
    )

    validate_submission(
        selected_items,
        final_output,
    )

    atomic_write_jsonl(
        OUT_JSONL,
        final_output,
    )

    fallback_count = sum(
        1
        for item in selected_items
        if raw_by_id[
            str(item["id"])
        ].get("used_fallback")
    )

    print("=" * 80)
    print("[INFO] Finished")
    print(f"[INFO] Final rows: {len(final_output)}")
    print(f"[INFO] Fallback rows: {fallback_count}")
    print(f"[INFO] Final output: {OUT_JSONL}")
    print("[INFO] Submission validation: PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
