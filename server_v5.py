# server.py
# MuseumAR Online Pipeline (SAM3 + Diffuser) — FIXED
# Key fixes:
# 1) DO NOT use -1 labels for diffuser (can silently kill supervision); keep labels in {0,1}
#    (optional: ignore background via 0->2 where 2 == ignore class when num_classes=2)
# 2) Force num_classes=2 for binary diffusion
# 3) Sanitize/normalize normals (NaN/Inf/zero -> re-estimate) to prevent G becoming NaN/zero
# 4) Pass frames as (frame, int_idx) not float

import os

# Keep your behavior: default GPU=1 if not set
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "1")

import base64
import io
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps

from diffuser.diffuser.diffuser import Diffuser
from diffuser.diffuser.frame import Frame

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

try:
    from sam3.sam3.visualization_utils import plot_results
except Exception:
    try:
        from sam3.visualization_utils import plot_results
    except Exception:
        plot_results = None

try:
    import open3d as o3d

    HAS_O3D = True
except Exception:
    HAS_O3D = False


# ============================================================
# Logging
# ============================================================
def _force_console_logger(name: str = "MuseumAR") -> logging.Logger:
    """
    Uvicorn/fastapi sometimes resets logging config.
    This ensures INFO logs always show on console.
    """
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)

    if not lg.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        h.setFormatter(fmt)
        lg.addHandler(h)

    lg.propagate = False
    return lg


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = _force_console_logger("MuseumAR")


# ============================================================
# Coordinate conventions
# ============================================================
# ARKit camera coords: x right, y up, forward is -Z
# OpenCV camera coords: x right, y down, forward is +Z
CV_FROM_ARKIT = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


# ============================================================
# Config
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]


def _pick_first_existing(paths):
    for p in paths:
        if p and Path(p).is_file():
            return Path(p)
    return Path(paths[0]) if paths else None


@dataclass(frozen=True)
class Cfg:
    # SAM3
    sam3_checkpoint: Path = Path(os.getenv("SAM3_CHECKPOINT", str(REPO_ROOT / "sam3" / "sam3.pt")))
    sam3_device: str = os.getenv("SAM3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    sam3_prompt_default: str = os.getenv("SAM3_PROMPT", "computer")
    sam3_confidence: float = float(os.getenv("SAM3_CONFIDENCE", "0.5"))
    sam3_bpe_path: Path = _pick_first_existing(
        [
            os.getenv("SAM3_BPE_PATH", "").strip(),
            str(REPO_ROOT / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
            str(REPO_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
        ]
    )

    # Diffuser
    diffuser_max_iters: int = int(os.getenv("DIFFUSER_MAX_ITERS", "200"))
    diffuser_num_pt_neighbors: int = int(os.getenv("DIFFUSER_NUM_PT_NEIGHBORS", "24"))
    diffuser_distance_mu: float = float(os.getenv("DIFFUSER_DISTANCE_MU", "0.05"))
    diffuser_normals_mu: float = float(os.getenv("DIFFUSER_NORMALS_MU", "0.175"))
    diffuser_px_to_pt_weight: float = float(os.getenv("DIFFUSER_PX_TO_PT_WEIGHT", "1e-4"))  # revert safer default
    diffuser_num_classes_fixed: int = int(os.getenv("DIFFUSER_NUM_CLASSES", "2"))  # ✅ force binary

    # ✅ optional: ignore background for diffusion supervision
    # if True: background pixels -> 2 (ignore) when num_classes=2 (diffuser will internally use 3 columns)
    diffuser_ignore_background: bool = os.getenv("DIFFUSER_IGNORE_BACKGROUND", "0").lower() in ("1", "true", "yes")

    # Sanity/debug
    depth_consist_eps_m: float = float(os.getenv("DEPTH_CONSIST_EPS_M", "0.05"))
    label_to_depth_mode: str = os.getenv("LABEL_TO_DEPTH_MODE", "warp").lower().strip()
    rgb_exif_transpose: bool = os.getenv("RGB_EXIF_TRANSPOSE", "0").lower() in ("1", "true", "yes")
    strict_depth_gate: bool = os.getenv("STRICT_DEPTH_GATE", "0").lower() in ("1", "true", "yes")
    debug_save_dir: str = os.getenv("DEBUG_SAVE_DIR", "./debug_dir").strip()

    # Diffuser frame grid
    # - "depth": labels/depth on depth resolution; intrinsics use K_depth (recommended)
    # - "rgb":   labels on RGB; depth warped to RGB; intrinsics use K_rgb
    diffuser_frame_grid: str = os.getenv("DIFFUSER_FRAME_GRID", "depth").lower().strip()

    # Sparse depth (only keep depth inside mask)
    diffuser_sparse_depth: bool = os.getenv("DIFFUSER_SPARSE_DEPTH", "0").lower() in ("1", "true", "yes")

    # Normals sanitize
    normals_bad_ratio_reestimate: float = float(os.getenv("NORMALS_BAD_RATIO_REESTIMATE", "0.01"))
    normals_est_radius: float = float(os.getenv("NORMALS_EST_RADIUS", "0.05"))
    normals_est_max_nn: int = int(os.getenv("NORMALS_EST_MAX_NN", "30"))

    # VLM (optional)
    vlm_api_url: str = os.getenv("VLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    vlm_api_key: str = os.getenv("VLM_API_KEY", "")
    vlm_model: str = os.getenv("VLM_MODEL", "qwen3-vl-plus-2025-12-19")
    vlm_system_prompt: str = os.getenv(
        "VLM_SYSTEM_PROMPT",
        "请用 JSON 描述图像中最靠近摄像头的的电脑（name/category/material/tags/confidence/one_sentence）。",
    )
    vlm_user_prompt_default: str = os.getenv(
        "VLM_USER_PROMPT",
        "请用 JSON 描述图像中的电脑（name/category/material/tags/confidence/one_sentence）。",
    )

# VLM_SYSTEM_PROMPT = os.getenv(
#     "VLM_SYSTEM_PROMPT",
#     """
# 你是“展品语义信息生成器”。你将看到一张博物馆/展厅场景中的单张 RGB 图片（可能包含展品、展柜、说明牌、背景游客、反光等）。你的任务是输出结构化语义信息，用于后续系统生成 AR 标签内容与多视图介绍。

# 重要边界（必须遵守）：
# 1) 你只负责语义与文案：名称/类别/材质/特征/简短介绍/可核验信息/不确定性。
# 2) 只输出一个严格 JSON 对象，不得输出 Markdown、解释、前后缀文本。
# 3) 不要捏造事实：年代、作者、出处、用途、具体名称等若无法从图像可靠判断，必须写“未知/疑似”，并在 uncertainty_notes 说明原因与需要核验的点。
# 4) 优先识别画面中最主要展品（main_object）。如有多件重要展品，提供 top-3 alternatives。
# 5) 输出文本要适合 AR：短、清晰、可扫读。避免长段落；使用要点列表。
# 6) confidence 取值 [0,1]；推测性内容必须降低 confidence，并写清推测依据。
# 7) 你无法访问互联网，禁止引用外部资料或编造来源。

# 输出 JSON 顶层字段（必须全部包含）：
# - schema_version: "museumar.semantic.v1"
# - language: "zh"
# - main_object: {
#     name_zh, name_en,
#     category,                 // 如：陶瓷/青铜/雕塑/书画/器具/建筑构件/化石等
#     material,                 // 可为数组或字符串，如无法确定写"未知"
#     visual_attributes: {color, shape, patterns, notable_parts},
#     scene_context: {in_case: bool, has_plaque: bool, reflections_or_glass: bool, occlusion: bool},
#     short_description: 字符串（<= 40字，尽量客观）
#     confidence: [0,1]
#   }
# - plaque_text: {
#     detected: bool,
#     extracted: [ {key: "name|era|material|origin|collector|museum|id|other", value: 字符串, confidence:[0,1]} ],
#     raw_snippet: 字符串,     // 能读多少写多少；读不清写空串
#     notes: 字符串
#   }
# - semantic_summary: {
#     one_sentence: 字符串（<= 30字）,
#     bullets: [字符串]（3-5条，每条<= 18字，事实性优先）,
#     tags: [字符串]（6-15个）
#   }
# - views: [
#     {
#       view_id: "overview"|"material"|"significance"|"history"|"craft"|"trivia",
#       title: 短标题（<=10字）,
#       short: 一句话（<=30字）,
#       details: [字符串]（2-5条，每条<=40字，尽量可核验；不确定要标“疑似”）,
#       confidence: [0,1]
#     }
#   ]
# - alternatives: [ {name_zh, category, reason, confidence} ]
# - uncertainty_notes: [字符串]（列出不确定点与原因）
# - followup_questions: [字符串]（3-8条，面向“如何通过铭牌/馆方信息核验”）
# - safety: {contains_people: bool, contains_sensitive_text: bool, notes: 字符串}

# 关于铭牌/说明牌：
# - 若画面中有铭牌且能读到部分信息，优先写入 plaque_text.extracted，并在 summary 与 views 中引用“牌面可见信息”。
# - 看不清不要瞎猜，raw_snippet 留空或写可见片段。

# views 最少提供 3 个：overview、material、significance。history/craft/trivia 只有在你有明显视觉依据时才补充，否则不要编。
# """.strip(),
# )

# VLM_USER_PROMPT_DEFAULT = os.getenv(
#     "VLM_USER_PROMPT",
#     """
# 这是展品的关键帧 RGB 图像。请为“后续贴标系统”生成语义信息：我们会把你的 JSON 用作标签文案与多视图卡片内容，并与分割/点云对齐到 3D 场景中。

# 请严格按系统要求，仅输出一个 JSON 对象。

# 生成重点：
# 1) 聚焦图中最主要展品，忽略背景游客和反光噪声。
# 2) 若能看到铭牌/说明牌，请尽量提取可读文字到 plaque_text；看不清则明确说明不清晰。
# 3) 文案要短：one_sentence 与 bullets 用于 AR 快速扫读；views 用于展品多视图介绍（至少 overview/material/significance）。
# 4) 对年代、作者、出处、用途、具体名称等不确定信息，宁可写“未知/疑似”，并在 uncertainty_notes + followup_questions 提出核验问题。

# 请严格按系统要求，仅输出一个 JSON 对象。
# """.strip(),
# )



CFG = Cfg()
app = FastAPI(title="MuseumAR Online Pipeline (SAM3 + Diffuser) — FIXED")


# ============================================================
# IO helpers
# ============================================================
def _read_all(f: UploadFile) -> bytes:
    try:
        return f.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}") from exc


def _decode_json_upload(f: UploadFile) -> Dict:
    raw = _read_all(f)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty meta json")
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid meta json: {exc}") from exc


def _decode_points(f: UploadFile) -> np.ndarray:
    data = _read_all(f)
    if len(data) % 12 != 0:
        raise HTTPException(status_code=400, detail=f"points size not multiple of 12: {len(data)}")
    return np.frombuffer(data, dtype=np.float32).reshape(-1, 3)


def _decode_normals(f: Optional[UploadFile], expected_rows: int) -> Optional[np.ndarray]:
    if f is None:
        return None
    data = _read_all(f)
    if len(data) % 12 != 0:
        raise HTTPException(status_code=400, detail=f"normals size not multiple of 12: {len(data)}")
    normals = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
    if normals.shape[0] != expected_rows:
        raise HTTPException(status_code=400, detail=f"normals count {normals.shape[0]} != points {expected_rows}")
    return normals


def _decode_depth(f: UploadFile, meta: Dict) -> np.ndarray:
    data = _read_all(f)
    try:
        h = int(meta["depth_resolution"]["h"])
        w = int(meta["depth_resolution"]["w"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail="meta.depth_resolution.{h,w} required") from exc

    expected = h * w * 4
    if len(data) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"depth size mismatch: got {len(data)}, expected {expected} (h={h}, w={w})",
        )

    depth = np.frombuffer(data, dtype=np.float32).reshape(h, w)

    # unit normalize (mm->m) with heuristics
    unit = str(meta.get("unit", "")).lower()
    if unit and "mill" in unit:
        depth = depth / 1000.0
    else:
        finite = depth[np.isfinite(depth)]
        if finite.size > 0 and np.nanmax(finite) > 50.0:
            depth = depth / 1000.0

    return depth.astype(np.float32)


def _decode_rgb(f: UploadFile) -> Tuple[bytes, Image.Image]:
    data = _read_all(f)
    try:
        img0 = Image.open(io.BytesIO(data))
        if CFG.rgb_exif_transpose:
            img0 = ImageOps.exif_transpose(img0)
        img = img0.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to parse rgb: {exc}") from exc
    return data, img


# ============================================================
# Debug save
# ============================================================
def _maybe_debug_dir() -> Optional[Path]:
    if not CFG.debug_save_dir:
        return None
    d = Path(CFG.debug_save_dir).resolve() / f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_write_bytes(path: Path, data: bytes) -> None:
    try:
        path.write_bytes(data)
    except Exception:
        pass


def _safe_write_text(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass


def _safe_save_npy(path: Path, arr: np.ndarray) -> None:
    try:
        np.save(path, arr)
    except Exception:
        pass


# ============================================================
# Intrinsics / Pose parsing
# ============================================================
def _as_mat33(x: Any) -> np.ndarray:
    a = np.array(x, dtype=np.float32)
    if a.size == 9:
        return a.reshape(3, 3)
    if a.shape == (3, 3):
        return a.astype(np.float32)
    raise ValueError(f"Cannot parse 3x3 from shape={a.shape} size={a.size}")


def _scale_K(K: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    K2 = K.copy()
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def _read_pose_any(meta: Dict) -> Any:
    if "T_C_W" in meta:
        return meta["T_C_W"]
    if "T_W_C" in meta:
        return meta["T_W_C"]
    if "camera_transform" in meta:
        return meta["camera_transform"]
    raise KeyError("Missing pose (T_C_W/T_W_C/camera_transform)")


def _get_T_C_W_cv(meta: Dict) -> np.ndarray:
    pose = _read_pose_any(meta)
    A = np.array(pose, dtype=np.float32)

    convention = str(meta.get("pose_convention", "T_W_C")).upper()

    if A.ndim == 1 and A.size == 16:
        # ARKit simd_float4x4 flatten is column-major
        T = A.reshape(4, 4, order="F")
    elif A.shape == (4, 4):
        # ✅ restore old behavior: default transpose
        # You can override by setting FORCE_POSE_TRANSPOSE=0 and FORCE_POSE_NO_TRANSPOSE=1 if needed.
        if os.getenv("FORCE_POSE_NO_TRANSPOSE", "0").lower() in ("1", "true", "yes"):
            T = A
        else:
            T = A.T
    else:
        raise ValueError(f"Unsupported pose shape: {A.shape}")

    if convention == "T_W_C":
        T_C_W_arkit = np.linalg.inv(T)
    elif convention == "T_C_W":
        T_C_W_arkit = T
    else:
        raise ValueError(f"Unknown pose_convention: {convention} (use T_W_C or T_C_W)")

    pose_already_cv = bool(meta.get("pose_is_cv") or os.getenv("POSE_IS_CV", "").lower() in ("1", "true", "yes"))
    T_C_W_cv = T_C_W_arkit if pose_already_cv else (CV_FROM_ARKIT @ T_C_W_arkit)
    return T_C_W_cv.astype(np.float32)


def _read_K_rgb_scaled(meta: Dict, rgb_hw_actual: Tuple[int, int]) -> np.ndarray:
    if "K_rgb" in meta:
        K = _as_mat33(meta["K_rgb"])
    elif "intrinsics" in meta:
        K = _as_mat33(meta["intrinsics"])
    elif "K" in meta:
        K = _as_mat33(meta["K"])
    else:
        raise KeyError("Missing intrinsics (K_rgb/intrinsics/K)")

    if "rgb_resolution" in meta:
        ref_h = int(meta["rgb_resolution"]["h"])
        ref_w = int(meta["rgb_resolution"]["w"])
        if (ref_h, ref_w) != rgb_hw_actual:
            K = _scale_K(K, (ref_h, ref_w), rgb_hw_actual)
    return K.astype(np.float32)


def _read_K_depth_scaled(
    meta: Dict,
    depth_hw_actual: Tuple[int, int],
    K_rgb_scaled: np.ndarray,
    rgb_hw_actual: Tuple[int, int],
) -> np.ndarray:
    if "K_depth" in meta:
        Kd = _as_mat33(meta["K_depth"])
        if "K_depth_resolution" in meta:
            ref_h = int(meta["K_depth_resolution"]["h"])
            ref_w = int(meta["K_depth_resolution"]["w"])
            if (ref_h, ref_w) != depth_hw_actual:
                Kd = _scale_K(Kd, (ref_h, ref_w), depth_hw_actual)
        return Kd.astype(np.float32)

    if "depth_intrinsics" in meta:
        Kd = _as_mat33(meta["depth_intrinsics"])
        if "depth_intrinsics_resolution" in meta:
            ref_h = int(meta["depth_intrinsics_resolution"]["h"])
            ref_w = int(meta["depth_intrinsics_resolution"]["w"])
            if (ref_h, ref_w) != depth_hw_actual:
                Kd = _scale_K(Kd, (ref_h, ref_w), depth_hw_actual)
        return Kd.astype(np.float32)

    return _scale_K(K_rgb_scaled, rgb_hw_actual, depth_hw_actual).astype(np.float32)


def _print_K_sanity(tag: str, K: np.ndarray, hw: Tuple[int, int]) -> None:
    H, W = int(hw[0]), int(hw[1])
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    logger.info("[%s] fx=%.2f fy=%.2f cx=%.2f cy=%.2f (H,W)=(%d,%d)", tag, fx, fy, cx, cy, H, W)


# ============================================================
# Models (cached)
# ============================================================
@lru_cache()
def _get_sam3_processor() -> Sam3Processor:
    if not CFG.sam3_checkpoint.is_file():
        raise RuntimeError(f"SAM3 checkpoint not found at {CFG.sam3_checkpoint}")
    if CFG.sam3_bpe_path is None or not Path(CFG.sam3_bpe_path).is_file():
        raise RuntimeError(f"SAM3 BPE not found at {CFG.sam3_bpe_path}")

    model = build_sam3_image_model(
        checkpoint_path=str(CFG.sam3_checkpoint),
        device=CFG.sam3_device,
        eval_mode=True,
        enable_inst_interactivity=False,
        load_from_HF=False,
        bpe_path=str(CFG.sam3_bpe_path),
    )
    return Sam3Processor(model, device=CFG.sam3_device, confidence_threshold=CFG.sam3_confidence)


@lru_cache()
def _get_diffuser() -> Diffuser:
    return Diffuser(
        num_pt_neighbors=CFG.diffuser_num_pt_neighbors,
        distance_mu=CFG.diffuser_distance_mu,
        normals_mu=CFG.diffuser_normals_mu,
        px_to_pt_weight=CFG.diffuser_px_to_pt_weight,
    )


# ============================================================
# SAM3 -> label_map
# ============================================================
def _label_map_from_best_mask(masks: torch.Tensor, scores: torch.Tensor, fallback_hw: Tuple[int, int]) -> np.ndarray:
    H, W = fallback_hw
    label_map = np.zeros((H, W), dtype=np.uint16)
    if masks is None or scores is None or scores.numel() == 0:
        return label_map

    best = int(scores.argmax().item())
    m = masks[best]
    if m.ndim == 3 and m.shape[0] == 1:
        m = m.squeeze(0)
    if m.ndim != 2:
        return label_map

    mask01 = m.detach().to(torch.uint8).cpu().numpy().astype(bool)
    label_map[mask01] = 1
    return label_map


def _label_map_png_bytes(label_map: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(label_map.astype(np.uint16)).save(buf, format="PNG")
    return buf.getvalue()


def _maybe_save_sam3_vis(rgb_img: Image.Image, state: Any, output: Dict[str, Any], dbg_dir: Optional[Path], prompt: str):
    if dbg_dir is None or plot_results is None:
        return
    out_dir = dbg_dir / "sam3_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        sig = inspect.signature(plot_results)
        params = sig.parameters
        kwargs: Dict[str, Any] = {}

        for k in ("save_dir", "out_dir", "output_dir", "save_path", "path"):
            if k in params:
                kwargs[k] = str(out_dir)
                break
        for k in ("output", "result", "pred", "outputs"):
            if k in params:
                kwargs[k] = output
                break
        for k in ("prompt", "text", "name", "prefix", "tag"):
            if k in params:
                kwargs[k] = str(prompt)
                break

        try:
            plot_results(rgb_img, state, **kwargs)
            return
        except TypeError:
            pass

        cwd = os.getcwd()
        try:
            os.chdir(str(out_dir))
            plot_results(rgb_img, state)
        finally:
            os.chdir(cwd)
    except Exception as exc:
        _safe_write_text(out_dir / "plot_results_error.txt", f"{type(exc).__name__}: {exc}")


def _run_sam3(image: Image.Image, prompt: str, dbg_dir: Optional[Path] = None) -> Dict:
    processor = _get_sam3_processor()
    with torch.inference_mode():
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=prompt)

    _maybe_save_sam3_vis(image, state, output, dbg_dir=dbg_dir, prompt=prompt)

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    H, W = image.height, image.width

    if masks is None or scores is None or scores.numel() == 0:
        label_map = np.zeros((H, W), dtype=np.uint16)
        return {
            "label_map": label_map,
            "mask_count": 0,
            "boxes": [],
            "scores": [],
            "best_idx": -1,
            "best_score": 0.0,
            "label_png_base64": base64.b64encode(_label_map_png_bytes(label_map)).decode("ascii"),
        }

    best_idx = int(scores.argmax().item())
    label_map = _label_map_from_best_mask(masks, scores, (H, W))
    boxes_list = boxes.detach().cpu().tolist() if boxes is not None else []
    scores_list = scores.detach().cpu().tolist()

    return {
        "label_map": label_map,
        "mask_count": int(scores.numel()),
        "boxes": boxes_list,
        "scores": scores_list,
        "best_idx": best_idx,
        "best_score": float(scores_list[best_idx]),
        "label_png_base64": base64.b64encode(_label_map_png_bytes(label_map)).decode("ascii"),
    }


# ============================================================
# Warps: RGB label -> Depth label (intrinsics-aware)
# ============================================================
def _warp_label_rgb_to_depth(
    label_rgb: np.ndarray,
    K_rgb: np.ndarray,
    rgb_hw: Tuple[int, int],
    K_depth: np.ndarray,
    depth_hw: Tuple[int, int],
) -> np.ndarray:
    dh, dw = int(depth_hw[0]), int(depth_hw[1])
    H, W = int(rgb_hw[0]), int(rgb_hw[1])

    uu, vv = np.meshgrid(np.arange(dw, dtype=np.float32), np.arange(dh, dtype=np.float32))

    fx_d, fy_d, cx_d, cy_d = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
    fx_r, fy_r, cx_r, cy_r = K_rgb[0, 0], K_rgb[1, 1], K_rgb[0, 2], K_rgb[1, 2]

    x_n = (uu - cx_d) / (fx_d + 1e-9)
    y_n = (vv - cy_d) / (fy_d + 1e-9)

    u_r = fx_r * x_n + cx_r
    v_r = fy_r * y_n + cy_r

    iu = np.floor(u_r + 0.5).astype(np.int32)
    iv = np.floor(v_r + 0.5).astype(np.int32)

    out = np.zeros((dh, dw), dtype=np.uint16)
    inside = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    out[inside] = label_rgb[iv[inside], iu[inside]].astype(np.uint16)
    return out


def _warp_depth_to_rgb(
    depth_map: np.ndarray,
    K_depth: np.ndarray,
    depth_hw: Tuple[int, int],
    K_rgb: np.ndarray,
    rgb_hw: Tuple[int, int],
) -> np.ndarray:
    dh, dw = int(depth_hw[0]), int(depth_hw[1])
    H, W = int(rgb_hw[0]), int(rgb_hw[1])

    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    fx_r, fy_r, cx_r, cy_r = K_rgb[0, 0], K_rgb[1, 1], K_rgb[0, 2], K_rgb[1, 2]
    fx_d, fy_d, cx_d, cy_d = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]

    x_n = (uu - cx_r) / (fx_r + 1e-9)
    y_n = (vv - cy_r) / (fy_r + 1e-9)

    u_d = fx_d * x_n + cx_d
    v_d = fy_d * y_n + cy_d

    iu = np.floor(u_d + 0.5).astype(np.int32)
    iv = np.floor(v_d + 0.5).astype(np.int32)

    out = np.full((H, W), np.nan, dtype=np.float32)
    inside = (iu >= 0) & (iu < dw) & (iv >= 0) & (iv < dh)
    out[inside] = depth_map[iv[inside], iu[inside]].astype(np.float32)
    return out


# ============================================================
# Sanity / Debug
# ============================================================
def _encode_npy_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as buf:
        np.save(buf, arr)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _proj_sanity_rgb(points_w: np.ndarray, T_C_W: np.ndarray, K_rgb: np.ndarray, label_map_rgb: np.ndarray) -> Dict:
    H, W = label_map_rgb.shape[:2]
    N = points_w.shape[0]
    pw = np.concatenate([points_w.astype(np.float32), np.ones((N, 1), dtype=np.float32)], axis=1)
    pc = (T_C_W @ pw.T).T[:, :3]

    z = pc[:, 2]
    valid = z > 1e-4
    if valid.sum() == 0:
        return {"label_hw": [H, W], "zpos": 0, "inimg": 0, "inmask": 0}

    pc = pc[valid]
    u = K_rgb[0, 0] * (pc[:, 0] / pc[:, 2]) + K_rgb[0, 2]
    v = K_rgb[1, 1] * (pc[:, 1] / pc[:, 2]) + K_rgb[1, 2]

    iu = np.floor(u + 0.5).astype(np.int32)
    iv = np.floor(v + 0.5).astype(np.int32)
    inimg = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)

    inmask = 0
    if inimg.sum() > 0:
        inmask = int((label_map_rgb[iv[inimg], iu[inimg]] == 1).sum())

    return {"label_hw": [H, W], "zpos": int(valid.sum()), "inimg": int(inimg.sum()), "inmask": inmask}


def _proj_sanity_depth_consistency(
    points_w: np.ndarray,
    T_C_W: np.ndarray,
    K_depth: np.ndarray,
    depth_map: np.ndarray,
    label_map_depth: np.ndarray,
) -> Dict:
    dh, dw = depth_map.shape
    N = points_w.shape[0]
    pw = np.concatenate([points_w.astype(np.float32), np.ones((N, 1), dtype=np.float32)], axis=1)
    pc = (T_C_W @ pw.T).T[:, :3]

    z = pc[:, 2]
    valid = z > 1e-4
    if valid.sum() == 0:
        return {
            "depth_hw": [dh, dw],
            "zpos": 0,
            "in_depth_img": 0,
            "depth_finite": 0,
            "depth_ok": 0,
            "inmask_depth_ok": 0,
            "mask_area_depth": int((label_map_depth == 1).sum()),
            "depth_consist_eps_m": float(CFG.depth_consist_eps_m),
        }

    pc = pc[valid]
    z = pc[:, 2]

    u = K_depth[0, 0] * (pc[:, 0] / z) + K_depth[0, 2]
    v = K_depth[1, 1] * (pc[:, 1] / z) + K_depth[1, 2]

    iu = np.floor(u + 0.5).astype(np.int32)
    iv = np.floor(v + 0.5).astype(np.int32)

    inimg = (iu >= 0) & (iu < dw) & (iv >= 0) & (iv < dh)
    if inimg.sum() == 0:
        return {
            "depth_hw": [dh, dw],
            "zpos": int(valid.sum()),
            "in_depth_img": 0,
            "depth_finite": 0,
            "depth_ok": 0,
            "inmask_depth_ok": 0,
            "mask_area_depth": int((label_map_depth == 1).sum()),
            "depth_consist_eps_m": float(CFG.depth_consist_eps_m),
        }

    iu2 = iu[inimg]
    iv2 = iv[inimg]
    z2 = z[inimg]

    d = depth_map[iv2, iu2]
    finite = np.isfinite(d) & (d > 1e-6)
    if finite.sum() == 0:
        return {
            "depth_hw": [dh, dw],
            "zpos": int(valid.sum()),
            "in_depth_img": int(inimg.sum()),
            "depth_finite": 0,
            "depth_ok": 0,
            "inmask_depth_ok": 0,
            "mask_area_depth": int((label_map_depth == 1).sum()),
            "depth_consist_eps_m": float(CFG.depth_consist_eps_m),
        }

    iu3 = iu2[finite]
    iv3 = iv2[finite]
    z3 = z2[finite]
    d3 = d[finite]

    err = np.abs(d3 - z3)
    ok = err <= CFG.depth_consist_eps_m
    depth_ok = int(ok.sum())

    inmask_ok = 0
    if depth_ok > 0:
        iu4 = iu3[ok]
        iv4 = iv3[ok]
        inmask_ok = int((label_map_depth[iv4, iu4] == 1).sum())

    return {
        "depth_hw": [dh, dw],
        "zpos": int(valid.sum()),
        "in_depth_img": int(inimg.sum()),
        "depth_finite": int(finite.sum()),
        "depth_ok": depth_ok,
        "inmask_depth_ok": inmask_ok,
        "mask_area_depth": int((label_map_depth == 1).sum()),
        "depth_consist_eps_m": float(CFG.depth_consist_eps_m),
        "depth_err_median": float(np.median(err)) if err.size else None,
        "depth_err_p90": float(np.quantile(err, 0.9)) if err.size else None,
        "depth_err_max": float(np.max(err)) if err.size else None,
    }


# ============================================================
# Diffuser helpers (FIXED)
# ============================================================
def _estimate_normals_open3d(points: np.ndarray) -> np.ndarray:
    if not HAS_O3D:
        raise HTTPException(status_code=400, detail="normals missing and open3d unavailable to estimate them")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=CFG.normals_est_radius, max_nn=CFG.normals_est_max_nn)
    )
    pcd.normalize_normals()
    return np.asarray(pcd.normals, dtype=np.float32)


def _sanitize_normals(points: np.ndarray, normals: Optional[np.ndarray]) -> np.ndarray:
    if normals is None:
        logger.info("[NORMALS_SANITY] normals missing -> estimate via open3d")
        return _estimate_normals_open3d(points)

    n = normals.astype(np.float32, copy=False)
    bad = ~np.isfinite(n).all(axis=1)
    l2 = np.linalg.norm(n, axis=1)
    bad |= (l2 < 1e-6)

    bad_ratio = float(bad.mean())
    logger.info("[NORMALS_SANITY] bad_ratio=%.6f", bad_ratio)

    if bad_ratio > CFG.normals_bad_ratio_reestimate:
        logger.info("[NORMALS_SANITY] too many bad normals -> re-estimate via open3d")
        return _estimate_normals_open3d(points)

    # fix small portion + normalize
    n2 = n.copy()
    n2[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    l2 = np.linalg.norm(n2, axis=1, keepdims=True)
    n2 = n2 / np.clip(l2, 1e-6, None)
    return n2.astype(np.float32)


def _make_frame(
    labels_2d: np.ndarray,
    depth_2d: np.ndarray,
    K_labels: np.ndarray,
    K_depth: np.ndarray,
    T_C_W: np.ndarray,
) -> Frame:
    if labels_2d.dtype != np.int32:
        labels_2d = labels_2d.astype(np.int32)

    return Frame(
        labels=labels_2d,
        depth=depth_2d.astype(np.float32),
        intrinsics_rgb=K_labels.astype(np.float32),
        intrinsics_depth=K_depth.astype(np.float32),
        extrinsics=T_C_W.astype(np.float32),
    )


def _prepare_diffuser_inputs(
    label_map_rgb: np.ndarray,
    label_map_depth: np.ndarray,
    depth_map: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    rgb_hw: Tuple[int, int],
    depth_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    grid = CFG.diffuser_frame_grid

    if grid == "rgb":
        depth_rgb = _warp_depth_to_rgb(depth_map, K_depth, depth_hw, K_rgb, rgb_hw)
        labels = label_map_rgb
        depth2d = depth_rgb
        K_labels = K_rgb
        K_depth2d = K_rgb

        if CFG.diffuser_sparse_depth:
            depth2d = depth2d.copy()
            depth2d[labels != 1] = np.nan

        return labels, depth2d, K_labels, K_depth2d, "rgb"

    # default: depth grid
    labels = label_map_depth
    depth2d = depth_map
    K_labels = K_depth
    K_depth2d = K_depth

    if CFG.diffuser_sparse_depth:
        depth2d = depth2d.copy()
        depth2d[labels != 1] = np.nan

    return labels, depth2d, K_labels, K_depth2d, "depth"


def _run_diffusion(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    labels_2d: np.ndarray,
    depth_2d: np.ndarray,
    K_labels: np.ndarray,
    K_depth: np.ndarray,
    T_C_W: np.ndarray,
) -> np.ndarray:
    normals2 = _sanitize_normals(points, normals)

    diffuser = _get_diffuser()
    frame = _make_frame(labels_2d, depth_2d, K_labels, K_depth, T_C_W)

    # ✅ force binary classes
    num_classes = int(CFG.diffuser_num_classes_fixed)

    # IMPORTANT: frames must be iterable yielding (frame, idx:int)
    labels, _ = diffuser.run(
        points,
        normals2,
        [(frame, 0)],
        num_classes=num_classes,
        max_iters=CFG.diffuser_max_iters,
    )
    return labels.astype(np.uint32)


# ============================================================
# VLM (optional)
# ============================================================
def _call_vlm(rgb_img: Image.Image, user_prompt: str, extra: Optional[Dict] = None) -> Dict:
    if not (CFG.vlm_api_url and CFG.vlm_api_key and CFG.vlm_model):
        return {"skipped": "VLM not configured (need VLM_API_URL/VLM_API_KEY/VLM_MODEL)"}
    extra = extra or {}

    buf = io.BytesIO()
    rgb_img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    payload = {
        "model": CFG.vlm_model,
        "temperature": float(extra.get("vlm_temperature", 0.2)),
        "stream": False,
        "messages": [
            {"role": "system", "content": CFG.vlm_system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
    }

    enable_thinking = bool(extra.get("vlm_enable_thinking", False))
    thinking_budget = extra.get("vlm_thinking_budget")
    if enable_thinking or thinking_budget is not None:
        payload["extra_body"] = {"enable_thinking": enable_thinking}
        if thinking_budget is not None:
            try:
                payload["extra_body"]["thinking_budget"] = int(thinking_budget)
            except Exception:
                pass

    headers = {"Authorization": f"Bearer {CFG.vlm_api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(CFG.vlm_api_url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": f"VLM request failed: {exc}"}


# ============================================================
# API
# ============================================================
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "sam3_checkpoint": str(CFG.sam3_checkpoint),
        "sam3_bpe_path": str(CFG.sam3_bpe_path),
        "sam3_device": CFG.sam3_device,
        "sam3_prompt_default": CFG.sam3_prompt_default,
        "has_open3d": HAS_O3D,
        "vlm_ready": bool(CFG.vlm_api_url and CFG.vlm_api_key and CFG.vlm_model),
        "debug_save_dir": CFG.debug_save_dir or None,
        "rgb_exif_transpose": CFG.rgb_exif_transpose,
        "strict_depth_gate": CFG.strict_depth_gate,
        "label_to_depth_mode": CFG.label_to_depth_mode,
        "depth_consist_eps_m": CFG.depth_consist_eps_m,
        "diffuser_frame_grid": CFG.diffuser_frame_grid,
        "diffuser_sparse_depth": CFG.diffuser_sparse_depth,
        "diffuser_ignore_background": CFG.diffuser_ignore_background,
        "diffuser_num_classes_fixed": CFG.diffuser_num_classes_fixed,
        "diffuser_px_to_pt_weight": CFG.diffuser_px_to_pt_weight,
    }


@app.post("/snapshot")
async def snapshot(
    meta: UploadFile = File(...),
    points: UploadFile = File(...),
    normals: Optional[UploadFile] = File(None),
    depth: UploadFile = File(...),
    rgb: UploadFile = File(...),
):
    dbg = _maybe_debug_dir()

    # ---- decode inputs ----
    meta_obj = _decode_json_upload(meta)
    pts = _decode_points(points)
    nrm = _decode_normals(normals, pts.shape[0])
    depth_map = _decode_depth(depth, meta_obj)
    rgb_bytes_raw, rgb_img = _decode_rgb(rgb)

    H_rgb, W_rgb = int(rgb_img.height), int(rgb_img.width)
    dh, dw = int(depth_map.shape[0]), int(depth_map.shape[1])
    rgb_hw = (H_rgb, W_rgb)
    depth_hw = (dh, dw)

    # ---- EARLY debug save ----
    if dbg is not None:
        _safe_write_text(dbg / "meta.json", json.dumps(meta_obj, ensure_ascii=False, indent=2))
        _safe_save_npy(dbg / "points.npy", pts)
        _safe_save_npy(dbg / "depth.npy", depth_map)
        _safe_write_bytes(dbg / "rgb_upload.bin", rgb_bytes_raw)
        try:
            rgb_img.save(dbg / "rgb.png")
        except Exception:
            pass
        if nrm is not None:
            _safe_save_npy(dbg / "normals.npy", nrm)

    # ---- pose & intrinsics (scaled) ----
    T_C_W = _get_T_C_W_cv(meta_obj)
    K_rgb = _read_K_rgb_scaled(meta_obj, rgb_hw_actual=rgb_hw)
    K_depth = _read_K_depth_scaled(meta_obj, depth_hw_actual=depth_hw, K_rgb_scaled=K_rgb, rgb_hw_actual=rgb_hw)

    _print_K_sanity("K_RGB_EFFECTIVE", K_rgb, rgb_hw)
    _print_K_sanity("K_DEPTH_EFFECTIVE", K_depth, depth_hw)

    finite = depth_map[np.isfinite(depth_map)]
    if finite.size > 0:
        logger.info("[DEPTH_RANGE] %.6f %.6f %s", float(np.min(finite)), float(np.max(finite)), str(depth_map.dtype))

    # ---- run SAM3 ----
    sam3_prompt = meta_obj.get("sam3_prompt", CFG.sam3_prompt_default)
    sam3_out = _run_sam3(rgb_img, sam3_prompt, dbg_dir=dbg)
    label_map_rgb = sam3_out["label_map"]  # (H_rgb, W_rgb)

    if dbg is not None:
        _safe_write_bytes(dbg / "label_map_rgb.png", base64.b64decode(sam3_out["label_png_base64"]))

    # ---- build label_map_depth ----
    if CFG.label_to_depth_mode == "warp":
        label_map_depth = _warp_label_rgb_to_depth(
            label_rgb=label_map_rgb,
            K_rgb=K_rgb,
            rgb_hw=rgb_hw,
            K_depth=K_depth,
            depth_hw=depth_hw,
        )
    else:
        label_map_depth = np.array(
            Image.fromarray(label_map_rgb.astype(np.uint16)).resize((dw, dh), resample=Image.NEAREST),
            dtype=np.uint16,
        )

    if dbg is not None:
        _safe_write_bytes(dbg / "label_map_depth.png", _label_map_png_bytes(label_map_depth))

    # ---- projection sanity ----
    proj_rgb = _proj_sanity_rgb(pts, T_C_W, K_rgb, label_map_rgb)
    logger.info("[PROJ_SANITY_RGB] %s", json.dumps(proj_rgb, ensure_ascii=False))

    proj_depth = _proj_sanity_depth_consistency(pts, T_C_W, K_depth, depth_map, label_map_depth)
    logger.info("[PROJ_SANITY_DEPTH] %s", json.dumps(proj_depth, ensure_ascii=False))

    if CFG.strict_depth_gate and proj_depth.get("depth_ok", 0) == 0:
        detail = {
            "error": "Depth-consistency sanity check failed: no points satisfy |depth(u,v)-z|<=eps.",
            "hint": "Usually means K_depth/pose/unit mismatch. Check: RGB_EXIF_TRANSPOSE, K scaling, pose transpose.",
            "proj_rgb": proj_rgb,
            "proj_depth": proj_depth,
            "debug_dir": str(dbg) if dbg is not None else None,
        }
        if dbg is not None:
            _safe_write_text(dbg / "fail_detail.json", json.dumps(detail, ensure_ascii=False, indent=2))
        raise HTTPException(status_code=400, detail=detail)

    # ---- SAM3 log ----
    logger.info("--- SAM3 Debug Info ---")
    logger.info("Prompt: '%s'", sam3_prompt)
    logger.info("Generated Masks: %d", sam3_out["mask_count"])
    if sam3_out["mask_count"] > 0:
        logger.info("Selected Index: %d (Score: %.4f)", sam3_out["best_idx"], sam3_out["best_score"])
        logger.info("All Scores: %s", [round(float(s), 3) for s in sam3_out["scores"]])
    else:
        logger.info("No masks found.")
    logger.info("-----------------------")

    # ---- Prepare Diffuser inputs ----
    labels_2d_u16, depth_2d, K_labels, K_depth2d, grid_tag = _prepare_diffuser_inputs(
        label_map_rgb=label_map_rgb,
        label_map_depth=label_map_depth,
        depth_map=depth_map,
        K_rgb=K_rgb,
        K_depth=K_depth,
        rgb_hw=rgb_hw,
        depth_hw=depth_hw,
    )

    # labels MUST be int32 indices
    labels_for_diffuser = labels_2d_u16.astype(np.int32, copy=True)  # values {0,1}

    # optional ignore background (2 == ignore for num_classes=2)
    if CFG.diffuser_ignore_background:
        labels_for_diffuser[labels_for_diffuser == 0] = 2  # ignore class index
        # keep 1 as foreground

    # ---- extra logs to ensure supervision is non-empty ----
    u, c = np.unique(labels_for_diffuser, return_counts=True)
    logger.info("[LABEL2D_UNIQ] %s", {int(uu): int(cc) for uu, cc in zip(u, c)})
    logger.info("[DEPTH2D_FINITE_RATIO] %.6f", float(np.isfinite(depth_2d).mean()))
    logger.info("[DIFFUSER_CFG] grid=%s sparse_depth=%s ignore_bg=%s num_classes=%d",
                grid_tag, CFG.diffuser_sparse_depth, CFG.diffuser_ignore_background, CFG.diffuser_num_classes_fixed)

    if dbg is not None:
        _safe_save_npy(dbg / f"diffuser_labels2d_{grid_tag}.npy", labels_for_diffuser.astype(np.int32))
        _safe_save_npy(dbg / f"diffuser_depth2d_{grid_tag}.npy", depth_2d.astype(np.float32))
        _safe_write_text(
            dbg / "diffuser_frame_cfg.json",
            json.dumps(
                {
                    "grid": grid_tag,
                    "labels_shape": list(labels_for_diffuser.shape),
                    "depth_shape": list(depth_2d.shape),
                    "K_labels": K_labels.tolist(),
                    "K_depth": K_depth2d.tolist(),
                    "sparse_depth": CFG.diffuser_sparse_depth,
                    "ignore_background": CFG.diffuser_ignore_background,
                    "num_classes_fixed": CFG.diffuser_num_classes_fixed,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        _safe_write_text(dbg / "T_C_W_cv.txt", np.array2string(T_C_W, precision=6, suppress_small=False))

    # ---- Diffuser ----
    t0 = time.time()
    diffuser_labels = _run_diffusion(
        points=pts,
        normals=nrm,
        labels_2d=labels_for_diffuser,
        depth_2d=depth_2d,
        K_labels=K_labels,
        K_depth=K_depth2d,
        T_C_W=T_C_W,
    )
    t1 = time.time()

    logger.info("Label diffusion took %.6f s", (t1 - t0))
    logger.info("[DIFFUSER_FRAME_GRID] %s labels=%s depth=%s", grid_tag, labels_for_diffuser.shape, depth_2d.shape)

    uniq, cnt = np.unique(diffuser_labels, return_counts=True)
    uniq_stat = {int(u): int(c) for u, c in zip(uniq, cnt)}
    logger.info("[DIFFUSER_LABEL_UNIQ] %s", uniq_stat)

    if dbg is not None:
        _safe_save_npy(dbg / "diffuser_labels.npy", diffuser_labels)
        _safe_write_text(dbg / "diffuser_uniq.json", json.dumps(uniq_stat, ensure_ascii=False, indent=2))

    # ---- VLM (optional) ----
    vlm_user_prompt = meta_obj.get("vlm_prompt", CFG.vlm_user_prompt_default)
    vlm_out = _call_vlm(
        rgb_img,
        vlm_user_prompt,
        extra={
            "vlm_enable_thinking": meta_obj.get("vlm_enable_thinking", False),
            "vlm_thinking_budget": meta_obj.get("vlm_thinking_budget"),
            "vlm_temperature": meta_obj.get("vlm_temperature", 0.2),
        },
    )

    result = {
        "ok": True,
        "points_count": int(pts.shape[0]),
        "normals_provided": nrm is not None,
        "proj_sanity_rgb": proj_rgb,
        "proj_sanity_depth": proj_depth,
        "intrinsics": {
            "rgb_hw": [H_rgb, W_rgb],
            "depth_hw": [dh, dw],
            "K_rgb": K_rgb.tolist(),
            "K_depth": K_depth.tolist(),
            "label_to_depth_mode": CFG.label_to_depth_mode,
            "rgb_exif_transpose": CFG.rgb_exif_transpose,
        },
        "sam3": {
            "prompt": sam3_prompt,
            "mask_count": int(sam3_out["mask_count"]),
            "scores": sam3_out["scores"],
            "boxes": sam3_out["boxes"],
            "debug_selection": {
                "selected_index": sam3_out["best_idx"],
                "selected_score": sam3_out["best_score"],
                "is_confident": sam3_out["best_score"] > CFG.sam3_confidence,
            },
            "label_png_base64": sam3_out["label_png_base64"],
            "label_shape_rgb": [int(label_map_rgb.shape[0]), int(label_map_rgb.shape[1])],
            "label_shape_depth": [int(label_map_depth.shape[0]), int(label_map_depth.shape[1])],
        },
        "diffuser": {
            "frame_grid": grid_tag,
            "labels_2d_shape": [int(labels_for_diffuser.shape[0]), int(labels_for_diffuser.shape[1])],
            "depth_2d_shape": [int(depth_2d.shape[0]), int(depth_2d.shape[1])],
            "K_labels": K_labels.tolist(),
            "K_depth": K_depth2d.tolist(),
            "sparse_depth": CFG.diffuser_sparse_depth,
            "ignore_background": CFG.diffuser_ignore_background,
            "num_classes_fixed": int(CFG.diffuser_num_classes_fixed),
            "max_iters": CFG.diffuser_max_iters,
            "px_to_pt_weight": float(CFG.diffuser_px_to_pt_weight),
            "point_labels_base64": _encode_npy_base64(diffuser_labels),
            "point_labels_dtype": "uint32",
            "point_labels_shape": [int(diffuser_labels.shape[0])],
            "label_uniq": uniq_stat,
        },
        "vlm": vlm_out,
        "debug_dir": str(dbg) if dbg is not None else None,
    }

    if dbg is not None:
        _safe_write_bytes(dbg / "response.json", json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
