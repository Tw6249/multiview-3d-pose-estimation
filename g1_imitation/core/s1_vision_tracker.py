"""Module S1: 2D 27-Point Whole-Body Keypoint Extractor & 3D Reprojection Tracking State Machine.

Specification Reference: `docs/方案规范.md` — 模块 S1
Input:
  - `images_bgr`: list/array of N_cams BGR images of shape (H, W, 3), uint8.
  - `proj_matrices`: np.ndarray of shape (N_cams, 3, 4), camera projection matrices P_c = K_c [R_c | t_c].
  - Previous 3D state `X_{t-1}` (27, 3) and velocity `V_{t-1}` (27, 3) when in `TRACK_MODE`.
Output:
  - `VisionTrackerResult`:
    - `keypoints_2d`: np.ndarray of shape (N_cams, 27, 3) with [u, v, c] in original image pixel coordinates.
    - `mode_used`: str ("DETECT_MODE" or "TRACK_MODE").
    - `crop_bboxes_xyxy`: np.ndarray of shape (N_cams, 4).
    - `gpu_inference_time_ms`: float (batched ONNX Runtime GPU inference time in milliseconds).
    - `total_step_time_ms`: float (preprocessing + GPU inference + postprocessing in milliseconds).
Failure Behavior:
  - Automatically switches from `TRACK_MODE` back to `DETECT_MODE` if mean 2D keypoint confidence < 0.45
    or if fewer than 14 valid 3D keypoints are available (`min_valid_3d_points_for_track = 14`).
  - Immediately slices the 133 whole-body outputs down to the 27 Golden keypoints (`GOLDEN_27_INDICES`)
    before downstream triangulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .constants import GOLDEN_27_INDICES, load_config


def ensure_cuda_dlls_loaded() -> None:
    """Register NVIDIA CUDA/cuDNN DLL directories on Windows before importing onnxruntime."""
    if sys.platform != "win32":
        return
    site_packages = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "nvidia"
    if site_packages.exists():
        for bin_dir in site_packages.glob("*/bin"):
            bin_str = str(bin_dir.resolve())
            if bin_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_str + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(bin_str)
                except OSError:
                    pass


ensure_cuda_dlls_loaded()
import onnxruntime as ort  # noqa: E402
from rtmlib.tools.object_detection.yolox import YOLOX  # noqa: E402


@dataclass(frozen=True)
class VisionTrackerResult:
    """Single-frame output of Module S1."""
    keypoints_2d: np.ndarray
    mode_used: str
    crop_bboxes_xyxy: np.ndarray
    gpu_inference_time_ms: float
    total_step_time_ms: float


def compute_aspect_aligned_bbox_cs(
    bbox_xyxy: np.ndarray,
    expansion_factor: float = 1.35,
    model_input_wh: tuple[int, int] = (192, 256),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert `[x1, y1, x2, y2]` to `(center, scale, expanded_xyxy)` with `expansion_factor` (1.35x)
    and 3:4 aspect ratio alignment (`model_input_wh = (192, 256)`).
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(16.0, (x2 - x1) * float(expansion_factor))
    h = max(16.0, (y2 - y1) * float(expansion_factor))

    aspect_ratio = float(model_input_wh[0]) / float(model_input_wh[1])  # 192 / 256 = 0.75 (3:4)
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    center = np.array([cx, cy], dtype=np.float64)
    scale = np.array([w, h], dtype=np.float64)
    expanded_xyxy = np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float64)
    return center, scale, expanded_xyxy


def crop_and_resize_affine(
    img_bgr: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    model_input_wh: tuple[int, int] = (192, 256),
) -> np.ndarray:
    """Affine crop `img_bgr` around `(center, scale)` into `(H_in, W_in, 3)` matching RTMPose/RTMW."""
    w_in, h_in = int(model_input_wh[0]), int(model_input_wh[1])
    cx, cy = float(center[0]), float(center[1])
    sw, sh = float(scale[0]), float(scale[1])

    src_pts = np.array(
        [
            [cx, cy],
            [cx, cy - 0.5 * sh],
            [cx - 0.5 * sw, cy],
        ],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [
            [w_in * 0.5, h_in * 0.5],
            [w_in * 0.5, 0.0],
            [0.0, h_in * 0.5],
        ],
        dtype=np.float32,
    )
    M = cv2.getAffineTransform(src_pts, dst_pts)
    warped = cv2.warpAffine(img_bgr, M, (w_in, h_in), flags=cv2.INTER_LINEAR)
    return warped


def _suppress_person_boxes(
    boxes: np.ndarray,
    person_scores: np.ndarray,
    score_threshold: float = 0.30,
    iou_threshold: float = 0.65,
    max_output: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter COCO class-0 (`person`) boxes by `score_threshold` and apply NMS."""
    boxes_arr = np.asarray(boxes, dtype=np.float64)
    scores_arr = np.asarray(person_scores, dtype=np.float64)
    valid = (scores_arr > score_threshold) & np.isfinite(scores_arr) & np.all(np.isfinite(boxes_arr), axis=1)
    boxes_arr = boxes_arr[valid]
    scores_arr = scores_arr[valid]
    if len(boxes_arr) == 0:
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)
    xywh = boxes_arr.copy()
    xywh[:, 2:] -= xywh[:, :2]
    nms_ids = cv2.dnn.NMSBoxes(
        xywh.tolist(),
        scores_arr.tolist(),
        float(score_threshold),
        float(iou_threshold),
    )
    if len(nms_ids) == 0:
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)
    ids = np.asarray(nms_ids, dtype=np.int64).ravel()[:max_output]
    return boxes_arr[ids].copy(), scores_arr[ids].copy()


def _ensure_opset12_rtmw_onnx(source_onnx: Path) -> Path:
    """Ensure the FP16 RTMW ONNX model uses at least opset 12 and folds the two dynamic
    `Shape -> Slice -> Concat` shape subgraphs before `Reshape_344` and `Reshape_363` into ONNX's
    native `[0, 0, -1]` constant shape tensor. This assigns 100% of nodes to `CUDAExecutionProvider`
    with zero CPU fallback roundtrips and enables single-launch CUDA Graph capture.
    """
    if not source_onnx.exists():
        return source_onnx
    derived = source_onnx.parent / "pure_gpu" / source_onnx.name
    if derived.exists():
        return derived
    try:
        import onnx

        model = onnx.load(str(source_onnx))
        if len(model.opset_import) > 0 and model.opset_import[0].version < 12:
            model.opset_import[0].version = 12

        const_name = "const_reshape_0_0_m1"
        if not any(init.name == const_name for init in model.graph.initializer):
            init_t = onnx.numpy_helper.from_array(np.array([0, 0, -1], dtype=np.int64), name=const_name)
            model.graph.initializer.append(init_t)

        drop_names = {
            "Shape_338",
            "Constant_339",
            "Constant_340",
            "Slice_341",
            "Constant_342",
            "Concat_343",
            "Shape_360",
            "Slice_361",
            "Concat_362",
        }
        keep_nodes = []
        for node in model.graph.node:
            if node.name in drop_names:
                continue
            if node.name in ("Reshape_344", "Reshape_363") and len(node.input) >= 2:
                node.input[1] = const_name
            keep_nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(keep_nodes)

        derived.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(derived))
        return derived
    except Exception:
        return source_onnx


def _ensure_cpu_nms_yolox_onnx(source_onnx: Path) -> Path:
    """Return the pre-NMS YOLOX ONNX path (`cpu_nms/<name>.onnx`) so the backbone runs on GPU without
    80-class ONNX NMS CPU bounce.
    """
    derived = source_onnx.parent / "cpu_nms" / source_onnx.name
    if derived.exists():
        return derived
    try:
        import onnx

        model = onnx.load(str(source_onnx))
        nms_nodes = [n for n in model.graph.node if n.op_type == "NonMaxSuppression"]
        if len(nms_nodes) == 1:
            derived.parent.mkdir(parents=True, exist_ok=True)
            onnx.utils.extract_model(
                str(source_onnx),
                str(derived),
                [v.name for v in model.graph.input],
                list(nms_nodes[0].input[:2]),
            )
            return derived
    except Exception:
        pass
    return source_onnx


class RTMW27VisionTracker:
    """Module S1: Batched RTMW 27-point extractor with GPU YOLOX primary detection + 3D reprojection fallback."""

    DETECT_MODE = "DETECT_MODE"
    TRACK_MODE = "TRACK_MODE"

    def __init__(
        self,
        rtmw_onnx_path: str | Path | None = None,
        yolox_onnx_path: str | Path | None = None,
        model_input_wh: tuple[int, int] = (192, 256),
        bbox_expansion_factor: float = 1.25,
        fallback_min_mean_conf: float = 0.45,
        min_valid_3d_points_for_track: int = 14,
        simcc_split_ratio: float = 2.0,
        fov_border_margin_px: float = 0.0,
        device: str = "cuda",
        use_detector_every_frame: bool = True,
    ) -> None:
        cfg = load_config()
        ckpt_root = Path(cfg["runtime"]["rtmlib_cache_root"])
        s1_cfg = cfg["s1_vision_tracker"]

        if rtmw_onnx_path is None:
            fp16_candidate = ckpt_root / s1_cfg["rtmw_fp16_onnx"]
            fp32_candidate = ckpt_root / s1_cfg["rtmw_fp32_onnx"]
            rtmw_onnx_path = fp16_candidate if fp16_candidate.exists() else fp32_candidate
        if yolox_onnx_path is None:
            yolox_onnx_path = ckpt_root / s1_cfg["yolox_onnx"]

        self.rtmw_onnx_path = _ensure_opset12_rtmw_onnx(Path(rtmw_onnx_path).resolve())
        self.yolox_onnx_path = Path(yolox_onnx_path).resolve()
        self.model_input_wh = (int(model_input_wh[0]), int(model_input_wh[1]))
        self.bbox_expansion_factor = float(bbox_expansion_factor)
        self.fallback_min_mean_conf = float(fallback_min_mean_conf)
        self.min_valid_3d_points_for_track = int(min_valid_3d_points_for_track)
        self.simcc_split_ratio = float(simcc_split_ratio)
        self.fov_border_margin_px = float(fov_border_margin_px)
        self.use_detector_every_frame = bool(use_detector_every_frame)
        self.golden_indices = np.array(GOLDEN_27_INDICES, dtype=np.int64)

        avail_providers = ort.get_available_providers()
        self.use_cuda = bool(device == "cuda" and "CUDAExecutionProvider" in avail_providers)
        if self.use_cuda:
            rtmw_providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "enable_cuda_graph": "1",
                    },
                ),
                "CPUExecutionProvider",
            ]
            det_providers = [
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                "CPUExecutionProvider",
            ]
        else:
            rtmw_providers = ["CPUExecutionProvider"]
            det_providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        sess_opts.log_severity_level = 3
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.rtmw_onnx_path), sess_options=sess_opts, providers=rtmw_providers)
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.is_fp16 = "float16" in self.input_meta.type

        # Pre-allocate static Batch=6 CUDA Graph IoBinding buffers when running on CUDA
        self._cuda_max_batch: int = 6
        self._cuda_io_binding: ort.IOBinding | None = None
        self._cuda_input_ort: ort.OrtValue | None = None
        self._cuda_out_x_ort: ort.OrtValue | None = None
        self._cuda_out_y_ort: ort.OrtValue | None = None
        self._cuda_host_input_buf: np.ndarray | None = None
        if self.use_cuda and "CUDAExecutionProvider" in self.session.get_providers():
            w_in, h_in = self.model_input_wh
            in_dtype = np.float16 if self.is_fp16 else np.float32
            w_simcc = int(round(w_in * self.simcc_split_ratio))
            h_simcc = int(round(h_in * self.simcc_split_ratio))
            self._cuda_host_input_buf = np.zeros((self._cuda_max_batch, 3, h_in, w_in), dtype=in_dtype)
            self._cuda_input_ort = ort.OrtValue.ortvalue_from_numpy(self._cuda_host_input_buf, "cuda", 0)
            self._cuda_out_x_ort = ort.OrtValue.ortvalue_from_shape_and_type(
                (self._cuda_max_batch, 133, w_simcc), np.float32, "cuda", 0
            )
            self._cuda_out_y_ort = ort.OrtValue.ortvalue_from_shape_and_type(
                (self._cuda_max_batch, 133, h_simcc), np.float32, "cuda", 0
            )
            self._cuda_io_binding = self.session.io_binding()
            self._cuda_io_binding.bind_ortvalue_input(self.input_name, self._cuda_input_ort)
            self._cuda_io_binding.bind_ortvalue_output("simcc_x", self._cuda_out_x_ort)
            self._cuda_io_binding.bind_ortvalue_output("simcc_y", self._cuda_out_y_ort)

        # ImageNet normalization constants reshaped for (N, H, W, 3) -> (N, 3, H, W)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 1, 3)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 1, 3)

        # Initialize YOLOX detector directly via ONNX Runtime (avoiding rtmlib BaseTool's default spinning thread pool)
        self.detector: YOLOX | None = None
        self.detector_session: ort.InferenceSession | None = None
        self.detector_input_name: str = ""
        self.detector_uses_cpu_nms: bool = False
        if self.yolox_onnx_path.exists():
            derived_yolox = _ensure_cpu_nms_yolox_onnx(self.yolox_onnx_path)
            self.detector_uses_cpu_nms = derived_yolox != self.yolox_onnx_path
            if self.detector_uses_cpu_nms:
                self.detector_session = ort.InferenceSession(
                    str(derived_yolox),
                    sess_options=sess_opts,
                    providers=det_providers,
                )
                self.detector_input_name = self.detector_session.get_inputs()[0].name
            else:
                self.detector = YOLOX(
                    str(derived_yolox),
                    model_input_size=(416, 416),
                    backend="onnxruntime",
                    device="cpu",
                )

        self.mode: str = self.DETECT_MODE
        self.prev_points_3d_m: np.ndarray | None = None
        self.prev_velocities_ms: np.ndarray | None = None

        self._warmup_gpu(batch_size=6)

    def _warmup_gpu(self, batch_size: int = 6) -> None:
        if self.detector_session is not None:
            dummy_det = np.zeros((1, 3, 416, 416), dtype=np.float32)
            for _ in range(3):
                _ = self.detector_session.run(None, {self.detector_input_name: dummy_det})
        if self._cuda_io_binding is not None and self._cuda_input_ort is not None and self._cuda_host_input_buf is not None:
            self._cuda_host_input_buf.fill(0)
            self._cuda_input_ort.update_inplace(self._cuda_host_input_buf)
            for _ in range(10):
                self.session.run_with_iobinding(self._cuda_io_binding)
            self._cuda_io_binding.synchronize_outputs()
        else:
            w_in, h_in = self.model_input_wh
            dtype = np.float16 if self.is_fp16 else np.float32
            dummy = np.zeros((batch_size, 3, h_in, w_in), dtype=dtype)
            for _ in range(10):
                _ = self.session.run(None, {self.input_name: dummy})

    def reset(self) -> None:
        """Reset tracking state machine back to `DETECT_MODE`."""
        self.mode = self.DETECT_MODE
        self.prev_points_3d_m = None
        self.prev_velocities_ms = None

    def update_3d_state(
        self,
        points_3d_m: np.ndarray,
        velocities_ms: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> None:
        """Feed back the latest triangulated/filtered 3D skeleton state to drive `TRACK_MODE`."""
        pts = np.asarray(points_3d_m, dtype=np.float64)
        valid_count = int(np.sum(np.all(np.isfinite(pts), axis=1)))
        if valid_count < self.min_valid_3d_points_for_track:
            self.mode = self.DETECT_MODE
            return

        if self.prev_points_3d_m is not None:
            # Carry forward previous valid 3D coordinates only for transiently missing (NaN) joints;
            # all valid joints use their exact current 3D positions without any EMA lag.
            finite_mask = np.isfinite(pts)
            pts_next = np.where(finite_mask, pts, self.prev_points_3d_m)
            if velocities_ms is not None:
                vels = np.asarray(velocities_ms, dtype=np.float64)
            else:
                # Estimate whole-body root translation velocity from core trunk joints (shoulders 5,6 & hips 11,12)
                # rather than differentiating individual noisy fingertips/ears/toes.
                core_idx = [5, 6, 11, 12]
                core_diff = pts_next[core_idx] - self.prev_points_3d_m[core_idx]
                core_v = np.nanmean(core_diff / float(dt), axis=0)
                core_v = np.clip(np.where(np.isfinite(core_v), core_v, 0.0), -2.5, 2.5)
                vels = np.broadcast_to(core_v, pts.shape).copy()
            self.prev_points_3d_m = pts_next.copy()
        else:
            vels = np.zeros_like(pts) if velocities_ms is None else np.asarray(velocities_ms, dtype=np.float64)
            self.prev_points_3d_m = pts.copy()

        self.prev_velocities_ms = np.where(np.isfinite(vels), vels, 0.0).copy()
        self.mode = self.TRACK_MODE

    def _detect_person_candidates_single_image(
        self,
        img_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run YOLOX on a single image and return COCO `class 0 = person` candidate boxes `(M, 4)` and scores `(M,)`."""
        if self.detector_session is not None:
            h_img, w_img = img_bgr.shape[:2]
            ratio = min(416.0 / float(h_img), 416.0 / float(w_img))
            new_w, new_h = int(w_img * ratio), int(h_img * ratio)
            resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if new_w == 416 and new_h == 416:
                padded = resized
            else:
                padded = np.full((416, 416, 3), 114, dtype=np.uint8)
                padded[:new_h, :new_w] = resized
            inp = np.ascontiguousarray(padded.transpose(2, 0, 1))[None].astype(np.float32)
            outputs = self.detector_session.run(None, {self.detector_input_name: inp})
            boxes_scaled = np.asarray(outputs[0][0], dtype=np.float64) / float(ratio)
            person_scores = np.asarray(outputs[1][0, 0], dtype=np.float64)
            return _suppress_person_boxes(boxes_scaled, person_scores, score_threshold=0.30, iou_threshold=0.65)

        if self.detector is None:
            return np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.float64)

        det_res = self.detector(img_bgr)
        if det_res is None or len(det_res) == 0:
            return np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        arr = np.asarray(det_res, dtype=np.float64)
        boxes = arr[:, :4]
        scores = arr[:, 4] if arr.shape[1] >= 5 else np.ones(arr.shape[0], dtype=np.float64)
        return boxes, scores

    def detect_person_candidates(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All YOLOX `person` boxes `(M, 4)` [x1, y1, x2, y2] px and scores `(M,)` of one image, no selection
        (used by the multi-person association module S1b)."""
        return self._detect_person_candidates_single_image(img_bgr)

    def _detect_bboxes_per_camera(
        self,
        images_bgr: list[np.ndarray],
        reproj_bboxes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Obtain per-camera person bounding boxes `[x1, y1, x2, y2]` primarily from current-frame YOLOX,
        using `reproj_bboxes` (when provided in `TRACK_MODE`) strictly for multi-person disambiguation
        and missed-detection fallback (e.g. severe border truncation).
        """
        n_cams = len(images_bgr)
        out_bboxes = np.zeros((n_cams, 4), dtype=np.float64)

        for c, img in enumerate(images_bgr):
            h_img, w_img = img.shape[:2]
            cands, scores = self._detect_person_candidates_single_image(img)
            if len(cands) == 1:
                out_bboxes[c] = cands[0]
            elif len(cands) >= 2:
                if reproj_bboxes is not None:
                    ref_center = 0.5 * (reproj_bboxes[c, :2] + reproj_bboxes[c, 2:4])
                    cand_centers = 0.5 * (cands[:, :2] + cands[:, 2:4])
                    dists = np.linalg.norm(cand_centers - ref_center[None, :], axis=1)
                    out_bboxes[c] = cands[int(np.argmin(dists))]
                else:
                    areas = np.maximum(cands[:, 2] - cands[:, 0], 1.0) * np.maximum(cands[:, 3] - cands[:, 1], 1.0)
                    out_bboxes[c] = cands[int(np.argmax(areas * scores))]
            else:
                # len(cands) == 0: YOLOX missed the person (e.g., cut in half at image border in Cam 1)
                if reproj_bboxes is not None:
                    out_bboxes[c] = reproj_bboxes[c]
                else:
                    out_bboxes[c] = np.array([w_img * 0.25, h_img * 0.10, w_img * 0.75, h_img * 0.90], dtype=np.float64)

        return out_bboxes

    @staticmethod
    def _reproject_3d_bbox(
        pts_3d_m: np.ndarray,
        P_c: np.ndarray,
        w_img: int,
        h_img: int,
    ) -> np.ndarray:
        """Project valid 3D keypoints with their 3D anatomical surface radii into camera `P_c`
        and return the exact visible person bounding box `[x1, y1, x2, y2]` (zero temporal smoothing).
        """
        # Physical 3D anatomical surface radius (meters) around each of the 27 skeleton joint centers
        radii_all_m = np.full(pts_3d_m.shape[0], 0.05, dtype=np.float64)
        if pts_3d_m.shape[0] >= 27:
            radii_all_m[0:3] = 0.11         # nose / eyes
            radii_all_m[3:5] = 0.16         # ears / cranium back
            radii_all_m[[5, 6, 11, 12]] = 0.16  # shoulders / hips / trunk thickness
            radii_all_m[[7, 8, 13, 14]] = 0.08  # elbows / knees

        valid = np.all(np.isfinite(pts_3d_m), axis=1)
        pts = pts_3d_m[valid]
        rad = radii_all_m[valid]
        if pts.shape[0] >= 5:
            med = np.median(pts, axis=0)
            inliers = np.linalg.norm(pts - med, axis=1) <= 1.05
            if int(np.sum(inliers)) >= 5:
                pts = pts[inliers]
                rad = rad[inliers]

        # Extract true camera focal lengths (fx, fy) from 3x4 projection matrix P_c = K [R | t]
        p2 = P_c[2, :3]
        p2_norm = max(float(np.linalg.norm(p2)), 1e-8)
        fx = float(np.linalg.norm(np.cross(P_c[0, :3], p2))) / (p2_norm * p2_norm)
        fy = float(np.linalg.norm(np.cross(P_c[1, :3], p2))) / (p2_norm * p2_norm)

        pts_h = np.column_stack((pts, np.ones(pts.shape[0], dtype=np.float64)))
        uvw = (P_c @ pts_h.T).T
        z = np.maximum(uvw[:, 2], 0.1)
        u = uvw[:, 0] / z
        v = uvw[:, 1] / z
        rx = fx * rad / z
        ry = fy * rad / z

        # Only joints whose projected centers lie within the camera image FOV determine the visible 2D box
        # (prevents joints truncated below the floor border, e.g. v = 1268 > 1024 in Cam 0, from skewing u_max)
        in_fov = (u >= 0.0) & (u < float(w_img)) & (v >= 0.0) & (v < float(h_img))
        if int(np.sum(in_fov)) >= 3:
            u = u[in_fov]
            v = v[in_fov]
            rx = rx[in_fov]
            ry = ry[in_fov]

        x1 = float(np.clip(np.min(u - rx), 0.0, w_img - 2.0))
        y1 = float(np.clip(np.min(v - ry), 0.0, h_img - 2.0))
        x2 = float(np.clip(np.max(u + rx), x1 + 16.0, w_img - 1.0))
        y2 = float(np.clip(np.max(v + ry), y1 + 16.0, h_img - 1.0))
        return np.array([x1, y1, x2, y2], dtype=np.float64)

    def _predict_track_bboxes_per_camera(
        self,
        images_bgr: list[np.ndarray],
        proj_matrices: np.ndarray,
        dt: float = 0.01,
    ) -> np.ndarray:
        """Compute 3D constant-velocity extrapolated reprojection crop boxes (Section S1.2.2)."""
        assert self.prev_points_3d_m is not None
        vels = np.zeros_like(self.prev_points_3d_m) if self.prev_velocities_ms is None else self.prev_velocities_ms
        X_pred = self.prev_points_3d_m + vels * float(dt)

        n_cams = len(images_bgr)
        bboxes = np.zeros((n_cams, 4), dtype=np.float64)
        for c, img in enumerate(images_bgr):
            h_img, w_img = img.shape[:2]
            bboxes[c] = self._reproject_3d_bbox(X_pred, proj_matrices[c], w_img, h_img)
        return bboxes

    def infer_batch_from_bboxes(
        self,
        images_bgr: list[np.ndarray],
        raw_bboxes_xyxy: np.ndarray,
        mode_label: str,
    ) -> VisionTrackerResult:
        """Run batched crop, GPU ONNX inference, and 27-point SimCC decoding on `images_bgr`."""
        t_start = time.perf_counter()
        n_cams = len(images_bgr)
        w_in, h_in = self.model_input_wh

        centers = np.zeros((n_cams, 2), dtype=np.float64)
        scales = np.zeros((n_cams, 2), dtype=np.float64)
        expanded_bboxes = np.zeros((n_cams, 4), dtype=np.float64)
        warped_batch = np.empty((n_cams, h_in, w_in, 3), dtype=np.uint8)

        for c in range(n_cams):
            center, scale, exp_box = compute_aspect_aligned_bbox_cs(
                raw_bboxes_xyxy[c],
                expansion_factor=self.bbox_expansion_factor,
                model_input_wh=self.model_input_wh,
            )
            centers[c] = center
            scales[c] = scale
            expanded_bboxes[c] = exp_box
            warped_batch[c] = crop_and_resize_affine(
                images_bgr[c],
                center=center,
                scale=scale,
                model_input_wh=self.model_input_wh,
            )

        # Normalize and transpose (N, H, W, 3) -> (N, 3, H, W)
        batch_float = (warped_batch.astype(np.float32) - self.mean) / self.std
        batch_nchw = np.ascontiguousarray(batch_float.transpose(0, 3, 1, 2))
        if self.is_fp16:
            batch_input = batch_nchw.astype(np.float16)
        else:
            batch_input = batch_nchw

        if (
            self._cuda_io_binding is not None
            and self._cuda_input_ort is not None
            and self._cuda_host_input_buf is not None
            and n_cams <= self._cuda_max_batch
        ):
            self._cuda_host_input_buf[:n_cams] = batch_input
            if n_cams < self._cuda_max_batch:
                self._cuda_host_input_buf[n_cams:] = 0
            self._cuda_input_ort.update_inplace(self._cuda_host_input_buf)
            t_gpu0 = time.perf_counter()
            self.session.run_with_iobinding(self._cuda_io_binding)
            self._cuda_io_binding.synchronize_outputs()
            gpu_ms = (time.perf_counter() - t_gpu0) * 1000.0
            simcc_x_b6, simcc_y_b6 = self._cuda_io_binding.copy_outputs_to_cpu()
            simcc_x_full = simcc_x_b6[:n_cams]
            simcc_y_full = simcc_y_b6[:n_cams]
        else:
            t_gpu0 = time.perf_counter()
            simcc_x_full, simcc_y_full = self.session.run(None, {self.input_name: batch_input})
            gpu_ms = (time.perf_counter() - t_gpu0) * 1000.0

        # Immediately slice the 27 Golden keypoints from the 133 whole-body outputs (Section S1.2.1)
        simcc_x = simcc_x_full[:, self.golden_indices, :]  # (N_cams, 27, W_simcc)
        simcc_y = simcc_y_full[:, self.golden_indices, :]  # (N_cams, 27, H_simcc)

        x_locs = np.argmax(simcc_x, axis=2).astype(np.float64)
        y_locs = np.argmax(simcc_y, axis=2).astype(np.float64)
        x_vals = np.max(simcc_x, axis=2).astype(np.float64)
        y_vals = np.max(simcc_y, axis=2).astype(np.float64)

        # Use direct SimCC response scores (0.5 * (max_x + max_y)) in [0, 1] matching rtmlib,
        # without an extra sigmoid that artificially maps zero/noise responses to >= 0.50.
        raw_scores = 0.5 * (x_vals + y_vals)
        scores = np.clip(raw_scores, 0.0, 1.0)
        invalid_mask = raw_scores <= 0.0
        scores[invalid_mask] = 0.0

        # Transform crop coordinates back to original full-frame pixel coordinates
        u_crop = x_locs / self.simcc_split_ratio
        v_crop = y_locs / self.simcc_split_ratio
        u_img = (u_crop / float(w_in)) * scales[:, 0:1] + (centers[:, 0:1] - 0.5 * scales[:, 0:1])
        v_img = (v_crop / float(h_in)) * scales[:, 1:2] + (centers[:, 1:2] - 0.5 * scales[:, 1:2])

        # Any keypoint outside the physical camera image FOV [margin, W - margin) x [margin, H - margin)
        # is not a valid visual observation and must have its confidence zeroed out before S2 triangulation.
        margin = max(0.0, self.fov_border_margin_px)
        for c in range(n_cams):
            h_c, w_c = images_bgr[c].shape[:2]
            oob = (
                (u_img[c] < margin)
                | (u_img[c] >= float(w_c) - margin)
                | (v_img[c] < margin)
                | (v_img[c] >= float(h_c) - margin)
            )
            scores[c, oob] = 0.0

        keypoints_2d = np.stack((u_img, v_img, scores), axis=-1)  # (N_cams, 27, 3)
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return VisionTrackerResult(
            keypoints_2d=keypoints_2d,
            mode_used=mode_label,
            crop_bboxes_xyxy=np.asarray(raw_bboxes_xyxy, dtype=np.float64).copy(),
            gpu_inference_time_ms=float(gpu_ms),
            total_step_time_ms=float(total_ms),
        )

    def process_frame(
        self,
        images_bgr: list[np.ndarray],
        proj_matrices: np.ndarray,
        dt: float = 0.01,
        force_detect_mode: bool = False,
    ) -> VisionTrackerResult:
        """Process one multi-camera frame using the `DETECT_MODE` / `TRACK_MODE` state machine."""
        use_track = (
            not force_detect_mode
            and self.mode == self.TRACK_MODE
            and self.prev_points_3d_m is not None
            and int(np.sum(np.all(np.isfinite(self.prev_points_3d_m), axis=1))) >= self.min_valid_3d_points_for_track
        )

        if self.use_detector_every_frame:
            if use_track:
                reproj_bboxes = self._predict_track_bboxes_per_camera(images_bgr, proj_matrices, dt=dt)
                raw_bboxes = self._detect_bboxes_per_camera(images_bgr, reproj_bboxes=reproj_bboxes)
                res = self.infer_batch_from_bboxes(images_bgr, raw_bboxes, mode_label=self.TRACK_MODE)
                mean_conf = float(np.mean(res.keypoints_2d[:, :, 2]))
                if mean_conf < self.fallback_min_mean_conf:
                    self.mode = self.DETECT_MODE
                return res

            det_bboxes = self._detect_bboxes_per_camera(images_bgr, reproj_bboxes=None)
            return self.infer_batch_from_bboxes(images_bgr, det_bboxes, mode_label=self.DETECT_MODE)

        # Literal Section S1.2.2 baseline: skip YOLOX in TRACK_MODE and crop purely from 3D reprojection
        if use_track:
            raw_bboxes = self._predict_track_bboxes_per_camera(images_bgr, proj_matrices, dt=dt)
            res = self.infer_batch_from_bboxes(images_bgr, raw_bboxes, mode_label=self.TRACK_MODE)
            mean_conf = float(np.mean(res.keypoints_2d[:, :, 2]))
            if mean_conf < self.fallback_min_mean_conf:
                # Fallback to DETECT_MODE on occlusion or tracking loss
                self.mode = self.DETECT_MODE
                det_bboxes = self._detect_bboxes_per_camera(images_bgr)
                res = self.infer_batch_from_bboxes(images_bgr, det_bboxes, mode_label=self.DETECT_MODE)
            return res

        det_bboxes = self._detect_bboxes_per_camera(images_bgr)
        return self.infer_batch_from_bboxes(images_bgr, det_bboxes, mode_label=self.DETECT_MODE)
