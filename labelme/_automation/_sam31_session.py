from __future__ import annotations

import importlib
import inspect
import os
import tempfile
import site
from pathlib import Path
from typing import Literal
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray


class Sam31Session:
    _model_name: str
    _predictor: Literal["image", "video"]
    _device: str
    _confidence_threshold: float
    _model: Any | None
    _processor: Any | None

    def __init__(
        self,
        model_name: str = "sam3.1:latest",
        predictor: Literal["image", "video"] | None = None,
        confidence_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._predictor = self._resolve_predictor(
            model_name=model_name,
            predictor=predictor,
        )
        self._confidence_threshold = confidence_threshold
        self._device = device or self._pick_device()
        self._model = None
        self._processor = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def predictor(self) -> Literal["image", "video"]:
        return self._predictor

    def run(
        self,
        image: NDArray[np.uint8],
        image_id: str,
        texts: list[str],
        min_score: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]:
        del image_id
        self._ensure_loaded()

        if self._predictor == "video":
            return self._run_with_video_predictor(image=image, texts=texts, min_score=min_score)

        if not hasattr(self._processor, "set_image"):
            raise RuntimeError(
                f"SAM3.1 predictor '{self._predictor}' does not expose set_image(). "
                "This text-to-annotation path needs image-style processor APIs."
            )
        if not hasattr(self._processor, "set_text_prompt"):
            raise RuntimeError(
                f"SAM3.1 predictor '{self._predictor}' does not expose set_text_prompt()."
            )
        if not hasattr(self._processor, "reset_all_prompts"):
            raise RuntimeError(
                f"SAM3.1 predictor '{self._predictor}' does not expose reset_all_prompts()."
            )

        from PIL import Image

        image_pil = Image.fromarray(image)
        try:
            assert self._processor is not None
            state = self._processor.set_image(image_pil)

            boxes_out: list[list[float]] = []
            scores_out: list[float] = []
            labels_out: list[int] = []

            for label_idx, text in enumerate(texts):
                state = self._processor.set_text_prompt(prompt=text, state=state)
                boxes = self._tensor_to_list(state.get("boxes", []))
                masks = state.get("masks", [])
                scores = self._tensor_to_list(state.get("scores", []))

                for i, score_raw in enumerate(scores):
                    score = float(score_raw)
                    if score < min_score:
                        continue

                    box = boxes[i] if i < len(boxes) else None
                    mask_tensor = masks[i] if i < len(masks) else None
                    if box is None:
                        continue

                    if mask_tensor is not None and hasattr(mask_tensor, "ndim"):
                        if mask_tensor.ndim == 3:
                            mask_tensor = mask_tensor[0]
                        mask_bbox = self._mask_to_xyxy(mask_tensor)
                    else:
                        mask_bbox = None

                    xyxy = mask_bbox if mask_bbox is not None else [float(v) for v in box]
                    boxes_out.append(xyxy)
                    scores_out.append(score)
                    labels_out.append(label_idx)

                self._processor.reset_all_prompts(state)
        finally:
            image_pil.close()

        boxes_arr = np.asarray(boxes_out, dtype=np.float32)
        if boxes_arr.size == 0:
            boxes_arr = np.empty((0, 4), dtype=np.float32)
        scores_arr = np.asarray(scores_out, dtype=np.float32)
        labels_arr = np.asarray(labels_out, dtype=np.int32)
        return boxes_arr, scores_arr, labels_arr

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._processor is not None and (
            self._predictor == "video" or self._model is not None
        ):
            return

        self._apply_dtype_patch()
        try:
            sam3_builder_module = importlib.import_module("sam3.model_builder")

            if self._predictor == "image":
                sam3_processor_module = importlib.import_module(
                    "sam3.model.sam3_image_processor"
                )
                processor_class = getattr(sam3_processor_module, "Sam3Processor")
                builder = getattr(sam3_builder_module, "build_sam3_image_model")
            else:
                sam3_processor_module = importlib.import_module(
                    "sam3.model.sam3_video_predictor"
                )
                processor_class = self._pick_first_attr(
                    module=sam3_processor_module,
                    names=["Sam3VideoPredictor", "Sam3VideoPredictorMultiGPU"],
                )
                builder = None
        except Exception as e:
            if self._predictor == "video":
                raise RuntimeError(
                    "SAM3.1 video predictor is not available in this installed sam3 package. "
                    "Please use 'SAM3.1 Image' in the UI or install a sam3 build that includes "
                    "video processor modules. "
                    f"Original error: {e}"
                ) from e
            raise RuntimeError(
                "SAM3.1 backend is unavailable. Install dependencies in your env: "
                "pip install sam3 torch. "
                f"Original error: {e}"
            ) from e

        logger.info(
            "Loading SAM3.1 (PyTorch) predictor={} on device={}...",
            self._predictor,
            self._device,
        )
        bpe_path = self._resolve_bpe_path()
        if self._predictor == "image":
            assert builder is not None
            self._model = self._call_builder(
                builder=builder,
                kwargs={
                    "bpe_path": bpe_path,
                    "device": self._device,
                    "load_from_HF": True,
                },
            )

            self._processor = self._call_builder(
                builder=processor_class,
                kwargs={
                    "model": self._model,
                    "device": self._device,
                    "confidence_threshold": self._confidence_threshold,
                },
            )
        else:
            self._model = None
            self._processor = self._call_builder(
                builder=processor_class,
                kwargs={"bpe_path": bpe_path},
            )

    def _run_with_video_predictor(
        self,
        image: NDArray[np.uint8],
        texts: list[str],
        min_score: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]:
        from PIL import Image

        if self._processor is None:
            raise RuntimeError("SAM3.1 video predictor is not initialized")
        if not hasattr(self._processor, "start_session") or not hasattr(
            self._processor, "add_prompt"
        ):
            raise RuntimeError(
                "SAM3.1 video predictor backend does not expose start_session/add_prompt"
            )

        tmp_path: str | None = None
        image_pil = Image.fromarray(image)
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            image_pil.save(tmp_path)

            session_info = self._processor.start_session(tmp_path)
            session_id = (
                session_info["session_id"]
                if isinstance(session_info, dict) and "session_id" in session_info
                else session_info
            )

            boxes_out: list[list[float]] = []
            scores_out: list[float] = []
            labels_out: list[int] = []

            for label_idx, text in enumerate(texts):
                response = self._processor.add_prompt(
                    session_id=session_id,
                    frame_idx=0,
                    text=text,
                )
                outputs = response.get("outputs", {}) if isinstance(response, dict) else {}
                boxes_xyxy, probs_list = self._extract_video_boxes_and_scores(
                    outputs=outputs,
                    min_score=min_score,
                    image_height=int(image.shape[0]),
                    image_width=int(image.shape[1]),
                )
                for i, box_xyxy in enumerate(boxes_xyxy):
                    boxes_out.append(box_xyxy)
                    score = probs_list[i] if i < len(probs_list) else 0.0
                    scores_out.append(score)
                    labels_out.append(label_idx)

            try:
                self._processor.close_session(session_id)
            except Exception:
                logger.warning("Failed to close SAM3.1 video session: {}", session_id)
        finally:
            image_pil.close()
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    logger.warning("Failed to remove temporary image file: {}", tmp_path)

        boxes_arr = np.asarray(boxes_out, dtype=np.float32)
        if boxes_arr.size == 0:
            boxes_arr = np.empty((0, 4), dtype=np.float32)
        scores_arr = np.asarray(scores_out, dtype=np.float32)
        labels_arr = np.asarray(labels_out, dtype=np.int32)
        return boxes_arr, scores_arr, labels_arr

    @staticmethod
    def _resolve_predictor(
        model_name: str,
        predictor: Literal["image", "video"] | None,
    ) -> Literal["image", "video"]:
        if predictor is not None:
            return predictor
        if model_name.endswith(":video"):
            return "video"
        return "image"

    @staticmethod
    def _pick_first_attr(module: Any, names: list[str]) -> Any:
        for name in names:
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(f"None of {names} found in module {module!r}")

    @staticmethod
    def _call_builder(builder: Any, kwargs: dict[str, Any]) -> Any:
        signature = inspect.signature(builder)
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
        return builder(**filtered_kwargs)

    @staticmethod
    def _resolve_bpe_path() -> str | None:
        candidates: list[Path] = []
        for root in site.getsitepackages():
            root_path = Path(root)
            candidates.append(root_path / "assets" / "bpe_simple_vocab_16e6.txt.gz")
            candidates.append(
                root_path
                / "osam"
                / "_models"
                / "yoloworld"
                / "clip"
                / "bpe_simple_vocab_16e6.txt.gz"
            )

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _tensor_to_list(value: Any) -> list[Any]:
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if hasattr(value, "cpu") and hasattr(value, "tolist"):
            return value.cpu().tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    @classmethod
    def _extract_video_boxes_and_scores(
        cls,
        outputs: Any,
        min_score: float,
        image_height: int,
        image_width: int,
    ) -> tuple[list[list[float]], list[float]]:
        if not isinstance(outputs, dict):
            return [], []

        raw_scores = cls._pick_first_present(
            outputs,
            keys=["out_probs", "out_scores", "scores", "probs", "confidences"],
        )
        scores = cls._normalize_scores(raw_scores)

        raw_xywh = cls._pick_first_present(
            outputs,
            keys=["out_boxes_xywh", "boxes_xywh", "bbox_xywh"],
        )
        boxes_xywh = cls._normalize_boxes(raw_xywh)
        if boxes_xywh:
            boxes_xyxy: list[list[float]] = []
            scores_out: list[float] = []
            for i, box_xywh in enumerate(boxes_xywh):
                if len(box_xywh) != 4:
                    continue
                score = scores[i] if i < len(scores) else 0.0
                if score < min_score:
                    continue
                x, y, w, h = [float(v) for v in box_xywh]
                box_xyxy = cls._sanitize_box_xyxy(
                    [x, y, x + w, y + h],
                    image_height=image_height,
                    image_width=image_width,
                )
                if box_xyxy is None:
                    continue
                boxes_xyxy.append(box_xyxy)
                scores_out.append(score)
            return boxes_xyxy, scores_out

        raw_xyxy = cls._pick_first_present(
            outputs,
            keys=["out_boxes_xyxy", "out_boxes", "boxes_xyxy", "boxes", "bboxes"],
        )
        boxes_xyxy = cls._normalize_boxes(raw_xyxy)
        if boxes_xyxy:
            boxes_out = []
            scores_out = []
            for i, box_xyxy in enumerate(boxes_xyxy):
                if len(box_xyxy) != 4:
                    continue
                score = scores[i] if i < len(scores) else 0.0
                if score < min_score:
                    continue
                sanitized = cls._sanitize_box_xyxy(
                    [float(v) for v in box_xyxy],
                    image_height=image_height,
                    image_width=image_width,
                )
                if sanitized is None:
                    continue
                boxes_out.append(sanitized)
                scores_out.append(score)
            return boxes_out, scores_out

        raw_masks = cls._pick_first_present(
            outputs,
            keys=["out_masks", "masks", "mask_logits", "pred_masks"],
        )
        masks = cls._normalize_masks(raw_masks)
        boxes_out = []
        scores_out = []
        for i, mask in enumerate(masks):
            mask_bbox = cls._mask_to_xyxy(mask)
            if mask_bbox is None:
                continue
            score = scores[i] if i < len(scores) else 0.0
            if score < min_score:
                continue
            sanitized = cls._sanitize_box_xyxy(
                mask_bbox,
                image_height=image_height,
                image_width=image_width,
            )
            if sanitized is None:
                continue
            boxes_out.append(sanitized)
            scores_out.append(score)
        return boxes_out, scores_out

    @staticmethod
    def _sanitize_box_xyxy(
        box_xyxy: list[float],
        image_height: int,
        image_width: int,
    ) -> list[float] | None:
        if len(box_xyxy) != 4:
            return None

        x1, y1, x2, y2 = [float(v) for v in box_xyxy]

        # Some backends return normalized coordinates in [0, 1].
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1 *= float(image_width)
            x2 *= float(image_width)
            y1 *= float(image_height)
            y2 *= float(image_height)

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        xmin = max(0.0, min(xmin, float(image_width - 1)))
        xmax = max(0.0, min(xmax, float(image_width - 1)))
        ymin = max(0.0, min(ymin, float(image_height - 1)))
        ymax = max(0.0, min(ymax, float(image_height - 1)))

        # Ignore degenerate boxes that won't be visible.
        if xmax - xmin < 1.0 or ymax - ymin < 1.0:
            return None
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def _pick_first_present(outputs: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if key in outputs:
                return outputs[key]
        return []

    @classmethod
    def _normalize_boxes(cls, value: Any) -> list[list[float]]:
        if value is None:
            return []
        boxes = cls._tensor_to_list(value)
        if not boxes:
            return []
        if boxes and isinstance(boxes[0], (int, float)):
            if len(boxes) == 4:
                return [[float(v) for v in boxes]]
            return []
        return [[float(v) for v in box] for box in boxes if isinstance(box, (list, tuple))]

    @classmethod
    def _normalize_scores(cls, value: Any) -> list[float]:
        if value is None:
            return []
        scores = cls._tensor_to_list(value)
        if not scores:
            return []
        if scores and isinstance(scores[0], (int, float)):
            return [float(v) for v in scores]
        if scores and isinstance(scores[0], (list, tuple)) and len(scores[0]) == 1:
            return [float(v[0]) for v in scores]
        return []

    @classmethod
    def _normalize_masks(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        masks = cls._tensor_to_list(value)
        if not masks:
            return []
        if not isinstance(masks, list):
            return [masks]
        if masks and not isinstance(masks[0], (list, tuple)):
            return [value]
        return masks

    @staticmethod
    def _mask_to_xyxy(mask_tensor: Any) -> list[float] | None:
        try:
            import torch

            ys, xs = torch.where(mask_tensor > 0)
            if ys.numel() == 0 or xs.numel() == 0:
                return None
            return [
                float(xs.min().item()),
                float(ys.min().item()),
                float(xs.max().item()),
                float(ys.max().item()),
            ]
        except Exception:
            return None

    @staticmethod
    def _apply_dtype_patch() -> None:
        try:
            sam3_vitdet = importlib.import_module("sam3.model.vitdet")
        except Exception:
            return

        if getattr(sam3_vitdet.Mlp, "_sam3_dtype_patch_applied", False):
            return
        if not hasattr(sam3_vitdet, "addmm_act"):
            return

        def patched_forward(self: Any, x: Any) -> Any:
            x = sam3_vitdet.addmm_act(type(self.act), self.fc1, x)
            x = x.float()
            x = self.drop1(x)
            x = self.norm(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x

        sam3_vitdet.Mlp.forward = patched_forward
        sam3_vitdet.Mlp._sam3_dtype_patch_applied = True
