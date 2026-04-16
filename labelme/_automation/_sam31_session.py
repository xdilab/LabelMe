from __future__ import annotations

import importlib
import site
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray


class Sam31Session:
    _model_name: str
    _device: str
    _confidence_threshold: float
    _model: Any | None
    _processor: Any | None

    def __init__(
        self,
        model_name: str = "sam3.1:latest",
        confidence_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._device = device or self._pick_device()
        self._model = None
        self._processor = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def run(
        self,
        image: NDArray[np.uint8],
        image_id: str,
        texts: list[str],
        min_score: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]:
        del image_id
        self._ensure_loaded()

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

                    xyxy = (
                        mask_bbox if mask_bbox is not None else [float(v) for v in box]
                    )
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
        if self._model is not None and self._processor is not None:
            return

        self._apply_dtype_patch()
        try:
            sam3_processor_module = importlib.import_module(
                "sam3.model.sam3_image_processor"
            )
            sam3_builder_module = importlib.import_module("sam3.model_builder")
            Sam3Processor = sam3_processor_module.Sam3Processor
            build_sam3_image_model = sam3_builder_module.build_sam3_image_model
        except Exception as e:
            raise RuntimeError(
                "SAM3.1 backend is unavailable. Install dependencies in your env: "
                "pip install sam3 torch"
            ) from e

        logger.info("Loading SAM3.1 (PyTorch) on device={}...", self._device)
        bpe_path = self._resolve_bpe_path()
        self._model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=self._device,
            load_from_HF=True,
        )
        self._processor = Sam3Processor(
            model=self._model,
            device=self._device,
            confidence_threshold=self._confidence_threshold,
        )

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
