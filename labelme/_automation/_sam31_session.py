from __future__ import annotations

import importlib
import site
from pathlib import Path
from typing import Protocol
from typing import SupportsFloat
from typing import SupportsIndex
from typing import cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

_State: type = dict[str, object]


class _ProcessorProtocol(Protocol):
    def set_image(self, image_pil: object) -> dict[str, object]: ...

    def set_text_prompt(
        self, prompt: str, state: dict[str, object]
    ) -> dict[str, object]: ...

    def reset_all_prompts(self, state: dict[str, object]) -> None: ...


class Sam31Session:
    _model_name: str
    _device: str
    _confidence_threshold: float
    _model: object | None
    _processor: _ProcessorProtocol | None

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
                masks = self._tensor_to_list(state.get("masks", []))
                scores = self._tensor_to_list(state.get("scores", []))

                for i, score_raw in enumerate(scores):
                    score = self._to_float(score_raw)
                    if score is None:
                        continue
                    if score < min_score:
                        continue

                    box = boxes[i] if i < len(boxes) else None
                    mask_tensor = masks[i] if i < len(masks) else None
                    if box is None:
                        continue

                    if mask_tensor is not None and hasattr(mask_tensor, "ndim"):
                        if mask_tensor.ndim == 3:
                            first_item = self._first_item(mask_tensor)
                            if first_item is None:
                                continue
                            mask_tensor = first_item
                        mask_bbox = self._mask_to_xyxy(mask_tensor)
                    else:
                        mask_bbox = None

                    xyxy = (
                        mask_bbox if mask_bbox is not None else self._to_float_list(box)
                    )
                    if xyxy is None or len(xyxy) != 4:
                        continue
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
            torch_module = importlib.import_module("torch")
            cuda = getattr(torch_module, "cuda", None)
            is_available = getattr(cuda, "is_available", None)
            if callable(is_available) and bool(is_available()):
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
        self._processor = cast(
            _ProcessorProtocol,
            Sam3Processor(
                model=self._model,
                device=self._device,
                confidence_threshold=self._confidence_threshold,
            ),
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
    def _tensor_to_list(value: object) -> list[object]:
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()  # type: ignore[attr-defined]
        if hasattr(value, "cpu") and hasattr(value, "tolist"):
            return value.cpu().tolist()  # type: ignore[attr-defined]
        if hasattr(value, "tolist"):
            return value.tolist()  # type: ignore[attr-defined]
        if isinstance(value, list):
            return cast(list[object], value)
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _to_float(value: object) -> float | None:
        if isinstance(value, int | float | str):
            return float(value)
        if isinstance(value, SupportsFloat):
            return float(value)
        if isinstance(value, SupportsIndex):
            return float(value)
        try:
            return float(str(value))
        except Exception:
            return None

    @classmethod
    def _to_float_list(cls, value: object) -> list[float] | None:
        values = cls._tensor_to_list(value)
        floats: list[float] = []
        for item in values:
            number = cls._to_float(item)
            if number is None:
                return None
            floats.append(number)
        return floats

    @staticmethod
    def _mask_to_xyxy(mask_tensor: object) -> list[float] | None:
        try:
            torch_module = importlib.import_module("torch")
            where_fn = getattr(torch_module, "where", None)
            if not callable(where_fn):
                return None

            gt_fn = getattr(mask_tensor, "__gt__", None)
            if not callable(gt_fn):
                return None
            positives = gt_fn(0)

            result = where_fn(positives)
            if not isinstance(result, tuple) or len(result) != 2:
                return None
            ys, xs = result

            ys_numel = getattr(ys, "numel", None)
            xs_numel = getattr(xs, "numel", None)
            if not callable(ys_numel) or not callable(xs_numel):
                return None
            if ys_numel() == 0 or xs_numel() == 0:
                return None

            xmin = Sam31Session._tensor_scalar_minmax_item(xs, "min")
            ymin = Sam31Session._tensor_scalar_minmax_item(ys, "min")
            xmax = Sam31Session._tensor_scalar_minmax_item(xs, "max")
            ymax = Sam31Session._tensor_scalar_minmax_item(ys, "max")
            if None in (xmin, ymin, xmax, ymax):
                return None

            return [
                cast(float, xmin),
                cast(float, ymin),
                cast(float, xmax),
                cast(float, ymax),
            ]
        except Exception:
            return None

    @staticmethod
    def _tensor_scalar_minmax_item(tensor: object, op: str) -> float | None:
        fn = getattr(tensor, op, None)
        if not callable(fn):
            return None
        scalar = fn()
        item_fn = getattr(scalar, "item", None)
        if not callable(item_fn):
            return None
        try:
            return float(item_fn())
        except Exception:
            return None

    @staticmethod
    def _first_item(value: object) -> object | None:
        get_item = getattr(value, "__getitem__", None)
        if not callable(get_item):
            return None
        try:
            return get_item(0)
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

        def patched_forward(self: object, x: object) -> object:
            act = getattr(self, "act", None)
            fc1 = getattr(self, "fc1", None)
            x = sam3_vitdet.addmm_act(type(act), fc1, x)

            to_float = getattr(x, "float", None)
            if callable(to_float):
                x = to_float()

            for layer_name in ("drop1", "norm", "fc2", "drop2"):
                layer = getattr(self, layer_name, None)
                if callable(layer):
                    x = layer(x)
            return x

        sam3_vitdet.Mlp.forward = patched_forward
        sam3_vitdet.Mlp._sam3_dtype_patch_applied = True
