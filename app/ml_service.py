from __future__ import annotations

import io
import time
from dataclasses import dataclass

from PIL import Image
import torch
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class PredictionResult:
    label: str
    confidence: float
    latency_s: float


class ResNet18Service:
    def __init__(self) -> None:
        self._weights = ResNet18_Weights.DEFAULT
        self._preprocess = self._weights.transforms()
        self._labels = self._weights.meta["categories"]
        self._model = resnet18(weights=self._weights)
        self._model.eval()

    def predict(self, image_bytes: bytes) -> PredictionResult:
        start = time.perf_counter()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0)

        with torch.inference_mode():
            logits = self._model(tensor)
            probs = torch.nn.functional.softmax(logits[0], dim=0)
            idx = int(torch.argmax(probs).item())
            confidence = float(probs[idx].item())

        return PredictionResult(
            label=self._labels[idx],
            confidence=confidence,
            latency_s=time.perf_counter() - start,
        )
