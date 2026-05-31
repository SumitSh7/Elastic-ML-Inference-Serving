from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import time
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from app.autoscaler import Autoscaler
from app.ml_service import ResNet18Service


REQUEST_COUNT = Counter("inference_requests_total", "Total number of inference requests")
FAILED_REQUESTS = Counter("inference_failed_total", "Total number of failed inference requests")
REQUEST_LATENCY = Histogram("inference_latency_seconds", "End-to-end inference latency")
QUEUE_DEPTH = Gauge("dispatcher_queue_depth", "Current request queue depth")
ACTIVE_REPLICAS = Gauge("ml_active_replicas", "Current worker replicas")


@dataclass
class InferenceTask:
    image_bytes: bytes
    response_queue: asyncio.Queue


class Dispatcher:
    def __init__(self, initial_replicas: int = 1) -> None:
        self.queue: asyncio.Queue[InferenceTask] = asyncio.Queue()
        self._workers: list[asyncio.Task[Any]] = []
        self._latency_window: deque[float] = deque(maxlen=300)
        self._autoscaler = Autoscaler()
        self._service = ResNet18Service()
        self._lock = asyncio.Lock()
        self._loop_task: asyncio.Task[Any] | None = None
        self._initial_replicas = initial_replicas

    async def start(self) -> None:
        await self._set_replicas(self._initial_replicas)
        self._loop_task = asyncio.create_task(self._autoscale_loop())

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        await self._set_replicas(0)

    async def enqueue(self, image_bytes: bytes) -> dict[str, Any]:
        result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        await self.queue.put(InferenceTask(image_bytes=image_bytes, response_queue=result_queue))
        QUEUE_DEPTH.set(self.queue.qsize())
        return await result_queue.get()

    async def _worker(self) -> None:
        while True:
            task = await self.queue.get()
            QUEUE_DEPTH.set(self.queue.qsize())
            try:
                result = await asyncio.to_thread(self._service.predict, task.image_bytes)
                self._latency_window.append(result.latency_s)
                await task.response_queue.put(
                    {"label": result.label, "confidence": result.confidence, "latency_s": result.latency_s}
                )
            except Exception as exc:  # pragma: no cover - defensive path
                FAILED_REQUESTS.inc()
                await task.response_queue.put({"error": str(exc)})
            finally:
                self.queue.task_done()

    async def _set_replicas(self, replicas: int) -> None:
        async with self._lock:
            current = len(self._workers)
            if replicas > current:
                for _ in range(replicas - current):
                    self._workers.append(asyncio.create_task(self._worker()))
            elif replicas < current:
                for _ in range(current - replicas):
                    task = self._workers.pop()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            ACTIVE_REPLICAS.set(len(self._workers))

    def _p95_latency(self) -> float:
        if not self._latency_window:
            return 0.0
        sorted_latencies = sorted(self._latency_window)
        idx = max(0, int(0.95 * (len(sorted_latencies) - 1)))
        return sorted_latencies[idx]

    async def _autoscale_loop(self) -> None:
        while True:
            await asyncio.sleep(15)
            desired = self._autoscaler.desired_replicas(
                current_replicas=len(self._workers),
                p95_latency_s=self._p95_latency(),
                queue_depth=self.queue.qsize(),
            )
            await self._set_replicas(desired)


app = FastAPI(title="Elastic ML Inference Serving")
dispatcher = Dispatcher(initial_replicas=1)


@app.on_event("startup")
async def startup() -> None:
    await dispatcher.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    await dispatcher.stop()


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image file")

    REQUEST_COUNT.inc()
    start = time.perf_counter()
    result = await dispatcher.enqueue(data)
    REQUEST_LATENCY.observe(time.perf_counter() - start)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
