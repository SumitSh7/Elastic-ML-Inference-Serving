from dataclasses import dataclass


@dataclass(frozen=True)
class AutoscalerConfig:
    min_replicas: int = 1
    max_replicas: int = 10
    target_latency_s: float = 0.5
    scale_up_step: int = 1
    scale_down_step: int = 1


class Autoscaler:
    def __init__(self, config: AutoscalerConfig | None = None) -> None:
        self.config = config or AutoscalerConfig()

    def desired_replicas(self, current_replicas: int, p95_latency_s: float, queue_depth: int) -> int:
        replicas = max(self.config.min_replicas, min(self.config.max_replicas, current_replicas))

        if queue_depth > replicas or p95_latency_s > self.config.target_latency_s:
            return min(self.config.max_replicas, replicas + self.config.scale_up_step)

        low_latency_threshold = self.config.target_latency_s * 0.5
        if queue_depth == 0 and p95_latency_s <= low_latency_threshold:
            return max(self.config.min_replicas, replicas - self.config.scale_down_step)

        return replicas
