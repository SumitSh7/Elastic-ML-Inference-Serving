import unittest

from app.autoscaler import Autoscaler, AutoscalerConfig


class AutoscalerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.autoscaler = Autoscaler(
            AutoscalerConfig(min_replicas=1, max_replicas=4, target_latency_s=0.5, scale_up_step=1, scale_down_step=1)
        )

    def test_scales_up_when_latency_is_high(self) -> None:
        self.assertEqual(self.autoscaler.desired_replicas(2, p95_latency_s=0.8, queue_depth=0), 3)

    def test_scales_up_when_queue_is_backed_up(self) -> None:
        self.assertEqual(self.autoscaler.desired_replicas(2, p95_latency_s=0.2, queue_depth=4), 3)

    def test_scales_down_when_idle_and_fast(self) -> None:
        self.assertEqual(self.autoscaler.desired_replicas(3, p95_latency_s=0.1, queue_depth=0), 2)

    def test_holds_when_within_target(self) -> None:
        self.assertEqual(self.autoscaler.desired_replicas(2, p95_latency_s=0.3, queue_depth=1), 2)

    def test_does_not_scale_down_without_latency_samples(self) -> None:
        self.assertEqual(self.autoscaler.desired_replicas(3, p95_latency_s=0.0, queue_depth=0), 3)

    def test_enforces_min_and_max_replicas(self) -> None:
        self.assertEqual(self.autoscaler.desired_replicas(4, p95_latency_s=0.9, queue_depth=10), 4)
        self.assertEqual(self.autoscaler.desired_replicas(1, p95_latency_s=0.1, queue_depth=0), 1)


if __name__ == "__main__":
    unittest.main()
