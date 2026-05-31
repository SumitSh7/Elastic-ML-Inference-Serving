from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import requests


def _send_request(endpoint: str, image_path: Path) -> None:
    with image_path.open("rb") as image_file:
        response = requests.post(endpoint, files={"image": image_file}, timeout=5)
    response.raise_for_status()


def run_load_test(endpoint: str, image_dir: Path, rps_pattern: list[int], duration_s: int) -> None:
    image_paths = sorted(path for path in image_dir.glob("**/*") if path.is_file())
    if not image_paths:
        raise ValueError(f"No images found under {image_dir}")

    cycle = itertools.cycle(image_paths)
    start = time.time()
    second = 0

    while time.time() - start < duration_s:
        rps = rps_pattern[second % len(rps_pattern)]
        tick_start = time.time()

        with ThreadPoolExecutor(max_workers=max(1, rps)) as executor:
            futures = [executor.submit(_send_request, endpoint, next(cycle)) for _ in range(rps)]
            for future in as_completed(futures):
                future.result()

        second += 1
        elapsed = time.time() - tick_start
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple load tester for Elastic ML Inference Serving")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument(
        "--pattern",
        default="1,5,10,5",
        help="comma-separated requests-per-second pattern",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pattern = [int(item) for item in args.pattern.split(",") if item.strip()]
    run_load_test(args.endpoint, Path(args.image_dir), pattern, args.duration)


if __name__ == "__main__":
    main()
