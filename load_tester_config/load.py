import time
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8001/add_to_queue"
IMG = "../imagenet-sample-images/Sasha.jpg"


def send(i):
    try:
        with open(IMG, "rb") as f:
            r = requests.post(URL, files={"image": f}, timeout=30)
            print(f"{i}: {r.status_code}")
    except Exception as e:
        print(f"{i}: ERROR - {e}")


# Read workload file
with open("../workload.txt", "r") as f:
    workload = [int(x) for x in f.read().split()]

print(f"Loaded {len(workload)} workload points")

request_id = 0

for second, rps in enumerate(workload):
    print(f"Second {second}: sending {rps} requests")

    with ThreadPoolExecutor(max_workers=max(rps, 1)) as ex:
        futures = []

        for _ in range(rps):
            futures.append(ex.submit(send, request_id))
            request_id += 1

        for future in futures:
            future.result()

    time.sleep(1)

print("DONE")