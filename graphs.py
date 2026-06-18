import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

PROM_URL = "http://localhost:9090"

END = datetime.now()
START = END - timedelta(minutes=20)

START_TS = START.timestamp()
END_TS = END.timestamp()

STEP = "15s"


def query_range(promql):
    r = requests.get(
        f"{PROM_URL}/api/v1/query_range",
        params={
            "query": promql,
            "start": START_TS,
            "end": END_TS,
            "step": STEP,
        },
    )

    result = r.json()["data"]["result"]

    print(f"Query returned {len(result)} series")

    dfs = []

    for i, series in enumerate(result):
        values = series["values"]

        df = pd.DataFrame(values, columns=["timestamp", "value"])

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="s"
        )

        df["value"] = pd.to_numeric(
            df["value"],
            errors="coerce"
        )

        df["series"] = i

        dfs.append(df)

    if not dfs:
        print(f"No data returned for query:\n{promql}")
        return pd.DataFrame(
            columns=["timestamp", "value", "series"]
        )

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------
# QUERIES
# ---------------------------------

QUEUE_QUERY = """
dispatcher_queue_size
"""

REPLICA_QUERY = """
kube_deployment_status_replicas{deployment="dispatcher"}
"""

CPU_QUERY = """
rate(container_cpu_usage_seconds_total{container="ml-app"}[5m])
"""

# ---------------------------------
# FETCH DATA
# ---------------------------------

queue_df = query_range(QUEUE_QUERY)
cpu_df = query_range(CPU_QUERY)
replica_df = query_range(REPLICA_QUERY)

print("CPU:", cpu_df.shape)
print("QUEUE:", queue_df.shape)
print("REPLICA:", replica_df.shape)

# ---------------------------------
# FIGURE
# ---------------------------------

plt.figure(figsize=(15, 10))

# -------------------------
# CPU
# -------------------------

ax1 = plt.subplot(311)

if not cpu_df.empty:
    for _, group in cpu_df.groupby("series"):
        ax1.plot(
            group["timestamp"],
            group["value"] * 100,
            label="CPU %"
        )

ax1.set_title("ML App CPU Usage")
ax1.set_ylabel("CPU")
ax1.grid(True)

# -------------------------
# QUEUE
# -------------------------

ax2 = plt.subplot(312)

if not queue_df.empty:
    ax2.plot(
        queue_df["timestamp"],
        queue_df["value"],
        linewidth=2
    )

ax2.fill_between(
    queue_df["timestamp"],
    queue_df["value"],
    alpha=0.3
)

ax2.set_title("Dispatcher Queue Size")
ax2.set_ylabel("Queue")
ax2.grid(True)

# -------------------------
# REPLICAS
# -------------------------

ax3 = plt.subplot(313)

if not replica_df.empty:
    ax3.step(
        replica_df["timestamp"],
        replica_df["value"],
        where="post",
        linewidth=2
    )

ax3.set_title("Dispatcher Replicas")
ax3.set_ylabel("Replicas")
ax3.grid(True)

plt.tight_layout()

plt.savefig(
    "autoscaler_results.png",
    dpi=300
)

print("Saved: autoscaler_results.png")

plt.show()