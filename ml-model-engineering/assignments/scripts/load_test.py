"""
scripts/load_test.py
---------------------
Very small load test: sends N sequential requests to the running
/predict endpoint and reports avg latency and p95 latency.

Prerequisite: the API must already be running, e.g.:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Run:
    python scripts/load_test.py --n 100 --url http://127.0.0.1:8000/predict
"""

from __future__ import annotations

import argparse
import statistics
import time

import requests

SAMPLE_PAYLOAD = {
    "customer_id": "9237-HQITU",
    "tenure_months": 2,
    "monthly_charges": 70.7,
    "total_charges": 151.65,
    "contract_type": "month_to_month",
    "payment_method": "electronic_check",
    "internet_service": "fiber_optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "senior_citizen": 0,
    "partner": 0,
    "dependents": 0,
    "phone_service": 1,
    "multiple_lines": 0,
    "paperless_billing": 1,
}


def run_load_test(url: str, n: int) -> None:
    latencies_ms = []
    errors = 0

    overall_start = time.perf_counter()
    for _ in range(n):
        start = time.perf_counter()
        try:
            resp = requests.post(url, json=SAMPLE_PAYLOAD, timeout=5)
            resp.raise_for_status()
        except Exception:
            errors += 1
            continue
        latencies_ms.append((time.perf_counter() - start) * 1000)
    overall_elapsed = time.perf_counter() - overall_start

    if not latencies_ms:
        print("All requests failed -- is the API running?")
        return

    latencies_ms.sort()
    p95_index = min(int(len(latencies_ms) * 0.95), len(latencies_ms) - 1)

    print(f"Requests sent: {n}  (errors: {errors})")
    print(f"Total wall time: {overall_elapsed:.2f}s  ({n / overall_elapsed:.1f} req/s)")
    print(f"Avg latency: {statistics.mean(latencies_ms):.2f} ms")
    print(f"p50 latency: {statistics.median(latencies_ms):.2f} ms")
    print(f"p95 latency: {latencies_ms[p95_index]:.2f} ms")
    print(f"Max latency: {max(latencies_ms):.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/predict")
    args = parser.parse_args()
    run_load_test(args.url, args.n)
