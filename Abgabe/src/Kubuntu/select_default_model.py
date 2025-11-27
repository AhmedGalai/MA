#!/usr/bin/env python3
"""
Wait for main_api to become healthy, then select a default model.

Usage:
  python select_default_model.py --url http://localhost:5000 --model cube.ply --timeout 180
"""
import argparse
import sys
import time

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:5000", help="Base URL of main API")
    p.add_argument("--model", default="cube.ply", help="Model name to select")
    p.add_argument("--timeout", type=int, default=180, help="Max seconds to wait for health")
    args = p.parse_args()

    base = args.url.rstrip("/")
    t0 = time.time()
    print(f"[select-default-model] Waiting for {base}/health ...")
    while True:
        try:
            r = requests.get(f"{base}/health", timeout=2)
            if r.status_code == 200:
                print("[select-default-model] Main API is healthy.")
                break
        except Exception:
            pass
        if time.time() - t0 > args.timeout:
            print("[select-default-model] Timed out waiting for Main API.")
            return 1
        time.sleep(2)

    try:
        r = requests.post(f"{base}/select_model", json={"model_name": args.model}, timeout=4)
        if r.status_code == 200:
            print(f"[select-default-model] Selected model: {args.model}")
            return 0
        else:
            try:
                print(f"[select-default-model] Failed to select model: HTTP {r.status_code} {r.text}")
            except Exception:
                print(f"[select-default-model] Failed to select model: HTTP {r.status_code}")
            return 2
    except Exception as e:
        print(f"[select-default-model] Error selecting model: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())

