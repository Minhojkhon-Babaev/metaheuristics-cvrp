"""Сохраняет траектории сходимости (рекорд по времени) для набора инстансов."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvrp.alns import ALNSParams, solve  # noqa: E402
from cvrp.experiment import load_optima  # noqa: E402
from cvrp.instance import read_instance  # noqa: E402

CASES = [
    ("E", "E-n33-k4"),
    ("P", "P-n76-k5"),
    ("E", "E-n101-k8"),
    ("F", "F-n135-k7"),
    ("M", "M-n151-k12"),
    ("M", "M-n200-k16"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "results" / "convergence.json"))
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    data = Path(args.data)
    optima = load_optima(data / "optima.csv")
    params = ALNSParams(time_limit=args.time_limit, stall_limit=10**9)

    payload = {}
    for set_name, name in CASES:
        inst = read_instance(data / "instances" / set_name / f"{name}.vrp")
        result = solve(inst, params, seed=args.seed)
        optimal = float(optima[name]["optimal"])
        payload[name] = {
            "set": set_name,
            "n_customers": inst.n_customers,
            "optimal": optimal,
            "init_cost": result.init_cost,
            "history": [[t, c, 100.0 * (c - optimal) / optimal] for t, c in result.history],
            "final_gap": 100.0 * (result.cost - optimal) / optimal,
            "iterations": result.iterations,
            "destroy_usage": result.destroy_usage,
            "repair_usage": result.repair_usage,
            "routes": result.routes,
            "coords": inst.coords,
            "depot": inst.depot,
        }
        print(f"{name:<12} gap={payload[name]['final_gap']:.2f}% точек рекорда={len(result.history)}")

    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    print(f"записано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
