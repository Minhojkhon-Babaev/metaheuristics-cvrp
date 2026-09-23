"""Быстрая проверка работоспособности: несколько инстансов, короткий бюджет."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvrp.alns import ALNSParams, solve  # noqa: E402
from cvrp.construct import clarke_wright  # noqa: E402
from cvrp.instance import read_instance  # noqa: E402
from cvrp.solution import validate  # noqa: E402

CASES = [
    ("E/E-n13-k4.vrp", 247),
    ("E/E-n22-k4.vrp", 375),
    ("E/E-n51-k5.vrp", 521),
    ("P/P-n76-k5.vrp", 627),
    ("E/E-n101-k8.vrp", 815),
    ("M/M-n200-k16.vrp", 1274),
]


def main() -> int:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    params = ALNSParams(time_limit=budget)
    gaps = []
    for rel, optimum in CASES:
        inst = read_instance(ROOT / "data" / "instances" / rel)
        cw = clarke_wright(inst)
        started = time.perf_counter()
        res = solve(inst, params, seed=1)
        from cvrp.solution import Solution

        validate(inst, Solution(res.routes, [sum(inst.demands[c] for c in r) for r in res.routes], res.cost))
        gap = 100.0 * (res.cost - optimum) / optimum
        gaps.append(gap)
        print(
            f"{inst.name:<14} n={inst.n_customers:>3} CW={cw.cost:>7.0f} "
            f"ALNS={res.cost:>7.0f} opt={optimum:>7} gap={gap:>6.2f}% "
            f"K={res.n_routes}/{inst.vehicles} it={res.iterations:>6} "
            f"t={time.perf_counter() - started:.1f}s t*={res.time_to_best:.1f}s"
        )
    print(f"средний gap: {sum(gaps) / len(gaps):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
