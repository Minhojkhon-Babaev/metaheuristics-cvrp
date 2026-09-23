"""Подбор параметров ALNS и выбор критерия остановки.

Схема — OFAT (one-factor-at-a-time) вокруг базовой конфигурации: меняется
один параметр, остальные фиксированы. Настройка ведётся на отдельном
подмножестве инстансов (tuning set), чтобы итоговые замеры на всех 45
инстансах не были «подогнаны».

Отдельный режим `--study budget` строит кривую «качество — время» для
выбора бюджета остановки.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvrp.alns import ALNSParams  # noqa: E402
from cvrp.experiment import load_optima, run_one  # noqa: E402

# Подмножество для настройки: разные наборы и разные размерности.
TUNING_SET = [
    ("E", "E-n33-k4"),
    ("E", "E-n76-k8"),
    ("E", "E-n101-k14"),
    ("F", "F-n72-k4"),
    ("M", "M-n121-k7"),
    ("M", "M-n200-k17"),
    ("P", "P-n45-k5"),
    ("P", "P-n65-k10"),
    ("P", "P-n101-k4"),
]

BASELINE = dict(
    n_neighbors=20,
    removal_min_frac=0.10,
    removal_max_frac=0.40,
    sa_start_frac=0.03,
    reaction=0.3,
    blink=0.01,
    segment=100,
)

GRID = {
    "n_neighbors": [8, 12, 20, 30, 40],
    "removal_max_frac": [0.15, 0.25, 0.40, 0.55, 0.70],
    "sa_start_frac": [0.0, 0.01, 0.03, 0.06, 0.12],
    "reaction": [0.0, 0.1, 0.3, 0.6, 1.0],
    "blink": [0.0, 0.01, 0.03, 0.10],
}

BUDGETS = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]


def _job(args):
    vrp_path, optimal, set_name, seed, params = args
    record, _ = run_one(vrp_path, optimal, set_name, seed, params)
    return record


def build_jobs(data: Path, optima, seeds: int, configs: list[tuple[str, object, ALNSParams]]):
    jobs, meta = [], []
    for label, value, params in configs:
        for set_name, name in TUNING_SET:
            vrp_path = data / "instances" / set_name / f"{name}.vrp"
            optimal = float(optima[name]["optimal"])
            for seed in range(1, seeds + 1):
                jobs.append((str(vrp_path), optimal, set_name, seed, params))
                meta.append((label, value))
    return jobs, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--study", choices=["params", "budget"], default="params")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    optima = load_optima(data / "optima.csv")

    configs: list[tuple[str, object, ALNSParams]] = []
    if args.study == "params":
        base = ALNSParams(time_limit=args.time_limit, stall_limit=10**9, **BASELINE)
        configs.append(("baseline", "-", base))
        for param, values in GRID.items():
            for value in values:
                if value == BASELINE[param]:
                    continue
                kwargs = dict(BASELINE)
                kwargs[param] = value
                configs.append(
                    (param, value, ALNSParams(time_limit=args.time_limit, stall_limit=10**9, **kwargs))
                )
        out_path = out / "tuning_params.csv"
    else:
        for budget in BUDGETS:
            configs.append(
                ("time_limit", budget, ALNSParams(time_limit=budget, stall_limit=10**9, **BASELINE))
            )
        for stall in [500, 1000, 2000, 5000, 10000, 20000]:
            configs.append(
                ("stall_limit", stall, ALNSParams(time_limit=120.0, stall_limit=stall, **BASELINE))
            )
        out_path = out / "tuning_budget.csv"

    jobs, meta = build_jobs(data, optima, args.seeds, configs)
    print(f"конфигураций: {len(configs)}, запусков: {len(jobs)}")

    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers or None) as pool:
        for done, record in enumerate(pool.map(_job, jobs), start=1):
            label, value = meta[done - 1]
            row = dataclasses.asdict(record)
            row["param"] = label
            row["value"] = value
            rows.append(row)
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} за {time.perf_counter() - started:.0f}s", flush=True)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[tuple[str, object], list[float]] = {}
    for row in rows:
        summary.setdefault((row["param"], row["value"]), []).append(row["gap"])
    print(f"\n{'параметр':<18}{'значение':>10}{'средний gap, %':>18}")
    for (param, value), gaps in sorted(summary.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        print(f"{param:<18}{str(value):>10}{sum(gaps) / len(gaps):>18.3f}")
    print(f"\nзаписано: {out_path}; время: {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
