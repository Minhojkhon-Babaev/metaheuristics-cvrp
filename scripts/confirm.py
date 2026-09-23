"""Контрольный прогон: базовая конфигурация против комбинации лучших значений OFAT.

OFAT меняет по одному параметру, поэтому объединение «лучших» значений не
гарантирует улучшения — взаимодействия параметров нужно проверять отдельно.
Обе конфигурации гоняются на одних и тех же парах «инстанс–сид», сравнение парное.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvrp.alns import ALNSParams  # noqa: E402
from cvrp.experiment import load_optima, run_one  # noqa: E402
from tune import BASELINE, TUNING_SET  # noqa: E402

COMBINED = dict(BASELINE, n_neighbors=30, removal_max_frac=0.55, sa_start_frac=0.06, reaction=0.1, blink=0.0)


def _job(args):
    vrp_path, optimal, set_name, seed, params = args
    record, _ = run_one(vrp_path, optimal, set_name, seed, params)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    data = ROOT / "data"
    optima = load_optima(data / "optima.csv")
    configs = {
        "baseline": ALNSParams(time_limit=args.time_limit, stall_limit=10**9, **BASELINE),
        "combined": ALNSParams(time_limit=args.time_limit, stall_limit=10**9, **COMBINED),
    }

    jobs, meta = [], []
    for label, params in configs.items():
        for set_name, name in TUNING_SET:
            path = data / "instances" / set_name / f"{name}.vrp"
            for seed in range(1, args.seeds + 1):
                jobs.append((str(path), float(optima[name]["optimal"]), set_name, seed, params))
                meta.append(label)

    gaps: dict[str, dict[tuple[str, int], float]] = {label: {} for label in configs}
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(_job, jobs)):
            gaps[meta[i]][(record.instance, record.seed)] = record.gap
            rows.append({"config": meta[i], "instance": record.instance, "seed": record.seed, "gap": record.gap})

    keys = sorted(set(gaps["baseline"]) & set(gaps["combined"]))
    deltas = [gaps["combined"][k] - gaps["baseline"][k] for k in keys]
    mean_delta = statistics.mean(deltas)
    stderr = statistics.pstdev(deltas) / len(deltas) ** 0.5

    out = ROOT / "results" / "tuning_confirm.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["config", "instance", "seed", "gap"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"базовая:      {statistics.mean(gaps['baseline'].values()):.3f}%")
    print(f"комбинация:   {statistics.mean(gaps['combined'].values()):.3f}%")
    print(f"парная разность: {mean_delta:+.3f} ± {stderr:.3f} п.п. на {len(keys)} парах")
    if mean_delta + 2 * stderr < 0:
        verdict = "комбинация лучше базовой"
    elif mean_delta - 2 * stderr > 0:
        verdict = "комбинация ХУЖЕ базовой — параметры взаимодействуют, оставляем базовую"
    else:
        verdict = "различие в пределах шума"
    print("вывод:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
