"""Вычислительный эксперимент на всех инстансах наборов E, F, M, P.

Каждый инстанс решается несколькими независимыми запусками (сидами),
результаты каждого запуска пишутся в results/runs.csv, лучшие найденные
маршруты — в results/solutions/.
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
from cvrp.experiment import RunRecord, TimeBudget, load_optima, run_one, write_solution  # noqa: E402

SETS = ("E", "F", "M", "P")


def _job(args):
    vrp_path, optimal, set_name, seed, params, budget = args
    return run_one(vrp_path, optimal, set_name, seed, params, budget)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--sets", nargs="*", default=list(SETS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0, help="0 = число ядер")
    parser.add_argument("--time-per-customer", type=float, default=0.25)
    parser.add_argument("--time-min", type=float, default=10.0)
    parser.add_argument("--time-max", type=float, default=60.0)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--removal-max-frac", type=float, default=0.40)
    parser.add_argument("--sa-start-frac", type=float, default=0.03)
    parser.add_argument("--reaction", type=float, default=0.3)
    parser.add_argument("--blink", type=float, default=0.01)
    parser.add_argument("--stall-limit", type=int, default=30000)
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    optima = load_optima(data / "optima.csv")

    params = ALNSParams(
        n_neighbors=args.neighbors,
        removal_max_frac=args.removal_max_frac,
        sa_start_frac=args.sa_start_frac,
        reaction=args.reaction,
        blink=args.blink,
        stall_limit=args.stall_limit,
    )
    budget = TimeBudget(args.time_per_customer, args.time_min, args.time_max)

    jobs = []
    for set_name in args.sets:
        for vrp_path in sorted((data / "instances" / set_name).glob("*.vrp")):
            name = vrp_path.stem
            optimal = float(optima[name]["optimal"])
            for seed in range(1, args.seeds + 1):
                jobs.append((str(vrp_path), optimal, set_name, seed, params, budget))

    print(f"запусков: {len(jobs)} (инстансов {len(jobs) // args.seeds}, сидов {args.seeds})")

    workers = args.workers or None
    started = time.perf_counter()
    records: list[RunRecord] = []
    best_routes: dict[str, tuple[float, list[list[int]]]] = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (record, routes) in enumerate(pool.map(_job, jobs), start=1):
            records.append(record)
            prev = best_routes.get(record.instance)
            if prev is None or record.cost < prev[0]:
                best_routes[record.instance] = (record.cost, routes)
            if done % 20 == 0 or done == len(jobs):
                elapsed = time.perf_counter() - started
                print(f"  {done}/{len(jobs)} за {elapsed:.0f}s", flush=True)

    records.sort(key=lambda r: (r.set, r.n_customers, r.instance, r.seed))
    runs_path = out / f"runs_{args.tag}.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in dataclasses.fields(RunRecord)])
        writer.writeheader()
        for record in records:
            writer.writerow(dataclasses.asdict(record))

    for name, (cost, routes) in best_routes.items():
        write_solution(out / "solutions" / f"{name}.sol", routes, cost)

    gaps = [r.gap for r in records]
    print(f"\nзаписано: {runs_path}")
    print(f"средний gap по всем запускам: {sum(gaps) / len(gaps):.3f}%")
    best_per_instance = {}
    for r in records:
        best_per_instance.setdefault(r.instance, []).append(r.gap)
    best_gaps = [min(v) for v in best_per_instance.values()]
    print(f"средний gap по лучшему из {args.seeds} запусков: {sum(best_gaps) / len(best_gaps):.3f}%")
    print(f"общее время: {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
