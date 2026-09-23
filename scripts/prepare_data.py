"""Сборка локального датасета CVRPLIB (наборы E, F, M, P).

Инстансы берутся из зеркала CVRPLIB, копируются в data/instances/<SET>/ и
проверяются: стоимость эталонного решения из .sol пересчитывается нашей
функцией расстояния. Совпадение подтверждает, что мы используем ту же метрику
(EUC_2D с округлением до целого), что и опубликованные оптимумы.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvrp.instance import read_instance  # noqa: E402
from cvrp.solution import total_cost  # noqa: E402

MIRROR_URL = "https://github.com/zhu-he/cvrp-data.git"
SETS = ("E", "F", "M", "P")
_COST_RE = re.compile(r"[Cc]ost\s+(\d+(?:\.\d+)?)")


def fetch_mirror(cache: Path) -> Path:
    if not cache.exists():
        subprocess.run(["git", "clone", "--depth", "1", MIRROR_URL, str(cache)], check=True)
    return cache


def read_solution_routes(path: Path) -> tuple[list[list[int]], float | None]:
    routes: list[list[int]] = []
    reported: float | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.lower().startswith("route"):
            _, _, body = line.partition(":")
            nodes = [int(tok) for tok in body.split()]
            if nodes:
                routes.append(nodes)
        else:
            match = _COST_RE.search(line)
            if match:
                reported = float(match.group(1))
    return routes, reported


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/tmp/cvrp-data-mirror", help="куда клонировать зеркало")
    parser.add_argument("--out", default=str(ROOT / "data"), help="каталог назначения")
    args = parser.parse_args()

    mirror = fetch_mirror(Path(args.cache))
    out = Path(args.out)
    rows = []
    mismatches = []

    for set_name in SETS:
        src_dir = mirror / set_name
        dst_dir = out / "instances" / set_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        for vrp_path in sorted(src_dir.glob("*.vrp")):
            shutil.copy2(vrp_path, dst_dir / vrp_path.name)
            sol_path = vrp_path.with_suffix(".sol")
            inst = read_instance(vrp_path)

            recomputed = None
            reported = None
            if sol_path.exists():
                shutil.copy2(sol_path, dst_dir / sol_path.name)
                routes, reported = read_solution_routes(sol_path)
                recomputed = total_cost(inst, routes)
                if reported is not None and abs(recomputed - reported) > 1e-6:
                    mismatches.append((inst.name, reported, recomputed))

            best_known = reported if reported is not None else inst.optimal
            rows.append(
                {
                    "instance": inst.name,
                    "set": set_name,
                    "n_customers": inst.n_customers,
                    "capacity": inst.capacity,
                    "vehicles": inst.vehicles or "",
                    "total_demand": sum(inst.demands),
                    "min_vehicles": -(-sum(inst.demands) // inst.capacity),
                    "optimal": best_known,
                    "optimal_from_comment": inst.optimal if inst.optimal is not None else "",
                    "sol_cost_recomputed": recomputed if recomputed is not None else "",
                }
            )

    rows.sort(key=lambda r: (r["set"], r["n_customers"], r["instance"]))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "optima.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_set: dict[str, int] = {}
    for row in rows:
        per_set[row["set"]] = per_set.get(row["set"], 0) + 1
    print(f"инстансов сохранено: {len(rows)} " + ", ".join(f"{k}={v}" for k, v in sorted(per_set.items())))

    if mismatches:
        print(f"ВНИМАНИЕ: расхождение метрики на {len(mismatches)} инстансах")
        for name, reported, recomputed in mismatches[:10]:
            print(f"  {name}: в .sol {reported}, пересчитано {recomputed}")
        return 1

    print("метрика совпала с эталонными решениями на всех инстансах (EUC_2D, округление до целого)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
