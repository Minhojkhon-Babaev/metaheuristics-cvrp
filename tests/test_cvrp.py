"""Проверки корректности: парсинг, метрика, допустимость решений.

Запуск: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvrp.alns import ALNSParams, solve  # noqa: E402
from cvrp.construct import clarke_wright  # noqa: E402
from cvrp.instance import read_instance  # noqa: E402
from cvrp.localsearch import LocalSearch  # noqa: E402
from cvrp.solution import Solution, total_cost, validate  # noqa: E402

DATA = ROOT / "data" / "instances"


def all_instances() -> list[Path]:
    return sorted(p for s in ("E", "F", "M", "P") for p in (DATA / s).glob("*.vrp"))


def read_sol(path: Path) -> tuple[list[list[int]], float]:
    routes, cost = [], 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.lower().startswith("route"):
            routes.append([int(t) for t in line.partition(":")[2].split()])
        elif line.lower().startswith("cost"):
            cost = float(line.split()[1])
    return routes, cost


class TestInstances(unittest.TestCase):
    def test_all_instances_parse(self):
        paths = all_instances()
        self.assertEqual(len(paths), 45, "ожидается 45 инстансов в наборах E, F, M, P")
        for path in paths:
            inst = read_instance(path)
            self.assertEqual(len(inst.dist), inst.dimension)
            self.assertEqual(inst.demands[inst.depot], 0)
            self.assertTrue(all(inst.demands[c] <= inst.capacity for c in inst.customers))

    def test_distance_matches_published_optima(self):
        """Наша метрика должна воспроизводить стоимость эталонных решений CVRPLIB."""
        for path in all_instances():
            sol_path = path.with_suffix(".sol")
            if not sol_path.exists():
                continue
            inst = read_instance(path)
            routes, reported = read_sol(sol_path)
            self.assertAlmostEqual(total_cost(inst, routes), reported, msg=inst.name)

    def test_symmetry_and_zero_diagonal(self):
        inst = read_instance(DATA / "E" / "E-n51-k5.vrp")
        for i in range(inst.dimension):
            self.assertEqual(inst.dist[i][i], 0)
            for j in range(inst.dimension):
                self.assertEqual(inst.dist[i][j], inst.dist[j][i])


class TestSearch(unittest.TestCase):
    def test_clarke_wright_feasible(self):
        for path in all_instances():
            inst = read_instance(path)
            validate(inst, clarke_wright(inst))

    def test_local_search_improves_and_keeps_feasibility(self):
        import random

        for name in ("E/E-n51-k5", "P/P-n55-k10", "M/M-n121-k7", "F/F-n72-k4"):
            inst = read_instance(DATA / f"{name}.vrp")
            start = clarke_wright(inst)
            rng = random.Random(7)
            improved = LocalSearch(inst, inst.nearest_neighbors(20), rng).run(start)
            validate(inst, improved)
            self.assertLessEqual(improved.cost, start.cost)

    def test_alns_returns_valid_solution(self):
        for name in ("E/E-n22-k4", "P/P-n16-k8", "E/E-n13-k4"):
            inst = read_instance(DATA / f"{name}.vrp")
            result = solve(inst, ALNSParams(time_limit=3.0, stall_limit=5000), seed=3)
            loads = [sum(inst.demands[c] for c in r) for r in result.routes]
            validate(inst, Solution(result.routes, loads, result.cost))
            self.assertLessEqual(result.cost, result.init_cost)
            if inst.vehicles:
                self.assertEqual(result.n_routes, inst.vehicles)

    def test_alns_is_reproducible_for_fixed_seed(self):
        inst = read_instance(DATA / "E" / "E-n33-k4.vrp")
        params = ALNSParams(time_limit=10.0, stall_limit=2000)
        first = solve(inst, params, seed=11)
        second = solve(inst, params, seed=11)
        self.assertEqual(first.cost, second.cost)

    def test_no_solution_beats_published_optimum(self):
        """Решение лучше оптимума означало бы ошибку в метрике или в проверке."""
        import csv

        with (ROOT / "data" / "optima.csv").open(encoding="utf-8") as fh:
            optima = {row["instance"]: float(row["optimal"]) for row in csv.DictReader(fh)}
        for name in ("E/E-n22-k4", "E/E-n51-k5", "P/P-n50-k10"):
            inst = read_instance(DATA / f"{name}.vrp")
            result = solve(inst, ALNSParams(time_limit=5.0, stall_limit=10**9), seed=5)
            self.assertGreaterEqual(result.cost, optima[inst.name] - 1e-9, inst.name)


if __name__ == "__main__":
    unittest.main()
