"""Служебный слой для вычислительных экспериментов: запуск, бюджет, метрики."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .alns import ALNSParams, solve
from .instance import Instance, read_instance
from .solution import Solution, validate


@dataclass
class TimeBudget:
    """Бюджет времени как функция размерности: T(n) = clip(a*n, t_min, t_max)."""

    per_customer: float = 0.25
    minimum: float = 10.0
    maximum: float = 60.0

    def for_instance(self, inst: Instance) -> float:
        return min(max(self.per_customer * inst.n_customers, self.minimum), self.maximum)


@dataclass
class RunRecord:
    instance: str
    set: str
    n_customers: int
    seed: int
    cost: float
    optimal: float
    gap: float
    n_routes: int
    k_reference: int
    init_cost: float
    init_gap: float
    iterations: int
    runtime: float
    time_to_best: float
    iter_to_best: int
    time_limit: float
    stop_reason: str


def load_optima(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open(encoding="utf-8") as fh:
        return {row["instance"]: row for row in csv.DictReader(fh)}


def run_one(
    vrp_path: str,
    optimal: float,
    set_name: str,
    seed: int,
    params: ALNSParams,
    budget: TimeBudget | None = None,
) -> tuple[RunRecord, list[list[int]]]:
    inst = read_instance(vrp_path)
    if budget is not None:
        params = ALNSParams(**{**asdict(params), "time_limit": budget.for_instance(inst)})

    result = solve(inst, params, seed=seed)
    loads = [sum(inst.demands[c] for c in r) for r in result.routes]
    validate(inst, Solution(result.routes, loads, result.cost))
    if inst.vehicles and result.n_routes != inst.vehicles:
        raise ValueError(
            f"{inst.name}: парк {result.n_routes} не совпал с требуемыми {inst.vehicles} машинами"
        )

    record = RunRecord(
        instance=result.instance,
        set=set_name,
        n_customers=result.n_customers,
        seed=seed,
        cost=result.cost,
        optimal=optimal,
        gap=100.0 * (result.cost - optimal) / optimal,
        n_routes=result.n_routes,
        k_reference=inst.vehicles or 0,
        init_cost=result.init_cost,
        init_gap=100.0 * (result.init_cost - optimal) / optimal,
        iterations=result.iterations,
        runtime=result.runtime,
        time_to_best=result.time_to_best,
        iter_to_best=result.iter_to_best,
        time_limit=params.time_limit,
        stop_reason=result.stop_reason,
    )
    return record, result.routes


def write_solution(path: Path, routes: list[list[int]], cost: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Route #{i + 1}: " + " ".join(str(c) for c in route) for i, route in enumerate(routes)]
    lines.append(f"Cost {int(cost)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
