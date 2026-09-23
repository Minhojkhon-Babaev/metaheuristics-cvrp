"""ALNS (Adaptive Large Neighborhood Search) для CVRP.

Схема одной итерации:
    текущее решение -> оператор разрушения -> оператор восстановления
    -> гранулярный локальный поиск -> критерий приёма (имитация отжига).

Веса операторов обновляются по сегментам на основе того, какой результат
они приносили (новый рекорд / улучшение / принятое ухудшение / отказ).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from .construct import clarke_wright, fit_fleet
from .instance import Instance
from .localsearch import LocalSearch
from .operators import (
    DESTROY_OPERATORS,
    REPAIR_OPERATORS,
    Repairer,
    apply_destroy,
    apply_repair,
    sa_temperature,
)
from .solution import Solution, make_solution


@dataclass
class ALNSParams:
    # соседства и размер разрушения
    n_neighbors: int = 20
    removal_min_frac: float = 0.10
    removal_max_frac: float = 0.40
    removal_abs_min: int = 4
    removal_abs_max: int = 50
    blink: float = 0.01

    # имитация отжига
    sa_start_frac: float = 0.03
    sa_end_frac: float = 0.0002

    # адаптивность весов
    segment: int = 100
    reaction: float = 0.3
    score_best: float = 33.0
    score_better: float = 13.0
    score_accepted: float = 9.0

    # критерий остановки
    time_limit: float = 20.0
    max_iterations: int = 10_000_000
    stall_limit: int = 30_000
    restart_stall: int = 4_000

    destroy_ops: tuple[str, ...] = tuple(DESTROY_OPERATORS)
    repair_ops: tuple[str, ...] = REPAIR_OPERATORS


@dataclass
class ALNSResult:
    instance: str
    n_customers: int
    cost: float
    routes: list[list[int]]
    n_routes: int
    fleet_target: int
    iterations: int
    runtime: float
    time_to_best: float
    iter_to_best: int
    init_cost: float
    stop_reason: str
    history: list[tuple[float, float]] = field(default_factory=list)
    destroy_usage: dict[str, int] = field(default_factory=dict)
    repair_usage: dict[str, int] = field(default_factory=dict)


def _roulette(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if pick <= acc:
            return i
    return len(weights) - 1


def solve(inst: Instance, params: ALNSParams | None = None, seed: int = 0) -> ALNSResult:
    params = params or ALNSParams()
    rng = random.Random(seed)
    start_time = time.perf_counter()

    neighbors = inst.nearest_neighbors(min(params.n_neighbors, max(1, inst.n_customers - 1)))
    repairer = Repairer(inst, neighbors, rng, blink=params.blink)

    # Размер парка в наборах E/F/M/P задан инстансом (k в имени) — именно для
    # него опубликованы оптимумы. Если столько маршрутов не набирается,
    # работаем с минимальным достижимым числом и сообщаем его в результате.
    current = clarke_wright(inst, rng, noise=0.0 if seed == 0 else 0.10)
    fleet = inst.vehicles or len(current.routes)
    current = fit_fleet(inst, current, fleet)

    local_search = LocalSearch(inst, neighbors, rng, fleet_min=fleet)
    current = local_search.run(current)
    current = fit_fleet(inst, current, fleet)
    best = current.copy()
    init_cost = current.cost
    fleet_target = fleet

    # Стартовая температура: решение на sa_start_frac хуже стартового
    # принимается с вероятностью 0.5 (схема Ропке — Писингера).
    t_start = max(params.sa_start_frac * init_cost / math.log(2.0), 1e-6)
    t_end = max(params.sa_end_frac * init_cost, 1e-9)
    t_start = max(t_start, t_end * 10)

    n_destroy = len(params.destroy_ops)
    n_repair = len(params.repair_ops)
    destroy_weights = [1.0] * n_destroy
    repair_weights = [1.0] * n_repair
    destroy_scores = [0.0] * n_destroy
    repair_scores = [0.0] * n_repair
    destroy_counts = [0] * n_destroy
    repair_counts = [0] * n_repair
    destroy_usage = {name: 0 for name in params.destroy_ops}
    repair_usage = {name: 0 for name in params.repair_ops}

    q_min = max(params.removal_abs_min, int(params.removal_min_frac * inst.n_customers))
    q_max = max(q_min + 1, min(params.removal_abs_max, int(params.removal_max_frac * inst.n_customers)))
    q_max = min(q_max, inst.n_customers)

    history: list[tuple[float, float]] = [(0.0, init_cost)]
    iterations = 0
    time_to_best = 0.0
    iter_to_best = 0
    since_improve = 0
    stop_reason = "time_limit"

    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= params.time_limit:
            stop_reason = "time_limit"
            break
        if iterations >= params.max_iterations:
            stop_reason = "max_iterations"
            break
        if since_improve >= params.stall_limit:
            stop_reason = "stall_limit"
            break

        iterations += 1
        temperature = sa_temperature(t_start, t_end, elapsed / params.time_limit)

        di = _roulette(destroy_weights, rng)
        ri = _roulette(repair_weights, rng)
        destroy_name = params.destroy_ops[di]
        repair_name = params.repair_ops[ri]
        destroy_counts[di] += 1
        repair_counts[ri] += 1
        destroy_usage[destroy_name] += 1
        repair_usage[repair_name] += 1

        q = rng.randint(q_min, q_max)
        routes, loads, removed = apply_destroy(inst, current, destroy_name, q, rng)

        if not apply_repair(repairer, repair_name, routes, loads, removed, fleet):
            since_improve += 1
            continue
        candidate = make_solution(inst, routes)
        if len(candidate.routes) != fleet:
            candidate = fit_fleet(inst, candidate, fleet)
            if len(candidate.routes) != fleet:
                since_improve += 1
                continue
        candidate = local_search.run(candidate, dirty=removed)
        if len(candidate.routes) != fleet:
            candidate = fit_fleet(inst, candidate, fleet)
            if len(candidate.routes) != fleet:
                since_improve += 1
                continue

        delta = candidate.cost - current.cost
        if candidate.cost < best.cost - 1e-9:
            best = candidate.copy()
            current = candidate
            time_to_best = time.perf_counter() - start_time
            iter_to_best = iterations
            since_improve = 0
            history.append((time_to_best, best.cost))
            destroy_scores[di] += params.score_best
            repair_scores[ri] += params.score_best
        else:
            since_improve += 1
            if delta < -1e-9:
                current = candidate
                destroy_scores[di] += params.score_better
                repair_scores[ri] += params.score_better
            elif delta <= 0 or rng.random() < math.exp(-delta / temperature):
                current = candidate
                destroy_scores[di] += params.score_accepted
                repair_scores[ri] += params.score_accepted

            if params.restart_stall and since_improve % params.restart_stall == 0:
                current = best.copy()

        if iterations % params.segment == 0:
            lam = params.reaction
            for i in range(n_destroy):
                if destroy_counts[i]:
                    destroy_weights[i] = (1 - lam) * destroy_weights[i] + lam * destroy_scores[i] / destroy_counts[i]
                    destroy_weights[i] = max(destroy_weights[i], 0.05)
            for i in range(n_repair):
                if repair_counts[i]:
                    repair_weights[i] = (1 - lam) * repair_weights[i] + lam * repair_scores[i] / repair_counts[i]
                    repair_weights[i] = max(repair_weights[i], 0.05)
            destroy_scores = [0.0] * n_destroy
            repair_scores = [0.0] * n_repair
            destroy_counts = [0] * n_destroy
            repair_counts = [0] * n_repair

    runtime = time.perf_counter() - start_time
    best.drop_empty()
    return ALNSResult(
        instance=inst.name,
        n_customers=inst.n_customers,
        cost=best.cost,
        routes=best.routes,
        n_routes=len(best.routes),
        fleet_target=fleet_target,
        iterations=iterations,
        runtime=runtime,
        time_to_best=time_to_best,
        iter_to_best=iter_to_best,
        init_cost=init_cost,
        stop_reason=stop_reason,
        history=history,
        destroy_usage=destroy_usage,
        repair_usage=repair_usage,
    )
