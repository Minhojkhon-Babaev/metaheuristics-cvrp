"""Представление решения CVRP и проверка его корректности."""

from __future__ import annotations

from dataclasses import dataclass, field

from .instance import Instance


@dataclass
class Solution:
    """Решение — список маршрутов; каждый маршрут содержит только клиентов.

    Депо неявно стоит в начале и в конце каждого маршрута.
    """

    routes: list[list[int]] = field(default_factory=list)
    loads: list[int] = field(default_factory=list)
    cost: float = 0.0

    def copy(self) -> "Solution":
        return Solution([r[:] for r in self.routes], self.loads[:], self.cost)

    def drop_empty(self) -> None:
        keep = [i for i, r in enumerate(self.routes) if r]
        self.routes = [self.routes[i] for i in keep]
        self.loads = [self.loads[i] for i in keep]

    def visited(self) -> list[int]:
        return [c for route in self.routes for c in route]


def route_cost(inst: Instance, route: list[int]) -> int:
    if not route:
        return 0
    dist = inst.dist
    depot = inst.depot
    total = dist[depot][route[0]] + dist[route[-1]][depot]
    for a, b in zip(route, route[1:]):
        total += dist[a][b]
    return total


def total_cost(inst: Instance, routes: list[list[int]]) -> int:
    return sum(route_cost(inst, r) for r in routes)


def make_solution(inst: Instance, routes: list[list[int]]) -> Solution:
    routes = [r[:] for r in routes if r]
    loads = [sum(inst.demands[c] for c in r) for r in routes]
    return Solution(routes, loads, total_cost(inst, routes))


def validate(inst: Instance, sol: Solution) -> None:
    """Бросает ValueError, если решение нарушает ограничения задачи."""
    seen: set[int] = set()
    for route, load in zip(sol.routes, sol.loads):
        if load > inst.capacity:
            raise ValueError(f"перегруз маршрута: {load} > {inst.capacity}")
        if sum(inst.demands[c] for c in route) != load:
            raise ValueError("рассинхронизация кэшированной загрузки маршрута")
        for c in route:
            if c == inst.depot:
                raise ValueError("депо встречается внутри маршрута")
            if c in seen:
                raise ValueError(f"клиент {c} обслужен более одного раза")
            seen.add(c)

    expected = set(inst.customers)
    if seen != expected:
        raise ValueError(f"обслужено {len(seen)} клиентов из {len(expected)}")

    recomputed = total_cost(inst, sol.routes)
    if abs(recomputed - sol.cost) > 1e-6:
        raise ValueError(f"кэшированная стоимость {sol.cost} != пересчитанной {recomputed}")
