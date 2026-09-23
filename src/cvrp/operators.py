"""Операторы разрушения и восстановления решения для ALNS."""

from __future__ import annotations

import math
import random

from .instance import Instance
from .solution import Solution, make_solution

INF = float("inf")


# --------------------------------------------------------------------- destroy


def random_removal(inst: Instance, sol: Solution, q: int, rng: random.Random) -> list[int]:
    pool = sol.visited()
    return rng.sample(pool, min(q, len(pool)))


def worst_removal(inst: Instance, sol: Solution, q: int, rng: random.Random, p: float = 4.0) -> list[int]:
    """Удаляет клиентов с наибольшим вкладом в стоимость (с рандомизацией ранга)."""
    dist = inst.dist
    depot = inst.depot
    scored: list[tuple[float, int]] = []
    for route in sol.routes:
        for idx, c in enumerate(route):
            prev = route[idx - 1] if idx > 0 else depot
            nxt = route[idx + 1] if idx + 1 < len(route) else depot
            scored.append((dist[prev][c] + dist[c][nxt] - dist[prev][nxt], c))
    scored.sort(key=lambda t: -t[0])

    removed: list[int] = []
    for _ in range(min(q, len(scored))):
        idx = int(len(scored) * (rng.random() ** p))
        removed.append(scored.pop(idx)[1])
    return removed


def shaw_removal(inst: Instance, sol: Solution, q: int, rng: random.Random, p: float = 6.0) -> list[int]:
    """Удаляет «похожих» клиентов — близких географически и по спросу."""
    dist = inst.dist
    demands = inst.demands
    pool = sol.visited()
    if not pool:
        return []

    seed = rng.choice(pool)
    removed = [seed]
    remaining = [c for c in pool if c != seed]
    max_demand = max(demands) or 1

    while remaining and len(removed) < q:
        ref = rng.choice(removed)
        remaining.sort(
            key=lambda c: dist[ref][c] + 0.5 * inst.capacity * abs(demands[ref] - demands[c]) / max_demand
        )
        idx = int(len(remaining) * (rng.random() ** p))
        removed.append(remaining.pop(idx))
    return removed


def string_removal(inst: Instance, sol: Solution, q: int, rng: random.Random, max_string: int = 10) -> list[int]:
    """SISR-подобное удаление строк подряд идущих клиентов из соседних маршрутов.

    Вокруг случайного опорного клиента выбираются маршруты, и из каждого
    вырезается непрерывный кусок — это разрушает целые «куски» географии,
    а не отдельные точки.
    """
    pool = sol.visited()
    if not pool:
        return []

    dist = inst.dist
    seed = rng.choice(pool)
    order = sorted(pool, key=lambda c: dist[seed][c])

    removed: list[int] = []
    touched_routes: set[int] = set()
    route_of = {c: ri for ri, route in enumerate(sol.routes) for c in route}

    for c in order:
        if len(removed) >= q:
            break
        ri = route_of[c]
        if ri in touched_routes:
            continue
        touched_routes.add(ri)
        route = sol.routes[ri]
        length = min(len(route), rng.randint(1, max_string), q - len(removed))
        if length <= 0:
            continue
        start = max(0, min(route.index(c) - rng.randint(0, length - 1), len(route) - length))
        removed.extend(route[start : start + length])
    return removed


def route_removal(inst: Instance, sol: Solution, q: int, rng: random.Random) -> list[int]:
    """Полностью расформировывает маршруты, отдавая предпочтение слабо загруженным.

    Это единственный оператор, способный сократить число машин: на инстансах
    с плотной упаковкой (наборы M и P) оптимум достигается ровно на
    нижней границе bin packing, и «лишний» маршрут иначе не растворяется.
    """
    order = sorted(range(len(sol.routes)), key=lambda ri: sol.loads[ri])
    removed: list[int] = []
    while order and len(removed) < q:
        pick = int(len(order) * (rng.random() ** 2))
        removed.extend(sol.routes[order.pop(pick)])
    return removed


DESTROY_OPERATORS = {
    "random": random_removal,
    "worst": worst_removal,
    "shaw": shaw_removal,
    "string": string_removal,
    "route": route_removal,
}


def apply_destroy(
    inst: Instance, sol: Solution, name: str, q: int, rng: random.Random
) -> tuple[list[list[int]], list[int], list[int]]:
    """Возвращает (маршруты без удалённых, загрузки, список удалённых клиентов).

    Опустевшие маршруты сохраняются в списке как пустые слоты: размер парка
    фиксирован, и восстановление обязано заполнить их обратно.
    """
    removed = DESTROY_OPERATORS[name](inst, sol, q, rng)
    removed_set = set(removed)
    routes = [[c for c in route if c not in removed_set] for route in sol.routes]
    loads = [sum(inst.demands[c] for c in route) for route in routes]
    return routes, loads, list(removed_set)


# ---------------------------------------------------------------------- repair


class Repairer:
    """Вставка клиентов в частичное решение по гранулярному списку позиций."""

    def __init__(self, inst: Instance, neighbors: list[list[int]], rng: random.Random, blink: float = 0.01):
        self.inst = inst
        self.dist = inst.dist
        self.demands = inst.demands
        self.capacity = inst.capacity
        self.depot = inst.depot
        self.neighbors = neighbors
        self.rng = rng
        self.blink = blink
        self.route_of = [-1] * inst.dimension
        self.pos_of = [-1] * inst.dimension
        self.empty: set[int] = set()

    def _index(self, routes: list[list[int]]) -> None:
        for i in range(self.inst.dimension):
            self.route_of[i] = -1
            self.pos_of[i] = -1
        self.empty = {ri for ri, route in enumerate(routes) if not route}
        for ri, route in enumerate(routes):
            for pos, c in enumerate(route):
                self.route_of[c] = ri
                self.pos_of[c] = pos

    def _best_position(
        self, u: int, routes: list[list[int]], loads: list[int], skip_route: int = -1
    ) -> tuple[float, int, int]:
        """Лучшая (стоимость, маршрут, позиция) для клиента u; INF, если мест нет."""
        dist = self.dist
        depot = self.depot
        demand_u = self.demands[u]
        blink = self.blink
        rng_random = self.rng.random

        best = (INF, -1, -1)
        for v in self.neighbors[u]:
            ri = self.route_of[v]
            if ri < 0 or ri == skip_route:
                continue
            if loads[ri] + demand_u > self.capacity:
                continue
            route = routes[ri]
            pos = self.pos_of[v]

            prev = route[pos - 1] if pos > 0 else depot
            nxt = route[pos + 1] if pos + 1 < len(route) else depot
            for anchor, follower, insert_at in ((prev, v, pos), (v, nxt, pos + 1)):
                if blink and rng_random() < blink:
                    continue
                cost = dist[anchor][u] + dist[u][follower] - dist[anchor][follower]
                if cost < best[0]:
                    best = (cost, ri, insert_at)

        for ri in self.empty:
            if ri == skip_route:
                continue
            cost = 2 * dist[depot][u]
            if cost < best[0]:
                best = (cost, ri, 0)

        if best[1] < 0:
            # Гранулярный список ничего не дал — полный перебор как запасной путь.
            # Новые маршруты не открываются: размер парка задан инстансом.
            for ri, route in enumerate(routes):
                if ri == skip_route or loads[ri] + demand_u > self.capacity:
                    continue
                for insert_at in range(len(route) + 1):
                    anchor = route[insert_at - 1] if insert_at > 0 else depot
                    follower = route[insert_at] if insert_at < len(route) else depot
                    cost = dist[anchor][u] + dist[u][follower] - dist[anchor][follower]
                    if cost < best[0]:
                        best = (cost, ri, insert_at)
        return best

    def _commit(self, routes: list[list[int]], loads: list[int], u: int, ri: int, pos: int) -> None:
        routes[ri].insert(pos, u)
        loads[ri] += self.demands[u]
        self.empty.discard(ri)
        for p in range(pos, len(routes[ri])):
            c = routes[ri][p]
            self.route_of[c] = ri
            self.pos_of[c] = p

    def finalize(self, routes: list[list[int]], loads: list[int], fleet: int) -> bool:
        """Доводит число непустых маршрутов ровно до `fleet`.

        Пустые слоты заполняются самым дешёвым отщеплением: выбирается клиент,
        вынос которого в отдельный маршрут увеличивает стоимость меньше всего.
        Лишние пустые слоты удаляются — так парк может сократиться, если
        восстановление уместило всех клиентов в меньшее число маршрутов.
        """
        dist = self.dist
        depot = self.depot

        while len(routes) - len(self.empty) < fleet and self.empty:
            best = (INF, -1, -1)
            for ri, route in enumerate(routes):
                if len(route) < 2:
                    continue
                for pos, c in enumerate(route):
                    prev = route[pos - 1] if pos > 0 else depot
                    nxt = route[pos + 1] if pos + 1 < len(route) else depot
                    delta = 2 * dist[depot][c] - (dist[prev][c] + dist[c][nxt] - dist[prev][nxt])
                    if delta < best[0]:
                        best = (delta, ri, pos)
            if best[1] < 0:
                break
            _, src, pos = best
            customer = routes[src].pop(pos)
            loads[src] -= self.demands[customer]
            slot = self.empty.pop()
            routes[slot].append(customer)
            loads[slot] += self.demands[customer]
            self._index(routes)

        if len(routes) - len(self.empty) != fleet:
            return False
        for ri in sorted(self.empty, reverse=True):
            del routes[ri]
            del loads[ri]
        self.empty.clear()
        return True

    def greedy(
        self, routes: list[list[int]], loads: list[int], removed: list[int], order: str = "random"
    ) -> bool:
        self._index(routes)
        queue = list(removed)
        if order == "random":
            self.rng.shuffle(queue)
        elif order == "demand":
            queue.sort(key=lambda c: -self.demands[c])
        elif order == "far":
            queue.sort(key=lambda c: -self.dist[self.depot][c])

        for u in queue:
            _, ri, pos = self._best_position(u, routes, loads)
            if ri < 0:
                return False
            self._commit(routes, loads, u, ri, pos)
        return True

    def regret(self, routes: list[list[int]], loads: list[int], removed: list[int], k: int = 2) -> bool:
        self._index(routes)
        pending = list(removed)
        while pending:
            best_u, best_score, best_place = None, -INF, None
            for u in pending:
                cost, ri, pos = self._best_position(u, routes, loads)
                if ri < 0:
                    return False
                else:
                    alt_cost, alt_ri, alt_pos = self._best_position(u, routes, loads, skip_route=ri)
                    if k >= 3 and alt_ri >= 0:
                        third, _, _ = self._best_position(u, routes, loads, skip_route=alt_ri)
                        penalty = (alt_cost - cost) + (min(third, INF) - cost if third < INF else 0.0)
                    else:
                        penalty = alt_cost - cost if alt_cost < INF else INF
                    score = penalty
                    place = (ri, pos)
                if score > best_score:
                    best_u, best_score, best_place = u, score, place
            pending.remove(best_u)
            self._commit(routes, loads, best_u, *best_place)
        return True


REPAIR_OPERATORS = ("greedy_random", "greedy_demand", "greedy_far", "regret2")


def apply_repair(
    repairer: Repairer,
    name: str,
    routes: list[list[int]],
    loads: list[int],
    removed: list[int],
    fleet: int,
) -> bool:
    """Вставляет удалённых клиентов и приводит парк к размеру `fleet`.

    Возвращает False, если кто-то из клиентов не помещается ни в один
    существующий маршрут: такую итерацию ALNS просто пропускает.
    """
    if name == "greedy_random":
        ok = repairer.greedy(routes, loads, removed, order="random")
    elif name == "greedy_demand":
        ok = repairer.greedy(routes, loads, removed, order="demand")
    elif name == "greedy_far":
        ok = repairer.greedy(routes, loads, removed, order="far")
    elif name == "regret2":
        ok = repairer.regret(routes, loads, removed, k=2)
    else:
        raise ValueError(f"неизвестный оператор восстановления: {name}")

    if not ok:
        return False
    return repairer.finalize(routes, loads, fleet)


def sa_temperature(start: float, end: float, progress: float) -> float:
    """Геометрическое охлаждение, привязанное к доле израсходованного бюджета."""
    progress = min(max(progress, 0.0), 1.0)
    return start * math.exp(progress * math.log(end / start))
