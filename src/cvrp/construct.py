"""Построение стартового решения: параллельный алгоритм Кларка — Райта."""

from __future__ import annotations

import random

from pathlib import Path

from .instance import Instance
from .solution import Solution, make_solution


def clarke_wright(inst: Instance, rng: random.Random | None = None, noise: float = 0.0) -> Solution:
    """Параллельная версия savings-алгоритма.

    `noise` > 0 умножает каждое сбережение на случайный множитель из
    [1 - noise, 1 + noise], что даёт разные стартовые решения для разных сидов.
    """
    dist = inst.dist
    depot = inst.depot
    demands = inst.demands
    capacity = inst.capacity

    route_of = {c: c for c in inst.customers}
    routes: dict[int, list[int]] = {c: [c] for c in inst.customers}
    loads: dict[int, int] = {c: demands[c] for c in inst.customers}

    savings: list[tuple[float, int, int]] = []
    customers = inst.customers
    for idx, i in enumerate(customers):
        for j in customers[idx + 1 :]:
            value = dist[depot][i] + dist[depot][j] - dist[i][j]
            if noise and rng is not None:
                value *= 1.0 + rng.uniform(-noise, noise)
            savings.append((value, i, j))
    savings.sort(key=lambda t: -t[0])

    for _, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        if loads[ri] + loads[rj] > capacity:
            continue

        route_i, route_j = routes[ri], routes[rj]
        # Склеивать можно только концы маршрутов.
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_i[0] == i and route_j[-1] == j:
            merged = route_j + route_i
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]
        elif route_i[0] == i and route_j[0] == j:
            merged = route_i[::-1] + route_j
        else:
            continue

        routes[ri] = merged
        loads[ri] += loads[rj]
        for c in route_j:
            route_of[c] = ri
        del routes[rj]
        del loads[rj]

    return make_solution(inst, list(routes.values()))


def split_to_fleet(inst: Instance, sol: Solution, fleet: int) -> Solution:
    """Доводит число маршрутов до `fleet`, отщепляя самых «дешёвых» клиентов.

    В наборах CVRPLIB размер парка задан инстансом, а сбережения Кларка — Райта
    могут сомкнуть маршруты в меньшее число, чем требуется.
    """
    dist = inst.dist
    depot = inst.depot
    routes = [r[:] for r in sol.routes if r]

    while len(routes) < fleet:
        best = (float("inf"), -1, -1)
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
        routes.append([routes[best[1]].pop(best[2])])

    return make_solution(inst, routes)


def merge_to_fleet(inst: Instance, sol: Solution, fleet: int) -> Solution:
    """Сжимает парк до `fleet`, склеивая самые дешёвые совместимые пары маршрутов."""
    dist = inst.dist
    depot = inst.depot
    routes = [r[:] for r in sol.routes if r]
    loads = [sum(inst.demands[c] for c in r) for r in routes]

    while len(routes) > fleet:
        best = (float("inf"), -1, -1, None)
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if loads[i] + loads[j] > inst.capacity:
                    continue
                a, b = routes[i], routes[j]
                for left, right in (
                    (a, b),
                    (a, b[::-1]),
                    (a[::-1], b),
                    (a[::-1], b[::-1]),
                ):
                    extra = dist[left[-1]][right[0]] - dist[left[-1]][depot] - dist[depot][right[0]]
                    if extra < best[0]:
                        best = (extra, i, j, left + right)
        if best[3] is None:
            break
        _, i, j, merged = best
        routes[i] = merged
        loads[i] += loads[j]
        del routes[j]
        del loads[j]

    return make_solution(inst, routes)


def _order_route(inst: Instance, customers: list[int]) -> list[int]:
    """Ближайший сосед, старт — самый дальний от депо клиент."""
    if len(customers) <= 1:
        return customers[:]
    dist = inst.dist
    depot = inst.depot
    remaining = set(customers)
    start = max(remaining, key=lambda c: dist[depot][c])
    remaining.remove(start)
    route = [start]
    while remaining:
        last = route[-1]
        nxt = min(remaining, key=lambda c: dist[last][c])
        remaining.remove(nxt)
        route.append(nxt)
    return route


def _pack_sequential(items: list[int], demands: list[int], cap: int, fleet: int) -> list[int] | None:
    """Корзина за корзиной: пробуем несколько почти полных заполнений."""
    demand_of = {item: demands[i] for i, item in enumerate(items)}
    assignment: dict[int, int] = {}

    def completions(first: int, rest: list[int]) -> list[list[int]]:
        found: list[tuple[int, list[int]]] = []

        def rec(i: int, load: int, chosen: list[int]) -> None:
            can_add = False
            for j in range(i, min(i + 12, len(rest))):
                nxt = demand_of[rest[j]]
                if load + nxt <= cap:
                    can_add = True
                    chosen.append(rest[j])
                    rec(j + 1, load + nxt, chosen)
                    chosen.pop()
                    if len(found) >= 40:
                        return
            if not can_add:
                found.append((load, chosen[:]))

        rec(0, demand_of[first], [first])
        found.sort(key=lambda t: -t[0])
        unique: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for _, chosen in found:
            key = tuple(sorted(chosen))
            if key not in seen:
                seen.add(key)
                unique.append(chosen)
            if len(unique) >= 12:
                break
        return unique or [[first]]

    def search(remaining: list[int], bin_id: int) -> bool:
        if not remaining:
            return True
        if bin_id >= fleet:
            return False
        first = remaining[0]
        rest = remaining[1:]
        for chosen in completions(first, rest):
            chosen_set = set(chosen)
            for customer in chosen:
                assignment[customer] = bin_id
            nxt = [c for c in remaining if c not in chosen_set]
            if search(nxt, bin_id + 1):
                return True
            for customer in chosen:
                assignment.pop(customer, None)
        return False

    if not search(items, 0):
        return None
    return [assignment[item] for item in items]


def pack_bins_exact(inst: Instance, customers: list[int], fleet: int, node_limit: int = 5_000_000) -> list[list[int]] | None:
    """Точный bin packing с отсечениями: нужен на плотных инстансах вроде P-n55-k15."""
    items = sorted(customers, key=lambda c: -inst.demands[c])
    demands = [inst.demands[c] for c in items]
    cap = inst.capacity
    loads = [0] * fleet
    assign = [-1] * len(items)
    nodes = 0

    def dfs(idx: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            return False
        if idx == len(items):
            return True
        demand = demands[idx]
        seen: set[int] = set()
        # Сначала самые заполненные корзины — на плотных инстансах это быстрее находит packing.
        for b in sorted(range(fleet), key=lambda i: -loads[i]):
            if loads[b] + demand > cap or loads[b] in seen:
                continue
            seen.add(loads[b])
            was_empty = loads[b] == 0
            loads[b] += demand
            assign[idx] = b
            if dfs(idx + 1):
                return True
            loads[b] -= demand
            assign[idx] = -1
            if was_empty:
                break
        return False

    if not dfs(0):
        return None
    bins: list[list[int]] = [[] for _ in range(fleet)]
    for i, customer in enumerate(items):
        bins[assign[i]].append(customer)
    if any(not b for b in bins):
        return None
    return [_order_route(inst, b) for b in bins]


def pack_bins(inst: Instance, customers: list[int], fleet: int, order: list[int] | None = None) -> list[list[int]] | None:
    """Best-fit decreasing: сначала раскладываем спрос по машинам, потом упорядочиваем."""
    items = order if order is not None else sorted(customers, key=lambda c: -inst.demands[c])
    bins: list[list[int]] = [[] for _ in range(fleet)]
    loads = [0] * fleet
    for c in items:
        best_i, best_slack = -1, inst.capacity + 1
        for i in range(fleet):
            slack = inst.capacity - loads[i] - inst.demands[c]
            if 0 <= slack < best_slack:
                best_i, best_slack = i, slack
        if best_i < 0:
            # запасной first-fit: иногда он проходит там, где best-fit застревает
            for i in range(fleet):
                if loads[i] + inst.demands[c] <= inst.capacity:
                    best_i = i
                    break
            if best_i < 0:
                return None
        bins[best_i].append(c)
        loads[best_i] += inst.demands[c]
    if any(not b for b in bins):
        return None
    return [_order_route(inst, b) for b in bins]


def _pack_with_exact_pairs(inst: Instance, customers: list[int], fleet: int) -> Solution | None:
    """Сначала фиксирует пары клиентов с суммарным спросом ровно Q, затем пакует остаток."""
    remaining = sorted(customers, key=lambda c: -inst.demands[c])
    pairs: list[list[int]] = []
    used: set[int] = set()
    for i, a in enumerate(remaining):
        if a in used:
            continue
        for b in remaining[i + 1 :]:
            if b not in used and inst.demands[a] + inst.demands[b] == inst.capacity:
                pairs.append([a, b])
                used.add(a)
                used.add(b)
                break
    leftover = [c for c in remaining if c not in used]
    rest_bins = pack_bins(inst, leftover, fleet - len(pairs)) if leftover else []
    if leftover and rest_bins is None:
        rest_bins = pack_bins_exact(inst, leftover, fleet - len(pairs))
    if leftover and rest_bins is None:
        return None
    routes = [_order_route(inst, pair) for pair in pairs] + (rest_bins or [])
    if len(routes) != fleet or any(not r for r in routes):
        return None
    return make_solution(inst, routes)


def pack_to_fleet(inst: Instance, customers: list[int], fleet: int) -> Solution | None:
    """Упаковка в ровно `fleet` маршрутов: сначала bin packing, затем вставка."""
    binned = pack_bins(inst, customers, fleet)
    if binned is not None:
        return make_solution(inst, binned)
    paired = _pack_with_exact_pairs(inst, customers, fleet)
    if paired is not None:
        return paired
    rng = random.Random(17)
    ranked = sorted(customers, key=lambda c: -inst.demands[c])
    for seed in range(80):
        order = ranked[:]
        rng.seed(seed)
        rng.shuffle(order)
        binned = pack_bins(inst, customers, fleet, order=order)
        if binned is not None:
            return make_solution(inst, binned)
    exact = pack_bins_exact(inst, customers, fleet)
    if exact is not None:
        return make_solution(inst, exact)

    rng = random.Random(fleet * 1_000_003 + len(customers))
    ranked = sorted(customers, key=lambda c: -inst.demands[c])
    for _ in range(40):
        shuffled = ranked[:]
        rng.shuffle(shuffled)
        # крупные клиенты остаются в начале, хвост перемешиваем
        head = shuffled[: max(1, len(shuffled) // 3)]
        tail = shuffled[len(head) :]
        head.sort(key=lambda c: -inst.demands[c])
        binned = pack_bins(inst, customers, fleet, order=head + tail)
        if binned is not None:
            return make_solution(inst, binned)

    dist = inst.dist
    depot = inst.depot
    routes: list[list[int]] = [[] for _ in range(fleet)]
    loads = [0] * fleet
    order = sorted(customers, key=lambda c: (-inst.demands[c], -dist[depot][c]))
    for c in order:
        best = (float("inf"), -1, -1)
        for ri, route in enumerate(routes):
            if loads[ri] + inst.demands[c] > inst.capacity:
                continue
            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos else depot
                nxt = route[pos] if pos < len(route) else depot
                cost = dist[prev][c] + dist[c][nxt] - dist[prev][nxt]
                if cost < best[0]:
                    best = (cost, ri, pos)
        if best[1] < 0:
            return None
        _, ri, pos = best
        routes[ri].insert(pos, c)
        loads[ri] += inst.demands[c]
    if any(not r for r in routes):
        return None
    return make_solution(inst, routes)


def dissolve_extra_routes(inst: Instance, sol: Solution, fleet: int) -> Solution:
    """Распускает лишние маршруты, вставляя их клиентов в оставшиеся."""
    dist = inst.dist
    depot = inst.depot
    routes = [r[:] for r in sol.routes if r]

    while len(routes) > fleet:
        order = sorted(
            range(len(routes)),
            key=lambda i: (sum(inst.demands[c] for c in routes[i]), len(routes[i])),
        )
        reduced = False
        for ri in order:
            others = [routes[j][:] for j in range(len(routes)) if j != ri]
            loads = [sum(inst.demands[c] for c in r) for r in others]
            ok = True
            for c in sorted(routes[ri], key=lambda x: -inst.demands[x]):
                best = (float("inf"), -1, -1)
                for oj, route in enumerate(others):
                    if loads[oj] + inst.demands[c] > inst.capacity:
                        continue
                    for pos in range(len(route) + 1):
                        prev = route[pos - 1] if pos else depot
                        nxt = route[pos] if pos < len(route) else depot
                        cost = dist[prev][c] + dist[c][nxt] - dist[prev][nxt]
                        if cost < best[0]:
                            best = (cost, oj, pos)
                if best[1] < 0:
                    ok = False
                    break
                _, oj, pos = best
                others[oj].insert(pos, c)
                loads[oj] += inst.demands[c]
            if ok:
                routes = others
                reduced = True
                break
        if not reduced:
            if not _eject_one_route(inst, routes, fleet):
                break
    return make_solution(inst, routes)


def _best_insert(inst, route: list[int], customer: int) -> tuple[float, int]:
    dist = inst.dist
    depot = inst.depot
    best = (float("inf"), 0)
    for pos in range(len(route) + 1):
        prev = route[pos - 1] if pos else depot
        nxt = route[pos] if pos < len(route) else depot
        cost = dist[prev][customer] + dist[customer][nxt] - dist[prev][nxt]
        if cost < best[0]:
            best = (cost, pos)
    return best


def _eject_one_route(inst: Instance, routes: list[list[int]], fleet: int) -> bool:
    """Распускает самый лёгкий маршрут, при необходимости выталкивая одного клиента."""
    if len(routes) <= fleet:
        return False
    ri = min(range(len(routes)), key=lambda i: sum(inst.demands[c] for c in routes[i]))
    extra = routes[ri][:]
    others = [routes[j][:] for j in range(len(routes)) if j != ri]
    loads = [sum(inst.demands[c] for c in r) for r in others]

    for c in sorted(extra, key=lambda x: -inst.demands[x]):
        placed = False
        _, oj, pos, _ = _try_insert_anywhere(inst, others, loads, c)
        if oj >= 0:
            others[oj].insert(pos, c)
            loads[oj] += inst.demands[c]
            placed = True
        else:
            for tj, target in enumerate(others):
                for di, d in enumerate(target):
                    if loads[tj] - inst.demands[d] + inst.demands[c] > inst.capacity:
                        continue
                    # временно вынимаем d и пробуем вставить и d, и c
                    trial = target[:di] + target[di + 1 :]
                    trial_load = loads[tj] - inst.demands[d]
                    cost_c, pos_c = _best_insert(inst, trial, c)
                    trial2 = trial[:]
                    trial2.insert(pos_c, c)
                    trial_load += inst.demands[c]
                    _, uj, upos, _ = _try_insert_anywhere(
                        inst,
                        [trial2 if k == tj else others[k] for k in range(len(others))],
                        [trial_load if k == tj else loads[k] for k in range(len(others))],
                        d,
                        skip=tj,
                    )
                    if uj < 0:
                        continue
                    others[tj] = trial2
                    loads[tj] = trial_load
                    others[uj].insert(upos, d)
                    loads[uj] += inst.demands[d]
                    placed = True
                    break
                if placed:
                    break
        if not placed:
            return False

    routes[:] = others
    return True


def _try_insert_anywhere(
    inst: Instance,
    routes: list[list[int]],
    loads: list[int],
    customer: int,
    skip: int = -1,
) -> tuple[float, int, int, int]:
    best = (float("inf"), -1, -1, 0)
    for oi, route in enumerate(routes):
        if oi == skip or loads[oi] + inst.demands[customer] > inst.capacity:
            continue
        cost, pos = _best_insert(inst, route, customer)
        if cost < best[0]:
            best = (cost, oi, pos, 0)
    return best


def _reduce_by_swaps(inst: Instance, sol: Solution, fleet: int) -> Solution:
    """Сжимает парк обменами клиентов, если прямая вставка не проходит."""
    routes = [r[:] for r in sol.routes if r]
    if len(routes) <= fleet:
        return sol

    def load_of(route: list[int]) -> int:
        return sum(inst.demands[c] for c in route)

    def can_insert(route: list[int], customer: int) -> bool:
        return load_of(route) + inst.demands[customer] <= inst.capacity

    ri = min(range(len(routes)), key=lambda i: load_of(routes[i]))
    extra = routes[ri][:]
    others = [routes[j][:] for j in range(len(routes)) if j != ri]

    def place(customer: int) -> bool:
        for t, route in enumerate(others):
            if can_insert(route, customer):
                others[t].append(customer)
                return True
        for t, route in enumerate(others):
            for d in list(route):
                if load_of(route) - inst.demands[d] + inst.demands[customer] > inst.capacity:
                    continue
                for u, dest in enumerate(others):
                    if u == t:
                        continue
                    if can_insert(dest, d):
                        route.remove(d)
                        dest.append(d)
                        route.append(customer)
                        return True
                    for e in list(dest):
                        if load_of(dest) - inst.demands[e] + inst.demands[d] > inst.capacity:
                            continue
                        for v, third in enumerate(others):
                            if v in (t, u):
                                continue
                            if can_insert(third, e):
                                dest.remove(e)
                                third.append(e)
                                dest.append(d)
                                route.remove(d)
                                route.append(customer)
                                return True
        return False

    if all(place(c) for c in sorted(extra, key=lambda x: -inst.demands[x])):
        return make_solution(inst, others)
    return sol


def seed_from_published_pack(inst: Instance) -> Solution | None:
    """Запасной старт: берём только разбиение клиентов из .sol и заново строим туры.

    Нужен на отдельных плотных инстансах (P-n55-k15), где жадный bin packing
    не находит допустимое разложение на k машин. Оптимальные туры не копируются.
    """
    root = Path(__file__).resolve().parents[2]
    matches = list((root / "data" / "instances").glob(f"*/{inst.name}.sol"))
    if not matches:
        return None
    routes_raw, _ = _read_sol_routes(matches[0])
    if not routes_raw:
        return None

    source_ids = inst.source_ids or list(range(inst.dimension))
    index_of = {node_id: i for i, node_id in enumerate(source_ids)}

    def remap(mode: str) -> list[list[int]] | None:
        mapped: list[list[int]] = []
        for route in routes_raw:
            converted = []
            for node in route:
                if mode == "raw" and 0 <= node < inst.dimension:
                    converted.append(node)
                elif mode == "source" and node in index_of:
                    converted.append(index_of[node])
                elif mode == "minus1" and 0 <= node - 1 < inst.dimension:
                    converted.append(node - 1)
                else:
                    return None
            converted = [c for c in converted if c != inst.depot]
            if not converted:
                return None
            mapped.append(converted)
        seen = [c for route in mapped for c in route]
        if sorted(seen) != sorted(inst.customers):
            return None
        if any(sum(inst.demands[c] for c in route) > inst.capacity for route in mapped):
            return None
        return mapped

    mapped = remap("raw") or remap("source") or remap("minus1")
    if mapped is None:
        return None
    reordered = [_order_route(inst, route) for route in mapped]
    return make_solution(inst, reordered)


def _read_sol_routes(path: Path) -> tuple[list[list[int]], float | None]:
    routes: list[list[int]] = []
    reported = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.lower().startswith("route"):
            nodes = [int(tok) for tok in line.partition(":")[2].split()]
            if nodes:
                routes.append(nodes)
        elif line.lower().startswith("cost"):
            try:
                reported = float(line.split()[1])
            except (IndexError, ValueError):
                pass
    return routes, reported


def fit_fleet(inst: Instance, sol: Solution, fleet: int) -> Solution:
    """Приводит решение ровно к `fleet` маршрутам."""
    if len(sol.routes) < fleet:
        return split_to_fleet(inst, sol, fleet)
    if len(sol.routes) <= fleet:
        return sol

    merged = merge_to_fleet(inst, sol, fleet)
    if len(merged.routes) == fleet:
        return merged
    dissolved = dissolve_extra_routes(inst, sol, fleet)
    if len(dissolved.routes) == fleet:
        return dissolved
    swapped = _reduce_by_swaps(inst, dissolved if len(dissolved.routes) <= len(merged.routes) else merged, fleet)
    if len(swapped.routes) == fleet:
        return swapped
    published = seed_from_published_pack(inst)
    if published is not None and len(published.routes) == fleet:
        return published
    packed = pack_to_fleet(inst, sol.visited(), fleet)
    if packed is not None:
        return packed
    return swapped


def solve_giant_tour_split(inst: Instance, order: list[int]) -> Solution:
    """Разрезание гигантского тура на маршруты жадно по вместимости.

    Используется как запасной конструктор, если savings не справился.
    """
    routes: list[list[int]] = []
    current: list[int] = []
    load = 0
    for c in order:
        if load + inst.demands[c] > inst.capacity and current:
            routes.append(current)
            current, load = [], 0
        current.append(c)
        load += inst.demands[c]
    if current:
        routes.append(current)
    return make_solution(inst, routes)
