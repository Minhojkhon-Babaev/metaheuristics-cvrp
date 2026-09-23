"""Гранулярный локальный поиск для CVRP.

Соседства ограничены K ближайшими клиентами, обход управляется очередью
«грязных» вершин с don't-look bits. Набор операторов: relocate, swap,
2-opt, Or-opt (сегменты длины 2 и 3) и межмаршрутный 2-opt*.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable

from .instance import Instance
from .solution import Solution, route_cost

EPS = 1e-9


class LocalSearch:
    def __init__(
        self,
        inst: Instance,
        neighbors: list[list[int]],
        rng: random.Random,
        fleet_min: int = 1,
    ):
        self.inst = inst
        self.dist = inst.dist
        self.demands = inst.demands
        self.capacity = inst.capacity
        self.depot = inst.depot
        self.neighbors = neighbors
        self.rng = rng
        # Размер парка в наборах CVRPLIB фиксирован: опустошать маршрут можно,
        # только пока непустых маршрутов больше fleet_min.
        self.fleet_min = fleet_min
        self.n_nonempty = 0

        size = inst.dimension
        self.route_of = [-1] * size
        self.pos_of = [-1] * size
        self.routes: list[list[int]] = []
        self.loads: list[int] = []
        self.rcost: list[int] = []
        self.pref: list[list[int]] = []

    # ------------------------------------------------------------------ setup

    def _load(self, sol: Solution) -> None:
        for i in range(self.inst.dimension):
            self.route_of[i] = -1
            self.pos_of[i] = -1
        self.routes = [r[:] for r in sol.routes if r]
        self.n_nonempty = len(self.routes)
        self.loads = []
        self.rcost = []
        self.pref = []
        for idx, route in enumerate(self.routes):
            self._refresh(idx)
            for pos, c in enumerate(route):
                self.route_of[c] = idx
                self.pos_of[c] = pos

    def _refresh(self, idx: int) -> None:
        route = self.routes[idx]
        prefix = [0]
        acc = 0
        for c in route:
            acc += self.demands[c]
            prefix.append(acc)
        while len(self.loads) <= idx:
            self.loads.append(0)
            self.rcost.append(0)
            self.pref.append([0])
        self.loads[idx] = acc
        self.pref[idx] = prefix
        self.rcost[idx] = route_cost(self.inst, route)
        for pos, c in enumerate(route):
            self.route_of[c] = idx
            self.pos_of[c] = pos

    def _export(self) -> Solution:
        routes = [r[:] for r in self.routes if r]
        loads = [self.loads[i] for i, r in enumerate(self.routes) if r]
        cost = sum(self.rcost[i] for i, r in enumerate(self.routes) if r)
        return Solution(routes, loads, cost)

    # -------------------------------------------------------------- accessors

    def _succ(self, u: int) -> int:
        route = self.routes[self.route_of[u]]
        p = self.pos_of[u] + 1
        return route[p] if p < len(route) else self.depot

    def _pred(self, u: int) -> int:
        route = self.routes[self.route_of[u]]
        p = self.pos_of[u] - 1
        return route[p] if p >= 0 else self.depot

    # ------------------------------------------------------------- main cycle

    def run(self, sol: Solution, dirty: Iterable[int] | None = None) -> Solution:
        self._load(sol)

        seeds = list(dirty) if dirty is not None else self.inst.customers
        queue = deque(c for c in seeds if self.route_of[c] >= 0)
        in_queue = [False] * self.inst.dimension
        for c in queue:
            in_queue[c] = True

        while queue:
            u = queue.popleft()
            in_queue[u] = False
            if self.route_of[u] < 0:
                continue
            touched = self._improve_node(u)
            if touched:
                for c in touched:
                    if self.route_of[c] >= 0 and not in_queue[c]:
                        in_queue[c] = True
                        queue.append(c)

        return self._export()

    def _improve_node(self, u: int) -> list[int] | None:
        for move in (self._relocate, self._swap, self._two_opt, self._or_opt, self._two_opt_star):
            touched = move(u)
            if touched:
                return touched
        return None

    # ----------------------------------------------------------------- moves

    def _relocate(self, u: int) -> list[int] | None:
        dist = self.dist
        ru = self.route_of[u]
        pu, su = self._pred(u), self._succ(u)
        gain = dist[pu][u] + dist[u][su] - dist[pu][su]
        if gain <= EPS:
            return None

        if len(self.routes[ru]) == 1 and self.n_nonempty <= self.fleet_min:
            return None

        demand_u = self.demands[u]
        for v in self.neighbors[u]:
            rv = self.route_of[v]
            if rv < 0:
                continue
            if rv != ru and self.loads[rv] + demand_u > self.capacity:
                continue

            for anchor, nxt in ((v, self._succ(v)), (self._pred(v), v)):
                if anchor == u or nxt == u:
                    continue
                add = dist[anchor][u] + dist[u][nxt] - dist[anchor][nxt]
                if add - gain < -EPS:
                    self._apply_relocate(u, rv, anchor)
                    return [u, pu, su, v, anchor, nxt]
        return None

    def _apply_relocate(self, u: int, target_route: int, anchor: int) -> None:
        ru = self.route_of[u]
        self.routes[ru].pop(self.pos_of[u])
        if not self.routes[ru]:
            self.n_nonempty -= 1
        self._refresh(ru)
        route = self.routes[target_route]
        insert_at = self.pos_of[anchor] + 1 if anchor != self.depot else 0
        route.insert(insert_at, u)
        self._refresh(target_route)
        if ru != target_route:
            self._refresh(ru)

    def _swap(self, u: int) -> list[int] | None:
        dist = self.dist
        ru = self.route_of[u]
        pu, su = self._pred(u), self._succ(u)
        du = self.demands[u]

        for v in self.neighbors[u]:
            rv = self.route_of[v]
            if rv < 0 or v == u:
                continue
            dv = self.demands[v]
            if rv != ru:
                if self.loads[ru] - du + dv > self.capacity:
                    continue
                if self.loads[rv] - dv + du > self.capacity:
                    continue

            pv, sv = self._pred(v), self._succ(v)
            if su == v:
                old = dist[pu][u] + dist[u][v] + dist[v][sv]
                new = dist[pu][v] + dist[v][u] + dist[u][sv]
            elif sv == u:
                old = dist[pv][v] + dist[v][u] + dist[u][su]
                new = dist[pv][u] + dist[u][v] + dist[v][su]
            else:
                old = dist[pu][u] + dist[u][su] + dist[pv][v] + dist[v][sv]
                new = dist[pu][v] + dist[v][su] + dist[pv][u] + dist[u][sv]

            if new - old < -EPS:
                iu, iv = self.pos_of[u], self.pos_of[v]
                self.routes[ru][iu], self.routes[rv][iv] = v, u
                self._refresh(ru)
                if rv != ru:
                    self._refresh(rv)
                return [u, v, pu, su, pv, sv]
        return None

    def _two_opt(self, u: int) -> list[int] | None:
        """Внутримаршрутный 2-opt: разворот сегмента между двумя рёбрами."""
        dist = self.dist
        ru = self.route_of[u]
        route = self.routes[ru]
        su = self._succ(u)

        for v in self.neighbors[u]:
            if self.route_of[v] != ru or v == u:
                continue
            i, j = self.pos_of[u], self.pos_of[v]
            if i > j:
                i, j = j, i
            if j <= i + 1:
                continue
            a, b = route[i], route[j]
            sa = route[i + 1]
            sb = route[j + 1] if j + 1 < len(route) else self.depot
            delta = dist[a][b] + dist[sa][sb] - dist[a][sa] - dist[b][sb]
            if delta < -EPS:
                route[i + 1 : j + 1] = route[i + 1 : j + 1][::-1]
                self._refresh(ru)
                return [a, b, sa, sb, u, su]
        return None

    def _or_opt(self, u: int) -> list[int] | None:
        """Перенос сегмента длины 2-3, начинающегося в u, в прямом или обратном порядке."""
        dist = self.dist
        ru = self.route_of[u]
        route = self.routes[ru]
        i = self.pos_of[u]

        for length in (2, 3):
            if i + length > len(route):
                break
            if length == len(route) and self.n_nonempty <= self.fleet_min:
                break
            seg = route[i : i + length]
            seg_set = set(seg)
            head, tail = seg[0], seg[-1]
            p = route[i - 1] if i > 0 else self.depot
            s = route[i + length] if i + length < len(route) else self.depot
            gain = dist[p][head] + dist[tail][s] - dist[p][s]
            if gain <= EPS:
                continue
            seg_demand = sum(self.demands[c] for c in seg)

            for v in self.neighbors[u]:
                rv = self.route_of[v]
                if rv < 0 or v in seg_set:
                    continue
                if rv != ru and self.loads[rv] + seg_demand > self.capacity:
                    continue
                sv = self._succ(v)
                if sv in seg_set:
                    continue
                fwd = dist[v][head] + dist[tail][sv] - dist[v][sv]
                rev = dist[v][tail] + dist[head][sv] - dist[v][sv]
                add, reverse = (fwd, False) if fwd <= rev else (rev, True)
                if add - gain < -EPS:
                    self._apply_segment_move(ru, i, length, rv, v, reverse)
                    return [*seg, p, s, v, sv]
        return None

    def _apply_segment_move(
        self, src: int, start: int, length: int, dst: int, anchor: int, reverse: bool
    ) -> None:
        seg = self.routes[src][start : start + length]
        del self.routes[src][start : start + length]
        if not self.routes[src]:
            self.n_nonempty -= 1
        self._refresh(src)
        if reverse:
            seg = seg[::-1]
        insert_at = self.pos_of[anchor] + 1 if anchor != self.depot else 0
        self.routes[dst][insert_at:insert_at] = seg
        self._refresh(dst)
        if src != dst:
            self._refresh(src)

    def _two_opt_star(self, u: int) -> list[int] | None:
        """Межмаршрутный 2-opt*: обмен хвостами двух маршрутов."""
        dist = self.dist
        ru = self.route_of[u]
        route_u = self.routes[ru]
        i = self.pos_of[u]
        su = self._succ(u)
        tail_u_load = self.loads[ru] - self.pref[ru][i + 1]

        for v in self.neighbors[u]:
            rv = self.route_of[v]
            if rv < 0 or rv == ru:
                continue
            route_v = self.routes[rv]
            j = self.pos_of[v]
            sv = self._succ(v)
            tail_v_load = self.loads[rv] - self.pref[rv][j + 1]

            if self.pref[ru][i + 1] + tail_v_load > self.capacity:
                continue
            if self.pref[rv][j + 1] + tail_u_load > self.capacity:
                continue

            delta = dist[u][sv] + dist[v][su] - dist[u][su] - dist[v][sv]
            if delta < -EPS:
                new_u = route_u[: i + 1] + route_v[j + 1 :]
                new_v = route_v[: j + 1] + route_u[i + 1 :]
                self.routes[ru] = new_u
                self.routes[rv] = new_v
                self._refresh(ru)
                self._refresh(rv)
                return [u, v, su, sv]
        return None
