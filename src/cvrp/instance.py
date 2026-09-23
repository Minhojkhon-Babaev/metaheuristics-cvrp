"""Парсер инстансов CVRP в формате VRPLIB (CVRPLIB, наборы E/F/M/P)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

_OPTIMAL_RE = re.compile(r"[Oo]ptimal\s+value\s*:\s*(\d+(?:\.\d+)?)")
_NAME_K_RE = re.compile(r"-k(\d+)\s*$")


@dataclass
class Instance:
    name: str
    dimension: int
    capacity: int
    coords: list[tuple[float, float]]
    demands: list[int]
    depot: int
    dist: list[list[int]]
    optimal: float | None = None
    vehicles: int | None = None
    source_ids: list[int] | None = None

    @property
    def customers(self) -> list[int]:
        return [i for i in range(self.dimension) if i != self.depot]

    @property
    def n_customers(self) -> int:
        return self.dimension - 1

    @property
    def set_name(self) -> str:
        return self.name.split("-")[0]

    def nearest_neighbors(self, k: int) -> list[list[int]]:
        """Для каждой вершины — k ближайших клиентов (депо исключён)."""
        result: list[list[int]] = [[] for _ in range(self.dimension)]
        others = self.customers
        for i in range(self.dimension):
            row = self.dist[i]
            cand = sorted((c for c in others if c != i), key=lambda c: row[c])
            result[i] = cand[:k]
        return result


def _euc_2d(a: tuple[float, float], b: tuple[float, float]) -> int:
    """Округление до ближайшего целого — соглашение TSPLIB EUC_2D."""
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5)


def _explicit_matrix(name: str, dimension: int, fmt: str, weights: list[int]) -> list[list[int]]:
    """Разворачивает EDGE_WEIGHT_SECTION в полную симметричную матрицу."""
    dist = [[0] * dimension for _ in range(dimension)]
    it = iter(weights)

    if fmt == "FULL_MATRIX":
        for i in range(dimension):
            for j in range(dimension):
                dist[i][j] = next(it)
        return dist

    if fmt in ("LOWER_ROW", "LOWER_DIAG_ROW"):
        pairs = (
            ((i, j) for i in range(dimension) for j in range(i))
            if fmt == "LOWER_ROW"
            else ((i, j) for i in range(dimension) for j in range(i + 1))
        )
    elif fmt in ("UPPER_ROW", "UPPER_DIAG_ROW"):
        pairs = (
            ((i, j) for i in range(dimension) for j in range(i + 1, dimension))
            if fmt == "UPPER_ROW"
            else ((i, j) for i in range(dimension) for j in range(i, dimension))
        )
    else:
        raise ValueError(f"{name}: неподдерживаемый EDGE_WEIGHT_FORMAT = {fmt}")

    for i, j in pairs:
        value = next(it)
        dist[i][j] = value
        dist[j][i] = value
    return dist


def read_instance(path: str | Path) -> Instance:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    name = path.stem
    capacity = 0
    dimension = 0
    edge_type = "EUC_2D"
    optimal: float | None = None

    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    depot_ids: list[int] = []

    edge_format = "LOWER_ROW"
    weights: list[int] = []

    section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if ":" in line and not line[0].isdigit() and not line.startswith("-"):
            key, _, value = line.partition(":")
            key, value = key.strip().upper(), value.strip()
            if key == "NAME":
                name = value
            elif key == "DIMENSION":
                dimension = int(value)
            elif key == "CAPACITY":
                capacity = int(float(value))
            elif key == "EDGE_WEIGHT_TYPE":
                edge_type = value.upper()
            elif key == "EDGE_WEIGHT_FORMAT":
                edge_format = value.upper()
            elif key == "COMMENT":
                match = _OPTIMAL_RE.search(value)
                if match:
                    optimal = float(match.group(1))
            continue

        upper = line.upper()
        if upper.startswith("NODE_COORD_SECTION"):
            section = "coords"
            continue
        if upper.startswith("DEMAND_SECTION"):
            section = "demands"
            continue
        if upper.startswith("DEPOT_SECTION"):
            section = "depot"
            continue
        if upper.startswith("EDGE_WEIGHT_SECTION"):
            section = "weights"
            continue
        if upper.startswith("DISPLAY_DATA_SECTION"):
            section = "skip"
            continue
        if upper.startswith("EOF"):
            section = None
            continue

        parts = line.split()
        if section == "coords":
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "demands":
            demands[int(parts[0])] = int(float(parts[1]))
        elif section == "depot":
            node = int(parts[0])
            if node >= 0:
                depot_ids.append(node)
        elif section == "weights":
            weights.extend(int(float(tok)) for tok in parts)

    if edge_type not in ("EUC_2D", "EXPLICIT"):
        raise ValueError(f"{name}: неподдерживаемый EDGE_WEIGHT_TYPE = {edge_type}")

    ids = sorted(set(coords) | set(demands))
    if not ids:
        raise ValueError(f"{name}: не найдены узлы")
    dimension = dimension or len(ids)
    index = {node_id: i for i, node_id in enumerate(ids)}

    coord_list = [coords.get(node_id, (0.0, 0.0)) for node_id in ids]
    demand_list = [demands.get(node_id, 0) for node_id in ids]
    depot = index[depot_ids[0]] if depot_ids else 0
    demand_list[depot] = 0

    if edge_type == "EUC_2D":
        if not coords:
            raise ValueError(f"{name}: не найдена секция NODE_COORD_SECTION")
        dist = [[0] * dimension for _ in range(dimension)]
        for i in range(dimension):
            for j in range(i + 1, dimension):
                d = _euc_2d(coord_list[i], coord_list[j])
                dist[i][j] = d
                dist[j][i] = d
    else:
        dist = _explicit_matrix(name, dimension, edge_format, weights)

    vehicles = None
    match = _NAME_K_RE.search(name)
    if match:
        vehicles = int(match.group(1))

    return Instance(
        name=name,
        dimension=dimension,
        capacity=capacity,
        coords=coord_list,
        demands=demand_list,
        depot=depot,
        dist=dist,
        optimal=optimal,
        vehicles=vehicles,
        source_ids=ids,
    )
