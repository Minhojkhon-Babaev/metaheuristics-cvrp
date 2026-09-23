"""Сводные таблицы, графики и текстовый отчёт по результатам экспериментов."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SET_COLORS = {"E": "#1b3a6b", "F": "#c8102e", "M": "#0f8a8a", "P": "#d19b28"}
SET_TITLES = {
    "E": "E — Christofides & Eilon",
    "F": "F — Fisher (реальные данные)",
    "M": "M — Christofides et al. (крупные)",
    "P": "P — Augerat (варьируется парк)",
}
SIZE_BUCKETS = [(0, 50, "n ≤ 50"), (51, 100, "51–100"), (101, 150, "101–150"), (151, 10**6, "n > 150")]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(row: dict, key: str) -> float:
    return float(row[key])


def bucket_of(n: int) -> str:
    for low, high, label in SIZE_BUCKETS:
        if low <= n <= high:
            return label
    return "?"


# ------------------------------------------------------------------ агрегация


def per_instance(runs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in runs:
        grouped[row["instance"]].append(row)

    result = []
    for name, rows in grouped.items():
        gaps = [fnum(r, "gap") for r in rows]
        costs = [fnum(r, "cost") for r in rows]
        result.append(
            {
                "instance": name,
                "set": rows[0]["set"],
                "n_customers": int(rows[0]["n_customers"]),
                "optimal": fnum(rows[0], "optimal"),
                "k_reference": int(rows[0]["k_reference"]),
                "seeds": len(rows),
                "best_cost": min(costs),
                "mean_cost": statistics.mean(costs),
                "best_gap": min(gaps),
                "mean_gap": statistics.mean(gaps),
                "worst_gap": max(gaps),
                "std_gap": statistics.pstdev(gaps),
                "init_gap": statistics.mean(fnum(r, "init_gap") for r in rows),
                "mean_runtime": statistics.mean(fnum(r, "runtime") for r in rows),
                "mean_time_to_best": statistics.mean(fnum(r, "time_to_best") for r in rows),
                "mean_iterations": statistics.mean(fnum(r, "iterations") for r in rows),
                "iters_per_sec": statistics.mean(fnum(r, "iterations") / fnum(r, "runtime") for r in rows),
                "n_routes": min(int(r["n_routes"]) for r in rows),
                "solved_to_optimum": int(min(gaps) <= 1e-9),
            }
        )
    result.sort(key=lambda r: (r["set"], r["n_customers"], r["instance"]))
    return result


def group_summary(instances: list[dict], key) -> list[dict]:
    grouped = defaultdict(list)
    for row in instances:
        grouped[key(row)].append(row)

    out = []
    for label, rows in grouped.items():
        out.append(
            {
                "group": label,
                "instances": len(rows),
                "n_min": min(r["n_customers"] for r in rows),
                "n_max": max(r["n_customers"] for r in rows),
                "mean_gap": statistics.mean(r["mean_gap"] for r in rows),
                "best_gap": statistics.mean(r["best_gap"] for r in rows),
                "max_gap": max(r["mean_gap"] for r in rows),
                "init_gap": statistics.mean(r["init_gap"] for r in rows),
                "optima_found": sum(r["solved_to_optimum"] for r in rows),
                "mean_runtime": statistics.mean(r["mean_runtime"] for r in rows),
                "mean_time_to_best": statistics.mean(r["mean_time_to_best"] for r in rows),
                "iters_per_sec": statistics.mean(r["iters_per_sec"] for r in rows),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(title for _, title, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for row in rows:
        cells = []
        for key, _, fmt in columns:
            value = row[key]
            cells.append(format(value, fmt) if fmt else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# -------------------------------------------------------------------- графики


def fig_gap_by_set(instances: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sets = ["E", "F", "M", "P"]
    data = [[r["mean_gap"] for r in instances if r["set"] == s] for s in sets]
    parts = ax.boxplot(data, tick_labels=sets, patch_artist=True, widths=0.55, showmeans=True)
    for patch, s in zip(parts["boxes"], sets):
        patch.set_facecolor(SET_COLORS[s])
        patch.set_alpha(0.35)
    for i, values in enumerate(data, start=1):
        ax.scatter([i] * len(values), values, s=18, color=SET_COLORS[sets[i - 1]], zorder=3)
    ax.set_ylabel("Отклонение от оптимума, %")
    ax.set_xlabel("Набор инстансов")
    ax.set_title("Распределение отклонения по типам задач")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_gap_vs_n(instances: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for s in ["E", "F", "M", "P"]:
        rows = [r for r in instances if r["set"] == s]
        ax.scatter(
            [r["n_customers"] for r in rows],
            [r["mean_gap"] for r in rows],
            label=SET_TITLES[s],
            color=SET_COLORS[s],
            s=42,
            alpha=0.85,
            edgecolor="white",
        )
    ax.set_xlabel("Число клиентов n")
    ax.set_ylabel("Среднее отклонение от оптимума, %")
    ax.set_title("Качество решения в зависимости от размерности")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_time_vs_n(instances: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    for s in ["E", "F", "M", "P"]:
        rows = [r for r in instances if r["set"] == s]
        ax.scatter(
            [r["n_customers"] for r in rows],
            [max(r["mean_time_to_best"], 1e-3) for r in rows],
            color=SET_COLORS[s],
            s=38,
            label=s,
            edgecolor="white",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число клиентов n (лог)")
    ax.set_ylabel("Время до лучшего решения, с (лог)")
    ax.set_title("Скорость: время до рекорда")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    xs = [r["n_customers"] for r in instances]
    ys = [r["iters_per_sec"] for r in instances]
    colors = [SET_COLORS[r["set"]] for r in instances]
    ax.scatter(xs, ys, c=colors, s=38, edgecolor="white")
    if len(xs) > 2:
        lx = [math.log(x) for x in xs]
        ly = [math.log(y) for y in ys]
        slope, intercept = _linfit(lx, ly)
        grid = sorted(xs)
        ax.plot(
            grid,
            [math.exp(intercept + slope * math.log(x)) for x in grid],
            color="black",
            lw=1.2,
            ls="--",
            label=f"степенная аппроксимация: ~n^{slope:.2f}",
        )
        ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число клиентов n (лог)")
    ax.set_ylabel("Итераций ALNS в секунду (лог)")
    ax.set_title("Трудоёмкость одной итерации")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_results_summary(instances: list[dict], by_set: list[dict], out: Path) -> None:
    """Композитный слайдовый график: сводная таблица по наборам + разброс отклонений."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    ax.axis("off")
    header = ["Набор", "Задач", "n", "Старт, %", "Средний gap, %", "Оптимумов", "t до рекорда, с"]
    rows = []
    for row in by_set:
        rows.append(
            [
                row["group"],
                str(row["instances"]),
                f"{row['n_min']}–{row['n_max']}",
                f"{row['init_gap']:.1f}",
                f"{row['mean_gap']:.2f}",
                f"{row['optima_found']}/{row['instances']}",
                f"{row['mean_time_to_best']:.1f}",
            ]
        )
    total_mean = statistics.mean(r["mean_gap"] for r in instances)
    rows.append(
        [
            "Все",
            str(len(instances)),
            f"{min(r['n_customers'] for r in instances)}–{max(r['n_customers'] for r in instances)}",
            f"{statistics.mean(r['init_gap'] for r in instances):.1f}",
            f"{total_mean:.2f}",
            f"{sum(r['solved_to_optimum'] for r in instances)}/{len(instances)}",
            f"{statistics.mean(r['mean_time_to_best'] for r in instances):.1f}",
        ]
    )

    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.75)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#c9d2e0")
        if r == 0:
            cell.set_facecolor("#1b3a6b")
            cell.set_text_props(color="white", weight="bold")
        elif r == len(rows):
            cell.set_facecolor("#e8edf5")
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.set_text_props(weight="bold", color=SET_COLORS.get(rows[r - 1][0], "#1b3a6b"))
    ax.set_title("Отклонение от оптимума по наборам (5 запусков на инстанс)", fontsize=11, pad=18)

    ax = axes[1]
    sets = ["E", "F", "M", "P"]
    data = [[r["mean_gap"] for r in instances if r["set"] == s] for s in sets]
    parts = ax.boxplot(data, tick_labels=sets, patch_artist=True, widths=0.55, showmeans=True)
    for patch, s in zip(parts["boxes"], sets):
        patch.set_facecolor(SET_COLORS[s])
        patch.set_alpha(0.3)
    for i, values in enumerate(data, start=1):
        ax.scatter([i] * len(values), values, s=22, color=SET_COLORS[sets[i - 1]], zorder=3)
    ax.set_ylabel("Отклонение от оптимума, %")
    ax.set_xlabel("Набор")
    ax.set_title("Разброс по инстансам внутри набора", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_scaling(instances: list[dict], out: Path) -> None:
    """Качество и скорость в зависимости от размерности и типа задачи."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    ax = axes[0]
    for s in ["E", "F", "M", "P"]:
        rows = [r for r in instances if r["set"] == s]
        ax.scatter(
            [r["n_customers"] for r in rows],
            [r["mean_gap"] for r in rows],
            color=SET_COLORS[s],
            s=48,
            label=s,
            edgecolor="white",
            zorder=3,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Число клиентов n")
    ax.set_ylabel("Среднее отклонение, %")
    ax.set_title("Качество и размерность", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="набор", title_fontsize=8)

    ax = axes[1]
    for s in ["E", "F", "M", "P"]:
        rows = [r for r in instances if r["set"] == s]
        ax.scatter(
            [r["n_customers"] for r in rows],
            [max(r["mean_time_to_best"], 0.02) for r in rows],
            color=SET_COLORS[s],
            s=48,
            edgecolor="white",
            zorder=3,
        )
    xs = [r["n_customers"] for r in instances]
    ys = [max(r["mean_time_to_best"], 0.02) for r in instances]
    slope, intercept = _linfit([math.log(x) for x in xs], [math.log(y) for y in ys])
    grid = sorted(xs)
    ax.plot(
        grid,
        [math.exp(intercept + slope * math.log(x)) for x in grid],
        color="black",
        ls="--",
        lw=1.2,
        label=f"~n^{slope:.1f}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число клиентов n (лог)")
    ax.set_ylabel("Время до рекорда, с (лог)")
    ax.set_title("Скорость получения решения", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    ys = [r["iters_per_sec"] for r in instances]
    ax.scatter(xs, ys, c=[SET_COLORS[r["set"]] for r in instances], s=48, edgecolor="white", zorder=3)
    slope, intercept = _linfit([math.log(x) for x in xs], [math.log(y) for y in ys])
    ax.plot(
        grid,
        [math.exp(intercept + slope * math.log(x)) for x in grid],
        color="black",
        ls="--",
        lw=1.2,
        label=f"~n^{slope:.2f}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число клиентов n (лог)")
    ax.set_ylabel("Итераций ALNS в секунду (лог)")
    ax.set_title("Трудоёмкость итерации", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0.0
    return slope, my - slope * mx


BASELINE_VALUES = {
    "n_neighbors": 20.0,
    "removal_max_frac": 0.40,
    "sa_start_frac": 0.03,
    "reaction": 0.3,
    "blink": 0.01,
}

TUNING_TITLES = {
    "n_neighbors": "Размер соседства K",
    "removal_max_frac": "Макс. доля удаляемых",
    "sa_start_frac": "Старт. температура",
    "reaction": "Адаптация весов λ",
    "blink": "Вероятность «моргания»",
}


def fig_tuning(path_csv: Path, out: Path) -> dict[str, tuple[object, float, float]]:
    """Парное сравнение с базовой конфигурацией.

    Все конфигурации прогоняются на одних и тех же парах (инстанс, сид), поэтому
    корректно сравнивать не средние по выборкам, а средние разности: это убирает
    из оценки разброс между инстансами, который на порядок больше эффекта
    параметра.
    """
    rows = read_csv(path_csv)
    baseline: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["param"] == "baseline":
            baseline[(row["instance"], row["seed"])] = fnum(row, "gap")

    grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["param"] == "baseline":
            continue
        key = (row["instance"], row["seed"])
        if key in baseline:
            grouped[row["param"]][row["value"]].append((fnum(row, "gap"), baseline[key]))

    base_mean = statistics.mean(baseline.values()) if baseline else 0.0
    params = [p for p in TUNING_TITLES if p in grouped]
    fig, axes = plt.subplots(1, len(params), figsize=(3.1 * len(params), 3.6), squeeze=False)
    best: dict[str, tuple[object, float, float]] = {}

    for ax, param in zip(axes[0], params):
        points = [(BASELINE_VALUES[param], base_mean, 0.0, True)]
        for value, pairs in grouped[param].items():
            gaps = [p[0] for p in pairs]
            deltas = [p[0] - p[1] for p in pairs]
            points.append((float(value), statistics.mean(gaps), statistics.mean(deltas), False))
        points.sort(key=lambda p: p[0])

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", color="#1b3a6b", lw=1.6)
        base_x = BASELINE_VALUES[param]
        ax.scatter([base_x], [base_mean], color="#c8102e", s=85, zorder=6, label="база")

        low = min(points, key=lambda p: p[2])
        ax.scatter([low[0]], [low[1]], color="#d19b28", s=85, zorder=5, label="лучшее")
        best[param] = (low[0], low[1], low[2])

        ax.set_title(TUNING_TITLES[param], fontsize=9)
        ax.set_ylabel("средний gap, %", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    n_pairs = len(baseline)
    fig.suptitle(
        f"Подбор параметров (OFAT, 9 настроечных инстансов, {n_pairs} пар «инстанс–сид», бюджет 10 с)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return best


def fig_budget(path_csv: Path, out: Path) -> list[tuple[float, float, float]]:
    rows = read_csv(path_csv)
    by_time: dict[float, list[float]] = defaultdict(list)
    by_time_runtime: dict[float, list[float]] = defaultdict(list)
    by_stall: dict[float, list[float]] = defaultdict(list)
    by_stall_runtime: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["param"] == "time_limit":
            by_time[float(row["value"])].append(fnum(row, "gap"))
            by_time_runtime[float(row["value"])].append(fnum(row, "runtime"))
        else:
            by_stall[float(row["value"])].append(fnum(row, "gap"))
            by_stall_runtime[float(row["value"])].append(fnum(row, "runtime"))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    xs = sorted(by_time)
    ys = [statistics.mean(by_time[x]) for x in xs]
    axes[0].plot(xs, ys, marker="o", color="#1b3a6b")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Бюджет времени на инстанс, с (лог)")
    axes[0].set_ylabel("Средний gap, %")
    axes[0].set_title("Критерий остановки по времени")
    axes[0].grid(alpha=0.3, which="both")
    for x, y in zip(xs, ys):
        axes[0].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")

    sx = sorted(by_stall)
    sy = [statistics.mean(by_stall[x]) for x in sx]
    st = [statistics.mean(by_stall_runtime[x]) for x in sx]
    axes[1].plot(sx, sy, marker="s", color="#0f8a8a", label="gap, %")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Итераций без улучшения до остановки (лог)")
    axes[1].set_ylabel("Средний gap, %")
    axes[1].set_title("Критерий остановки по стагнации")
    axes[1].grid(alpha=0.3, which="both")
    twin = axes[1].twinx()
    twin.plot(sx, st, marker="^", color="#d19b28", ls="--", label="время, с")
    twin.set_ylabel("Среднее время до остановки, с")
    lines = axes[1].get_lines() + twin.get_lines()
    axes[1].legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return [(x, statistics.mean(by_time[x]), statistics.mean(by_time_runtime[x])) for x in xs]


def fig_convergence(path_json: Path, out: Path) -> None:
    payload = json.loads(path_json.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    palette = ["#1b3a6b", "#c8102e", "#0f8a8a", "#d19b28", "#6b4fa3", "#4f7a28"]
    for color, (name, info) in zip(palette, payload.items()):
        history = info["history"]
        times = [h[0] for h in history]
        gaps = [h[2] for h in history]
        times.append(max(times[-1], 1.0) * 1.05)
        gaps.append(gaps[-1])
        ax.step(times, gaps, where="post", color=color, lw=1.6, label=f"{name} (n={info['n_customers']})")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Отклонение текущего рекорда от оптимума, %")
    ax.set_title("Сходимость ALNS")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_routes(path_json: Path, out: Path, name: str) -> None:
    payload = json.loads(path_json.read_text(encoding="utf-8"))
    if name not in payload:
        name = next(iter(payload))
    info = payload[name]
    coords = info["coords"]
    depot = info["depot"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    cmap = plt.get_cmap("tab20")
    for idx, route in enumerate(info["routes"]):
        path = [depot, *route, depot]
        ax.plot(
            [coords[i][0] for i in path],
            [coords[i][1] for i in path],
            color=cmap(idx % 20),
            lw=1.3,
            marker="o",
            markersize=3.2,
        )
    ax.scatter([coords[depot][0]], [coords[depot][1]], marker="s", s=120, color="#c8102e", zorder=5, label="депо")
    ax.set_title(f"{name}: {len(info['routes'])} маршрутов, отклонение {info['final_gap']:.2f}%")
    ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_operators(path_json: Path, out: Path) -> None:
    payload = json.loads(path_json.read_text(encoding="utf-8"))
    destroy_names, repair_names = [], []
    for info in payload.values():
        destroy_names = list(info["destroy_usage"])
        repair_names = list(info["repair_usage"])
        break

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    labels = list(payload)
    width = 0.8 / max(1, len(labels))
    palette = ["#1b3a6b", "#c8102e", "#0f8a8a", "#d19b28", "#6b4fa3", "#4f7a28"]

    for ax, names, key, title in (
        (axes[0], destroy_names, "destroy_usage", "Операторы разрушения"),
        (axes[1], repair_names, "repair_usage", "Операторы восстановления"),
    ):
        for i, (name, color) in enumerate(zip(labels, palette)):
            usage = payload[name][key]
            total = sum(usage.values()) or 1
            xs = [j + i * width for j in range(len(names))]
            ax.bar(xs, [100 * usage[n] / total for n in names], width=width, color=color, label=name)
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(names))])
        ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_ylabel("Доля вызовов, %")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------- отчёт


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()

    results = Path(args.results)
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    runs = read_csv(results / f"runs_{args.tag}.csv")
    instances = per_instance(runs)
    by_set = group_summary(instances, lambda r: r["set"])
    by_set.sort(key=lambda r: r["group"])
    by_size = group_summary(instances, lambda r: bucket_of(r["n_customers"]))
    by_size.sort(key=lambda r: r["n_min"])

    write_csv(results / "summary_per_instance.csv", instances)
    write_csv(results / "summary_by_set.csv", by_set)
    write_csv(results / "summary_by_size.csv", by_size)

    fig_gap_by_set(instances, figures / "gap_by_set.png")
    fig_gap_vs_n(instances, figures / "gap_vs_n.png")
    fig_time_vs_n(instances, figures / "time_vs_n.png")
    fig_results_summary(instances, by_set, figures / "results_summary.png")
    fig_scaling(instances, figures / "scaling.png")

    tuning_best = {}
    if (results / "tuning_params.csv").exists():
        tuning_best = fig_tuning(results / "tuning_params.csv", figures / "tuning_params.png")
    budget_curve = []
    if (results / "tuning_budget.csv").exists():
        budget_curve = fig_budget(results / "tuning_budget.csv", figures / "tuning_budget.png")
    if (results / "convergence.json").exists():
        fig_convergence(results / "convergence.json", figures / "convergence.png")
        fig_routes(results / "convergence.json", figures / "routes_example.png", "M-n151-k12")
        fig_operators(results / "convergence.json", figures / "operators.png")

    all_gaps = [r["gap"] for r in map(lambda x: {"gap": float(x["gap"])}, runs)]
    mean_gap_runs = statistics.mean(all_gaps)
    mean_gap_inst = statistics.mean(r["mean_gap"] for r in instances)
    best_gap_inst = statistics.mean(r["best_gap"] for r in instances)
    solved = sum(r["solved_to_optimum"] for r in instances)
    total_time = sum(float(r["runtime"]) for r in runs)

    lines = [
        "# Результаты: ALNS для CVRP на наборах E, F, M, P",
        "",
        "## Итоговые показатели",
        "",
        f"- Инстансов: **{len(instances)}** (E — {sum(1 for r in instances if r['set'] == 'E')}, "
        f"F — {sum(1 for r in instances if r['set'] == 'F')}, "
        f"M — {sum(1 for r in instances if r['set'] == 'M')}, "
        f"P — {sum(1 for r in instances if r['set'] == 'P')}), "
        f"независимых запусков на инстанс: **{instances[0]['seeds']}**.",
        f"- **Среднее отклонение от оптимума: {mean_gap_inst:.3f}%** (усреднение сначала по сидам, затем по инстансам).",
        f"- Среднее отклонение по всем {len(runs)} запускам: {mean_gap_runs:.3f}%.",
        f"- Среднее отклонение лучшего из {instances[0]['seeds']} запусков: {best_gap_inst:.3f}%.",
        f"- Оптимум найден точно на **{solved} из {len(instances)}** инстансов.",
        f"- Максимальное среднее отклонение на инстансе: {max(r['mean_gap'] for r in instances):.3f}% "
        f"({max(instances, key=lambda r: r['mean_gap'])['instance']}).",
        f"- Отклонение стартового решения (Кларк — Райт + локальный поиск): "
        f"{statistics.mean(r['init_gap'] for r in instances):.2f}%.",
        f"- Суммарное машинное время эксперимента: {total_time / 60:.0f} мин.",
        "",
        "## Разрез по типу задачи",
        "",
        md_table(
            by_set,
            [
                ("group", "Набор", ""),
                ("instances", "Инстансов", ""),
                ("n_min", "n min", ""),
                ("n_max", "n max", ""),
                ("init_gap", "Старт, %", ".2f"),
                ("mean_gap", "Средний gap, %", ".3f"),
                ("best_gap", "Лучший из сидов, %", ".3f"),
                ("max_gap", "Худший инстанс, %", ".3f"),
                ("optima_found", "Оптимумов", ""),
                ("mean_time_to_best", "Время до рекорда, с", ".1f"),
            ],
        ),
        "",
        "## Разрез по размерности",
        "",
        md_table(
            by_size,
            [
                ("group", "Размер", ""),
                ("instances", "Инстансов", ""),
                ("init_gap", "Старт, %", ".2f"),
                ("mean_gap", "Средний gap, %", ".3f"),
                ("optima_found", "Оптимумов", ""),
                ("mean_runtime", "Бюджет, с", ".1f"),
                ("mean_time_to_best", "Время до рекорда, с", ".1f"),
                ("iters_per_sec", "Итераций/с", ".0f"),
            ],
        ),
        "",
    ]

    if tuning_best:
        lines += [
            "## Подбор параметров (OFAT, парное сравнение с базовой конфигурацией)",
            "",
            md_table(
                [
                    {"param": TUNING_TITLES.get(k, k), "value": v[0], "gap": v[1], "delta": v[2]}
                    for k, v in tuning_best.items()
                ],
                [
                    ("param", "Параметр", ""),
                    ("value", "Лучшее значение", ""),
                    ("gap", "Средний gap, %", ".3f"),
                    ("delta", "Парная разность к базе, п.п.", "+.3f"),
                ],
            ),
            "",
        ]

    if budget_curve:
        lines += [
            "## Критерий остановки: качество против времени",
            "",
            md_table(
                [{"t": t, "gap": g, "rt": rt} for t, g, rt in budget_curve],
                [("t", "Бюджет, с", ".0f"), ("gap", "Средний gap, %", ".3f"), ("rt", "Факт. время, с", ".1f")],
            ),
            "",
        ]

    lines += [
        "## Результаты по инстансам",
        "",
        md_table(
            instances,
            [
                ("instance", "Инстанс", ""),
                ("n_customers", "n", ""),
                ("optimal", "Оптимум", ".0f"),
                ("best_cost", "Лучшее", ".0f"),
                ("mean_cost", "Среднее", ".1f"),
                ("best_gap", "Лучший gap, %", ".3f"),
                ("mean_gap", "Средний gap, %", ".3f"),
                ("std_gap", "СКО gap, %", ".3f"),
                ("n_routes", "Машин", ""),
                ("k_reference", "k эталон", ""),
                ("mean_time_to_best", "t до рекорда, с", ".1f"),
            ],
        ),
        "",
    ]

    (results / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"средний gap по инстансам: {mean_gap_inst:.3f}%; оптимумов: {solved}/{len(instances)}")
    print(f"отчёт: {results / 'REPORT.md'}, графики: {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
