#!/usr/bin/env python3
"""Summarize the stage-1 fixed-ef versus DABS recall/NDC experiment."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_summaries(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["row_type"] != "SUMMARY":
                continue
            mode = row["mode"]
            gamma_text = row.get("dabs_gamma", "nan")
            gamma = float(gamma_text) if gamma_text else math.nan
            ef = int(float(row["ef"]))
            parameter = f"gamma={gamma:g}" if mode == "dabs" else f"ef={ef}"
            rows.append(
                {
                    "source": str(path),
                    "method": mode,
                    "parameter": parameter,
                    "ef": ef,
                    "dabs_gamma": gamma,
                    "recall": float(row["recall"]),
                    "ndc": float(row["ndc"]),
                    "latency_us": float(row["latency_us"]),
                    "qps": float(row["qps"]),
                    "hops": float(row["hops"]),
                }
            )
    if not rows:
        raise ValueError(f"No SUMMARY rows found in {path}")
    return rows


def pareto_frontier(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    ordered = sorted(rows, key=lambda row: (float(row["ndc"]), -float(row["recall"])))
    frontier: list[dict[str, float | str]] = []
    best_recall = -math.inf
    for row in ordered:
        recall = float(row["recall"])
        if recall > best_recall:
            frontier.append(row)
            best_recall = recall
    return frontier


def best_at_target(
    rows: list[dict[str, float | str]], method: str, target: float
) -> dict[str, float | str] | None:
    candidates = [
        row for row in rows
        if row["method"] == method and float(row["recall"]) >= target
    ]
    return min(candidates, key=lambda row: float(row["ndc"]), default=None)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dabs", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--targets", default="0.95,0.98,0.99")
    args = parser.parse_args()

    rows = read_summaries(args.baseline) + read_summaries(args.dabs)
    targets = [float(value) for value in args.targets.split(",") if value]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_path = args.output_prefix.with_name(args.output_prefix.name + "_all_points.csv")
    frontier_path = args.output_prefix.with_name(args.output_prefix.name + "_pareto.csv")
    target_path = args.output_prefix.with_name(args.output_prefix.name + "_targets.csv")
    report_path = args.output_prefix.with_name(args.output_prefix.name + "_report.md")

    fields = [
        "method", "parameter", "ef", "dabs_gamma", "recall", "ndc",
        "latency_us", "qps", "hops", "source",
    ]
    write_csv(all_path, sorted(rows, key=lambda row: (float(row["ndc"]), row["method"])), fields)

    frontier = pareto_frontier(rows)
    write_csv(frontier_path, frontier, fields)

    target_rows: list[dict[str, object]] = []
    report_lines = [
        "# Stage 1: fixed-ef 与 DABS",
        "",
        "判定基于相同查询、相同图和实际 Recall–NDC；参数值本身不做横向对应。",
        "",
        "| 目标 Recall | Baseline 最低 NDC | DABS 最低 NDC | DABS NDC 变化 |",
        "|---:|---:|---:|---:|",
    ]
    for target in targets:
        baseline = best_at_target(rows, "baseline", target)
        dabs = best_at_target(rows, "dabs", target)
        saving = math.nan
        if baseline is not None and dabs is not None:
            saving = 100.0 * (
                float(baseline["ndc"]) - float(dabs["ndc"])
            ) / float(baseline["ndc"])

        for method, selected in (("baseline", baseline), ("dabs", dabs)):
            target_rows.append(
                {
                    "target_recall": target,
                    "method": method,
                    "parameter": selected["parameter"] if selected else "",
                    "achieved_recall": selected["recall"] if selected else "",
                    "ndc": selected["ndc"] if selected else "",
                    "latency_us": selected["latency_us"] if selected else "",
                    "dabs_ndc_saving_vs_baseline_pct": saving if method == "dabs" else "",
                }
            )

        b_text = (
            f"{float(baseline['ndc']):.2f} ({baseline['parameter']})"
            if baseline else "未达到"
        )
        d_text = (
            f"{float(dabs['ndc']):.2f} ({dabs['parameter']})"
            if dabs else "未达到"
        )
        s_text = f"节省 {saving:.2f}%" if math.isfinite(saving) else "不可比较"
        report_lines.append(f"| {target:.1%} | {b_text} | {d_text} | {s_text} |")

    write_csv(
        target_path,
        target_rows,
        [
            "target_recall", "method", "parameter", "achieved_recall", "ndc",
            "latency_us", "dabs_ndc_saving_vs_baseline_pct",
        ],
    )

    best_baseline_recall = max(float(row["recall"]) for row in rows if row["method"] == "baseline")
    best_dabs_recall = max(float(row["recall"]) for row in rows if row["method"] == "dabs")
    report_lines += [
        "",
        f"本次参数网格最高 Recall：baseline={best_baseline_recall:.4%}，DABS={best_dabs_recall:.4%}。",
        "",
        "只把 Pareto 前沿上的点带入下一阶段。若 DABS 在目标 Recall 处没有覆盖到，继续增大 gamma；"
        "若 baseline 没覆盖到，则补更大的 ef。首轮 200-query 只用于排错和缩小参数，结论需在未参与调参的查询上复验。",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(report_path)
    print(frontier_path)
    print(target_path)


if __name__ == "__main__":
    main()
