#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import chi2


matplotlib.use("Agg")
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tmp"
PREFIX_PERFORMANCE_CLICKS = "test_clicks_param_shift_"
DEFAULT_FOLDER_PATH = (
    "results/Baidu_ULTR_position/"
    "baidu_subset=train_Baidu_ULTRA_part1.npz,data=Custom_dataset_deep,"
    "experiment=Baidu_ULTR_position,logging_policy_ranker=ordered,"
    "logging_policy_sampler=e_greedy,policy_temperature=0.0,"
    "relevance=deep,relevance_tower=deeper"
)
DEFAULT_DATA_PATH = "../ltr_datasets/train_Baidu_ULTRA_part1.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RQ3 Baidu identifiability analysis from the terminal "
            "using a results folder and a parsed Baidu dataset file."
        )
    )
    parser.add_argument(
        "--folder-path",
        type=str,
        help="Path to the results folder that contains the param-shift CSV files.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the parsed Baidu .npz dataset file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Optional path for the saved identifiability plot.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        help="Optional path for a CSV export of the per-parameter summary.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.001, 0.0001],
        help="Chi-square significance thresholds used for identifiability.",
    )
    return parser.parse_args()


def prompt_with_default(value: str | None, prompt_text: str, default_value: str) -> str:
    if value:
        return value
    prompted = input(f"{prompt_text} [{default_value}]: ").strip()
    return prompted or default_value


def resolve_existing_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Results folder does not exist: {path}")
    return path


def resolve_data_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    candidates = [path]
    if path.suffix == "":
        candidates.append(path.with_suffix(".npz"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    tried = ", ".join(str(candidate.expanduser().resolve()) for candidate in candidates)
    raise FileNotFoundError(f"Dataset file not found. Tried: {tried}")


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return Path(os.path.abspath(path))


def build_default_output_paths(data_path: Path) -> tuple[str, str]:
    file_stem = data_path.stem
    plot_path = DEFAULT_OUTPUT_DIR / f"identifiability_{file_stem}.pdf"
    summary_path = DEFAULT_OUTPUT_DIR / f"identifiability_{file_stem}.csv"
    return str(plot_path.relative_to(REPO_ROOT)), str(summary_path.relative_to(REPO_ROOT))


def load_and_concat_multi_shift_files(
    folder_path: Path,
    prefix: str,
    column_name_relevance: str,
    column_name_idx: str,
) -> pd.DataFrame:
    shift_files = sorted(folder_path.glob(f"{prefix}*.csv"))
    if not shift_files:
        raise FileNotFoundError(
            f"No files starting with '{prefix}' were found in {folder_path}"
        )

    def sort_key(path: Path) -> tuple[int, float]:
        name = path.name
        relevance_shift = float(name.split("_")[-2].replace(".csv", ""))
        param_idx = int(name.split("idx")[-1].replace(".csv", ""))
        return param_idx, relevance_shift

    frames = []
    for path in sorted(shift_files, key=sort_key):
        df = pd.read_csv(path)
        df[column_name_relevance] = float(path.name.split("_")[-2].replace(".csv", ""))
        df[column_name_idx] = int(path.name.split("idx")[-1].replace(".csv", ""))
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_perf_df(folder_path: Path, data_path: Path) -> pd.DataFrame:
    perf_df = load_and_concat_multi_shift_files(
        folder_path,
        PREFIX_PERFORMANCE_CLICKS,
        "relevance_shift",
        "param_idx",
    )

    data = np.load(data_path, allow_pickle=True)
    if "padded_positions" not in data:
        raise KeyError(f"'padded_positions' not found in dataset file: {data_path}")

    padded_positions = data["padded_positions"]
    counts = np.sum(padded_positions != -1, axis=0)
    perf_df["sample_count"] = perf_df["param_idx"].map(
        {idx: int(count) for idx, count in enumerate(counts)}
    )

    baseline_loss = perf_df.loc[perf_df["relevance_shift"] == 0, "loss"].mean()
    if pd.isna(baseline_loss):
        raise ValueError("Could not find a baseline row where relevance_shift == 0.")

    perf_df["delta_loss"] = perf_df["loss"] - baseline_loss
    perf_df["delta_loss_times_samples"] = perf_df["delta_loss"] * perf_df["sample_count"]
    perf_df = perf_df.dropna(
        subset=["relevance_shift", "param_idx", "loss", "sample_count", "delta_loss"]
    ).copy()

    return perf_df


def compute_identifiability(
    df: pd.DataFrame, thresholds: list[float]
) -> tuple[dict[int, tuple], bool]:
    required = {
        "relevance_shift",
        "param_idx",
        "loss",
        "sample_count",
        "delta_loss",
        "delta_loss_times_samples",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    model_threshold = 0.05 / df["param_idx"].nunique()
    full_identifiability = True
    chi2_critical = {thr: chi2.ppf(1 - thr, df=1) for thr in thresholds}

    working_df = df.copy()
    working_df["min_pval"] = np.inf
    for thr, crit in chi2_critical.items():
        mask = working_df["delta_loss_times_samples"] >= crit
        working_df.loc[mask, "min_pval"] = np.minimum(
            working_df.loc[mask, "min_pval"],
            thr,
        )

    def first_identifiable(sub_df: pd.DataFrame, sign: int) -> tuple[float | None, float | None]:
        signed = sub_df[sub_df["relevance_shift"] * sign > 0]
        if signed.empty:
            return None, None
        signed = signed.sort_values("relevance_shift", ascending=(sign > 0))
        signed = signed[signed["min_pval"] < np.inf]
        if signed.empty:
            return None, None
        row = signed.iloc[0]
        return float(row["min_pval"]), float(row["relevance_shift"])

    results: dict[int, tuple] = {}

    print(
        f"{'Param':>5} | {'Neg p':>7} | {'Neg d':>7} | {'Pos d':>7} | "
        f"{'Max dLoss':>10} | {'Samples':>9} | {'Conclusion':>22}"
    )
    print("-" * 97)

    for param, sub_df in working_df.groupby("param_idx"):
        neg_thr, neg_shift = first_identifiable(sub_df, sign=-1)
        pos_thr, pos_shift = first_identifiable(sub_df, sign=1)

        if neg_thr is None or pos_thr is None:
            full_identifiability = False
        elif min(neg_thr, pos_thr) > model_threshold:
            full_identifiability = False

        max_delta = float(sub_df["delta_loss"].max())
        total_samples = int(sub_df["sample_count"].sum())

        if neg_thr is None and pos_thr is None:
            conclusion = "unidentified"
        elif neg_thr is not None and pos_thr is not None:
            conclusion = "identified"
        else:
            conclusion = "practically unidentified"

        results[int(param)] = (
            (neg_thr, pos_thr),
            (neg_shift, pos_shift),
            max_delta,
            total_samples,
            conclusion,
        )

        print(
            f"{int(param):>5} | {str(neg_thr):>7} | {str(neg_shift):>7} | "
            f"{str(pos_shift):>7} | {max_delta:>10.6f} | "
            f"{total_samples:>9} | {conclusion:>22}"
        )

    if full_identifiability:
        print(
            "Final Model conclusion: The model is identified with a "
            "bonferroni-corrected p-value of 0.05."
        )
    else:
        print("Final Model conclusion: The model is not found to be identified.")

    return results, full_identifiability


def identifiability_results_to_df(results: dict[int, tuple]) -> pd.DataFrame:
    rows = []
    for param_idx, values in sorted(results.items()):
        (neg_p, pos_p), (neg_shift, pos_shift), max_delta_loss, total_samples, conclusion = values
        rows.append(
            {
                "param_idx": param_idx,
                "neg_p_value": neg_p,
                "pos_p_value": pos_p,
                "neg_shift": neg_shift,
                "pos_shift": pos_shift,
                "max_delta_loss": max_delta_loss,
                "total_samples": total_samples,
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)


def plot_identifiability_summary(
    identifiability_results: dict[int, tuple],
    perf_df: pd.DataFrame,
    output_path: Path,
) -> None:
    color_map = {
        "identified": "green",
        "practically unidentified": "#FF6200",
        "unidentified": "gray",
    }
    sample_count_color = "#FF0000"
    comparison_lines = ((3.84, r"$\alpha$=0.05"), (10.83, r"$\alpha$=0.001"))

    params = sorted(identifiability_results.keys())
    sample_counts = [identifiability_results[p][3] for p in params]
    conclusions = [identifiability_results[p][4].strip().lower() for p in params]
    dot_colors = [color_map[conclusion] for conclusion in conclusions]

    neg_df = perf_df[perf_df["relevance_shift"] < 0]
    pos_df = perf_df[perf_df["relevance_shift"] > 0]

    delta_loss_times_samples_neg = (
        neg_df.assign(stat=lambda d: d["delta_loss"] * d["sample_count"])
        .groupby("param_idx")["stat"]
        .max()
    )
    delta_loss_times_samples_pos = (
        pos_df.assign(stat=lambda d: d["delta_loss"] * d["sample_count"])
        .groupby("param_idx")["stat"]
        .max()
    )

    neg_values = [delta_loss_times_samples_neg.get(param, 0) for param in params]
    pos_values = [delta_loss_times_samples_pos.get(param, 0) for param in params]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7, 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 0.9], "hspace": 0.13},
    )

    ax_top.set_yscale("log")
    for index, param in enumerate(params):
        y_neg = neg_values[index]
        y_pos = pos_values[index]
        color = dot_colors[index]

        ax_top.vlines(
            x=param,
            ymin=min(y_neg, y_pos),
            ymax=max(y_neg, y_pos),
            color=color,
            linewidth=2,
            linestyle=":",
        )
        ax_top.scatter(
            param,
            y_neg,
            color=color,
            edgecolor="black",
            s=100,
            linewidth=1,
            zorder=3,
        )
        ax_top.scatter(
            param,
            y_pos,
            color=color,
            edgecolor="black",
            s=100,
            linewidth=1,
            zorder=3,
        )

    ax_top.set_ylabel("Unnormalized Likelihood Diff (Log)", fontsize=10)
    for y_value, label in comparison_lines:
        ax_top.axhline(y=y_value, color="black", ls="--", lw=1.5, zorder=1)
        ax_top.text(
            max(params) * 0.94,
            y_value * 1.05,
            label,
            ha="left",
            va="bottom",
            fontsize=8,
        )

    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(True)

    ax_bottom.plot(
        params,
        sample_counts,
        color=sample_count_color,
        marker="o",
        linewidth=2,
        label="Sample Count",
    )
    ax_bottom.set_ylabel("Sample Count", color="black", fontsize=10)
    ax_bottom.tick_params(axis="y", colors="black")
    ax_bottom.set_yticks([0, 1e6, 2e6, 3e6])
    ax_bottom.set_xlabel("Parameter Index", fontsize=12)
    ax_bottom.set_xticks(params)
    ax_bottom.set_xticklabels(params, rotation=0)
    ax_bottom.spines["top"].set_visible(False)

    legend_ident = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=label.capitalize(),
        )
        for label, color in color_map.items()
    ]
    legend_elements = legend_ident + [
        Line2D(
            [0],
            [0],
            color=sample_count_color,
            lw=2,
            marker="o",
            label="Sample Count",
        )
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.00),
        ncol=4,
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.24)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    print("RQ3 identifiability CLI")
    print("Enter the results folder and dataset path to run the analysis.")

    folder_input = prompt_with_default(
        args.folder_path,
        "Folder path",
        DEFAULT_FOLDER_PATH,
    )
    data_input = prompt_with_default(
        args.data_path,
        "Data path",
        DEFAULT_DATA_PATH,
    )

    folder_path = resolve_existing_dir(folder_input)
    data_path = resolve_data_path(data_input)
    default_output_path, default_summary_csv = build_default_output_paths(data_path)

    output_input = prompt_with_default(
        args.output_path,
        "Plot output path",
        default_output_path,
    )
    summary_input = prompt_with_default(
        args.summary_csv,
        "Summary CSV path",
        default_summary_csv,
    )

    perf_df = load_perf_df(folder_path, data_path)
    identifiability_results, _ = compute_identifiability(
        perf_df,
        thresholds=args.thresholds,
    )

    summary_df = identifiability_results_to_df(identifiability_results)

    summary_path = resolve_output_path(summary_input)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary CSV to: {summary_path}")

    output_path = resolve_output_path(output_input)
    plot_identifiability_summary(identifiability_results, perf_df, output_path)
    print(f"Saved plot to: {output_path}")

    print(f"Using results folder: {folder_path}")
    print(f"Using dataset file: {data_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
