from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .box_counting import box_counting_dimension
from .cellular_automaton import CAResult


def save_ode_curves(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for column in ["S", "E", "I", "R"]:
        ax.plot(df["dia"], df[column], label=column)
    ax.set_title("Dinâmica do modelo SEIR ambiental")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Número de indivíduos")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_environment_curve(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["dia"], df["C"])
    ax.set_title("Evolução da contaminação ambiental")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Contaminação C(t)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_scenario_bars(
    metrics: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(metrics["cenario"], metrics[value_column])
    ax.set_title(title)
    ax.set_xlabel("Cenário")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_ca_time_series(history: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for column in ["S", "E", "I", "R"]:
        ax.plot(history["dia"], history[column], label=column)
    ax.set_title("Dinâmica epidemiológica do autômato celular")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Número de indivíduos")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_state_map(states: np.ndarray, day: int, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(states, interpolation="nearest", vmin=0, vmax=4)
    ax.set_title(f"Estados epidemiológicos — dia {day}")
    ax.set_xlabel("Coluna da grade")
    ax.set_ylabel("Linha da grade")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3, 4])
    colorbar.ax.set_yticklabels(["Vazio", "S", "E", "I", "R"])
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_contamination_map(
    contamination: np.ndarray,
    day: int,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(
        contamination,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(f"Contaminação ambiental — dia {day}")
    ax.set_xlabel("Coluna da grade")
    ax.set_ylabel("Linha da grade")
    fig.colorbar(image, ax=ax, label="Nível de contaminação")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_box_counting_plot(
    contamination: np.ndarray,
    threshold: float,
    output: Path,
) -> float:
    result = box_counting_dimension(contamination >= threshold)
    fig, ax = plt.subplots(figsize=(8, 6))

    if len(result.log_inverse_scale) > 0:
        ax.scatter(result.log_inverse_scale, result.log_counts)
        if len(result.log_inverse_scale) >= 2:
            coefficients = np.polyfit(
                result.log_inverse_scale,
                result.log_counts,
                1,
            )
            ax.plot(
                result.log_inverse_scale,
                np.polyval(coefficients, result.log_inverse_scale),
            )

    ax.set_title(f"Box-counting da área contaminada (D = {result.dimension:.3f})")
    ax.set_xlabel("log(1/ε)")
    ax.set_ylabel("log N(ε)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return result.dimension


def save_reference_ca_figures(
    result: CAResult,
    threshold: float,
    output_dir: Path,
) -> None:
    history = result.history
    peak_day = int(history.loc[history["I"].idxmax(), "dia"])
    final_day = int(history["dia"].iloc[-1])

    for label, day in {
        "inicial": 0,
        "pico": peak_day,
        "final": final_day,
    }.items():
        save_state_map(
            result.state_history[day],
            day,
            output_dir / f"ca_estados_{label}.png",
        )
        save_contamination_map(
            result.contamination_history[day],
            day,
            output_dir / f"ca_contaminacao_{label}.png",
        )

    save_box_counting_plot(
        result.contamination_history[peak_day],
        threshold,
        output_dir / "ca_box_counting_pico.png",
    )
