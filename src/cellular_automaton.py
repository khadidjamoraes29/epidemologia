from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .box_counting import box_counting_dimension
from .config import CAParameters

EMPTY = 0
SUSCEPTIBLE = 1
EXPOSED = 2
INFECTED = 3
RECOVERED = 4


@dataclass
class CAResult:
    metrics: Dict[str, Any]
    history: pd.DataFrame
    state_history: List[np.ndarray]
    contamination_history: List[np.ndarray]
    vulnerability: np.ndarray
    flood_map: np.ndarray
    rodent_risk: np.ndarray


def neighbor_mean(matrix: np.ndarray) -> np.ndarray:
    padded = np.pad(matrix, 1, mode="edge")
    total = np.zeros_like(matrix, dtype=float)
    rows, cols = matrix.shape
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue
            total += padded[di:di + rows, dj:dj + cols]
    return total / 8.0


def _smooth_random_field(
    rng: np.random.Generator,
    size: int,
    iterations: int = 8,
) -> np.ndarray:
    field = rng.random((size, size))
    for _ in range(iterations):
        field = 0.55 * field + 0.45 * neighbor_mean(field)
    min_v = float(field.min())
    max_v = float(field.max())
    if np.isclose(min_v, max_v):
        return np.zeros_like(field)
    return (field - min_v) / (max_v - min_v)


def initialize_environment(
    p: CAParameters,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.random.Generator,
]:
    if not 0.0 < p.occupancy_rate <= 1.0:
        raise ValueError("occupancy_rate deve estar no intervalo (0, 1].")

    rng = np.random.default_rng(p.seed)
    size = p.grid_size

    occupied = rng.random((size, size)) < p.occupancy_rate
    states = np.full((size, size), EMPTY, dtype=np.int8)
    states[occupied] = SUSCEPTIBLE

    positions = np.argwhere(occupied)
    n_people = len(positions)
    if n_people == 0:
        raise ValueError("A grade não possui indivíduos.")

    n_exposed = max(1, int(round(n_people * p.initial_exposed_fraction)))
    n_infected = max(1, int(round(n_people * p.initial_infected_fraction)))
    n_selected = min(n_people, n_exposed + n_infected)
    selected = positions[rng.choice(n_people, size=n_selected, replace=False)]

    split = min(n_exposed, len(selected))
    for row, col in selected[:split]:
        states[row, col] = EXPOSED
    for row, col in selected[split:]:
        states[row, col] = INFECTED

    vulnerability = np.clip(
        0.65 + 0.70 * _smooth_random_field(rng, size),
        0.50,
        1.40,
    )
    rodent_risk = _smooth_random_field(rng, size)
    flood_map = _smooth_random_field(rng, size)

    y_axis = np.linspace(0.0, 1.0, size)[:, None]
    low_area = np.exp(-((y_axis - 0.72) ** 2) / (2.0 * 0.16 ** 2))
    flood_map = np.clip(0.55 * flood_map + 0.45 * low_area, 0.0, 1.0)

    contamination = np.clip(0.03 * rodent_risk, 0.0, 1.0)
    return states, contamination, vulnerability, flood_map, rodent_risk, rng


def flood_active(day: int, p: CAParameters) -> float:
    return p.flood_intensity if p.flood_start <= day <= p.flood_end else 0.0


def _count_states(states: np.ndarray) -> Dict[str, int]:
    return {
        "vazias": int(np.sum(states == EMPTY)),
        "S": int(np.sum(states == SUSCEPTIBLE)),
        "E": int(np.sum(states == EXPOSED)),
        "I": int(np.sum(states == INFECTED)),
        "R": int(np.sum(states == RECOVERED)),
    }


def simulate_ca(p: CAParameters, scenario_name: str) -> CAResult:
    (
        states,
        contamination,
        vulnerability,
        flood_map,
        rodent_risk,
        rng,
    ) = initialize_environment(p)

    state_history: List[np.ndarray] = [states.copy()]
    contamination_history: List[np.ndarray] = [contamination.copy()]
    records: List[Dict[str, float | int]] = []

    counts = _count_states(states)
    cumulative_symptomatic = counts["I"]
    initial_mask = contamination >= p.contamination_threshold

    records.append(
        {
            "dia": 0,
            **counts,
            "novos_expostos": 0,
            "novos_sintomaticos": 0,
            "casos_sintomaticos_acumulados": cumulative_symptomatic,
            "contaminacao_media": float(contamination.mean()),
            "contaminacao_maxima": float(contamination.max()),
            "area_contaminada": int(initial_mask.sum()),
            "dimensao_box_counting": box_counting_dimension(initial_mask).dimension,
        }
    )

    p_e_to_i = 1.0 - np.exp(-p.sigma)
    p_i_to_r = 1.0 - np.exp(-p.gamma)

    for day in range(1, p.simulation_days + 1):
        local_mean = neighbor_mean(contamination)
        contamination = (
            (1.0 - p.eta) * contamination
            + p.diffusion * (local_mean - contamination)
            + p.alpha * rodent_risk
            + p.phi * flood_map * flood_active(day, p)
        )
        contamination = np.clip(contamination, 0.0, 1.0)

        next_states = states.copy()

        p_exposure = 1.0 - np.exp(-p.beta * vulnerability * contamination)
        new_exposed = (
            (states == SUSCEPTIBLE)
            & (rng.random(states.shape) < p_exposure)
        )
        new_infected = (
            (states == EXPOSED)
            & (rng.random(states.shape) < p_e_to_i)
        )
        new_recovered = (
            (states == INFECTED)
            & (rng.random(states.shape) < p_i_to_r)
        )

        next_states[new_exposed] = EXPOSED
        next_states[new_infected] = INFECTED
        next_states[new_recovered] = RECOVERED
        states = next_states

        new_symptomatic = int(new_infected.sum())
        cumulative_symptomatic += new_symptomatic
        active_mask = contamination >= p.contamination_threshold
        counts = _count_states(states)

        records.append(
            {
                "dia": day,
                **counts,
                "novos_expostos": int(new_exposed.sum()),
                "novos_sintomaticos": new_symptomatic,
                "casos_sintomaticos_acumulados": cumulative_symptomatic,
                "contaminacao_media": float(contamination.mean()),
                "contaminacao_maxima": float(contamination.max()),
                "area_contaminada": int(active_mask.sum()),
                "dimensao_box_counting": box_counting_dimension(active_mask).dimension,
            }
        )

        state_history.append(states.copy())
        contamination_history.append(contamination.copy())

    history = pd.DataFrame(records)
    peak_idx = int(history["I"].idxmax())
    peak = history.loc[peak_idx]
    occupied_population = int(history.loc[0, ["S", "E", "I", "R"]].sum())

    metrics = {
        "cenario": scenario_name,
        "populacao_ocupada": occupied_population,
        "casos_sintomaticos_acumulados": int(
            history["casos_sintomaticos_acumulados"].iloc[-1]
        ),
        "pico_sintomaticos": int(peak["I"]),
        "dia_pico": int(peak["dia"]),
        "area_contaminada_maxima": int(history["area_contaminada"].max()),
        "contaminacao_maxima": float(history["contaminacao_maxima"].max()),
        "dimensao_box_counting_no_pico": float(
            history.loc[peak_idx, "dimensao_box_counting"]
        ),
        "dia_area_contaminada_maxima": int(
            history.loc[history["area_contaminada"].idxmax(), "dia"]
        ),
    }

    return CAResult(
        metrics=metrics,
        history=history,
        state_history=state_history,
        contamination_history=contamination_history,
        vulnerability=vulnerability,
        flood_map=flood_map,
        rodent_risk=rodent_risk,
    )
