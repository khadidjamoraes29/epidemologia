from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .config import ODEParameters


def flood_function(t: float, p: ODEParameters) -> float:
    return p.flood_intensity if p.flood_start <= t <= p.flood_end else 0.0


def seir_environment_rhs(t: float, y: np.ndarray, p: ODEParameters) -> np.ndarray:
    s, e, i, r, c = y
    c = max(float(c), 0.0)
    lambda_t = p.beta * c / (p.k_saturation + c)

    return np.array(
        [
            -lambda_t * s,
            lambda_t * s - p.sigma * e,
            p.sigma * e - p.gamma * i,
            p.gamma * i,
            p.alpha + p.phi * flood_function(t, p) - p.eta * c,
        ],
        dtype=float,
    )


def initial_conditions(p: ODEParameters) -> np.ndarray:
    s0 = p.population - p.initial_exposed - p.initial_infected - p.initial_recovered
    if s0 < 0:
        raise ValueError("A soma dos compartimentos iniciais excede a população.")
    return np.array(
        [
            s0,
            p.initial_exposed,
            p.initial_infected,
            p.initial_recovered,
            p.initial_contamination,
        ],
        dtype=float,
    )


def simulate_ode(p: ODEParameters) -> pd.DataFrame:
    if p.simulation_days <= 0:
        raise ValueError("simulation_days deve ser positivo.")

    t_eval = np.arange(0, p.simulation_days + 1, dtype=float)
    solution = solve_ivp(
        fun=lambda t, y: seir_environment_rhs(t, y, p),
        t_span=(0.0, float(p.simulation_days)),
        y0=initial_conditions(p),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-9,
        max_step=0.25,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    values = np.maximum(solution.y.T, 0.0)
    df = pd.DataFrame(values, columns=["S", "E", "I", "R", "C"])
    df.insert(0, "dia", solution.t.astype(int))
    df["populacao_humana"] = df[["S", "E", "I", "R"]].sum(axis=1)
    df["casos_sintomaticos_acumulados"] = df["I"] + df["R"]
    df["expostos_acumulados"] = p.population - df["S"]
    return df


def ode_metrics(df: pd.DataFrame, scenario_name: str, p: ODEParameters) -> Dict[str, Any]:
    peak_index = int(df["I"].idxmax())
    peak = df.loc[peak_index]
    return {
        "cenario": scenario_name,
        "populacao": p.population,
        "casos_sintomaticos_acumulados": float(
            df["casos_sintomaticos_acumulados"].iloc[-1]
        ),
        "expostos_acumulados": float(df["expostos_acumulados"].iloc[-1]),
        "pico_sintomaticos": float(peak["I"]),
        "dia_pico": int(peak["dia"]),
        "contaminacao_maxima": float(df["C"].max()),
        "suscetiveis_finais": float(df["S"].iloc[-1]),
    }
