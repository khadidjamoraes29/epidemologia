from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict


@dataclass(frozen=True)
class ODEParameters:
    population: int = 10_000
    initial_exposed: float = 10.0
    initial_infected: float = 2.0
    initial_recovered: float = 0.0
    initial_contamination: float = 0.02

    beta: float = 0.035
    sigma: float = 1.0 / 10.0
    gamma: float = 1.0 / 14.0
    k_saturation: float = 0.25

    alpha: float = 0.002
    phi: float = 0.12
    eta: float = 0.08

    flood_start: float = 5.0
    flood_end: float = 15.0
    flood_intensity: float = 1.0
    simulation_days: int = 120


@dataclass(frozen=True)
class CAParameters:
    grid_size: int = 60
    occupancy_rate: float = 0.70
    initial_exposed_fraction: float = 0.004
    initial_infected_fraction: float = 0.001

    beta: float = 0.30
    sigma: float = 1.0 / 10.0
    gamma: float = 1.0 / 14.0

    alpha: float = 0.004
    phi: float = 0.20
    eta: float = 0.10
    diffusion: float = 0.18

    flood_start: int = 5
    flood_end: int = 15
    flood_intensity: float = 1.0

    contamination_threshold: float = 0.25
    simulation_days: int = 120
    seed: int = 42


def ode_scenarios(base: ODEParameters | None = None) -> Dict[str, ODEParameters]:
    p = base or ODEParameters()
    return {
        "referencia": p,
        "enchente_intensa": replace(p, phi=p.phi * 1.5, flood_end=p.flood_end + 5),
        "protecao_individual": replace(p, beta=p.beta * 0.60),
        "descontaminacao": replace(p, eta=p.eta * 1.50),
        "intervencao_combinada": replace(
            p, beta=p.beta * 0.60, eta=p.eta * 1.50
        ),
    }


def ca_scenarios(base: CAParameters | None = None) -> Dict[str, CAParameters]:
    p = base or CAParameters()
    return {
        "referencia": p,
        "enchente_intensa": replace(p, phi=p.phi * 1.5, flood_end=p.flood_end + 5),
        "protecao_individual": replace(p, beta=p.beta * 0.60),
        "descontaminacao": replace(p, eta=p.eta * 1.50),
        "intervencao_combinada": replace(
            p, beta=p.beta * 0.60, eta=p.eta * 1.50
        ),
    }
