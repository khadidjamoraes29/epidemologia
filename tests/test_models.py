from __future__ import annotations

import unittest
import numpy as np

from src.box_counting import box_counting_dimension
from src.cellular_automaton import simulate_ca
from src.compartmental_model import simulate_ode
from src.config import CAParameters, ODEParameters


class ODEModelTests(unittest.TestCase):
    def test_population_is_conserved(self) -> None:
        p = ODEParameters(simulation_days=30)
        result = simulate_ode(p)
        self.assertTrue(
            np.allclose(
                result["populacao_humana"].to_numpy(),
                p.population,
                rtol=1e-6,
                atol=1e-4,
            )
        )

    def test_values_are_nonnegative(self) -> None:
        result = simulate_ode(ODEParameters(simulation_days=20))
        self.assertTrue(
            (result[["S", "E", "I", "R", "C"]] >= 0).all().all()
        )


class CellularAutomatonTests(unittest.TestCase):
    def test_population_is_conserved(self) -> None:
        result = simulate_ca(
            CAParameters(grid_size=20, simulation_days=15, seed=7),
            "teste",
        )
        occupied = result.history[["S", "E", "I", "R"]].sum(axis=1)
        self.assertEqual(int(occupied.nunique()), 1)

    def test_contamination_is_bounded(self) -> None:
        result = simulate_ca(
            CAParameters(grid_size=20, simulation_days=10, seed=11),
            "teste",
        )
        for grid in result.contamination_history:
            self.assertGreaterEqual(float(grid.min()), 0.0)
            self.assertLessEqual(float(grid.max()), 1.0)


class BoxCountingTests(unittest.TestCase):
    def test_full_plane_dimension_is_near_two(self) -> None:
        result = box_counting_dimension(np.ones((64, 64), dtype=bool))
        self.assertAlmostEqual(result.dimension, 2.0, delta=0.15)

    def test_empty_mask_dimension_is_zero(self) -> None:
        result = box_counting_dimension(np.zeros((32, 32), dtype=bool))
        self.assertEqual(result.dimension, 0.0)


if __name__ == "__main__":
    unittest.main()
