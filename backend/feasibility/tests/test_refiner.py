import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import pytest
from feasibility.core.refiner import CounterfactualFeasibilityRefiner
from unittest.mock import MagicMock

class TestRefiner:
    @pytest.fixture
    def setup(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 4)
        constraints = {
            'immutable': [0],
            'actionable': {
                1: {'type': 'lower_bound', 'bound': 'original'}
            }
        }
        refiner = CounterfactualFeasibilityRefiner(
            X_train=X_train,
            constraints_config=constraints
        )
        return refiner, X_train

    def test_initialization(self, setup):
        refiner, X_train = setup
        assert refiner.X_train.shape == X_train.shape
        assert len(refiner.constraint_mgr.constraints['immutable']) == 1

    def test_hard_constraint_enforcement(self, setup):
        refiner, _ = setup
        original = np.array([0.5, 1.0, 2.0, 3.0])
        raw_cf = np.array([0.6, 0.8, 2.1, 3.1])  # Violates lower bound on feature 1
        
        corrected = refiner._refine_single(original, raw_cf)
        assert corrected[0] == original[0]  # Immutable
        assert corrected[1] == original[1]  # Lower bound enforcement
        assert corrected[2] == raw_cf[2]
        assert corrected[3] == raw_cf[3]

    def test_optimization_trigger(self, setup):
        refiner, _ = setup
        refiner.density_validator.is_plausible = MagicMock(return_value=False)
        
        original = np.array([0.5, 1.0, 2.0, 3.0])
        raw_cf = np.array([0.5, 1.2, 2.1, 3.1])
        
        corrected = refiner._refine_single(original, raw_cf)
        # Optimization should run since CF is implausible
        assert not np.array_equal(corrected, raw_cf)
        assert corrected[0] == original[0]  # Immutable preserved

    def test_batch_processing(self, setup):
        refiner, _ = setup
        original = np.array([0.5, 1.0, 2.0, 3.0])
        raw_cfs = [
            np.array([0.6, 0.9, 2.1, 3.1]),
            np.array([0.5, 1.1, 2.0, 3.0]),
            np.array([0.5, 1.0, 1.9, 3.2])
        ]
        
        feasible_cfs = refiner.refine(original, raw_cfs)
        assert len(feasible_cfs) == len(raw_cfs)
        for cf in feasible_cfs:
            assert cf[0] == original[0]  # All immutable features preserved