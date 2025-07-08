import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import pytest
from feasibility.core.constraints import ConstraintManager

def test_immutable_constraints():
    constraints = {'immutable': [0, 2]}
    mgr = ConstraintManager(constraints)
    original = np.array([1.0, 2.0, 3.0, 4.0])
    cf = np.array([5.0, 6.0, 7.0, 8.0])
    
    corrected = mgr.apply_hard_constraints(original, cf)
    assert corrected[0] == original[0]
    assert corrected[2] == original[2]
    assert corrected[1] == cf[1]
    assert corrected[3] == cf[3]

def test_lower_bound_constraint():
    constraints = {
        'actionable': {
            1: {'type': 'lower_bound', 'bound': 'original'}
        }
    }
    mgr = ConstraintManager(constraints)
    original = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Test below bound
    cf = np.array([1.0, 1.5, 3.0, 4.0])
    corrected = mgr.apply_hard_constraints(original, cf)
    assert corrected[1] == original[1]
    
    # Test above bound
    cf = np.array([1.0, 2.5, 3.0, 4.0])
    corrected = mgr.apply_hard_constraints(original, cf)
    assert corrected[1] == cf[1]

def test_upper_bound_constraint():
    constraints = {
        'actionable': {
            1: {'type': 'upper_bound', 'bound': 3.0}
        }
    }
    mgr = ConstraintManager(constraints)
    original = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Test above bound
    cf = np.array([1.0, 3.5, 3.0, 4.0])
    corrected = mgr.apply_hard_constraints(original, cf)
    assert corrected[1] == 3.0
    
    # Test below bound
    cf = np.array([1.0, 2.5, 3.0, 4.0])
    corrected = mgr.apply_hard_constraints(original, cf)
    assert corrected[1] == cf[1]

def test_monotonic_constraint():
    constraints = {
        'actionable': {
            1: {'type': 'monotonic', 'direction': 'increase'}
        }
    }
    mgr = ConstraintManager(constraints)
    original = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Test decrease (should be blocked)
    cf = np.array([1.0, 1.5, 3.0, 4.0])
    corrected = mgr.apply_hard_constraints(original, cf)
    assert corrected[1] == original[1]
    
    # Test increase (should be allowed)
    cf = np.array([1.0, 2.5, 3.0, 4.0])
    corrected = mgr.apply_hard_constraints(original, cf)
    assert corrected[1] == cf[1]

def test_causal_constraints():
    constraints = {
        'causal_rules': [
            {'if_feature': 1, 'then_feature': 2, 'relation': 'positive'}
        ]
    }
    mgr = ConstraintManager(constraints)
    original = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Valid change (both increase)
    cf = np.array([1.0, 3.0, 3.5, 4.0])
    violations = mgr.check_causal_constraints(original, cf)
    assert len(violations) == 0
    
    # Invalid change (if increases but then decreases)
    cf = np.array([1.0, 3.0, 2.5, 4.0])
    violations = mgr.check_causal_constraints(original, cf)
    assert len(violations) == 1