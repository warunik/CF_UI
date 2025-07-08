import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from feasibility.core.refiner import CounterfactualFeasibilityRefiner
from feasibility.utils.metrics import calculate_feasibility_score