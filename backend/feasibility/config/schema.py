from pydantic import BaseModel, Field
from typing import Dict, Union, Literal, Optional, List

class ActionableConstraintSpec(BaseModel):
    type: Literal['lower_bound', 'upper_bound', 'monotonic']
    bound: Union[Literal['original'], float, None] = None
    direction: Optional[Literal['increase', 'decrease']] = None

class CausalRule(BaseModel):
    if_feature: int
    then_feature: int
    relation: Literal['positive', 'negative']

class ConstraintsConfig(BaseModel):
    immutable: List[int] = Field(default_factory=list)
    actionable: Dict[int, ActionableConstraintSpec] = Field(default_factory=dict)
    causal_rules: List[CausalRule] = Field(default_factory=list)
    density_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    loss_weights: Dict[str, float] = Field(
        default={
            'proximity': 0.5,
            'density': 0.3,
            'fidelity': 0.1,
            'violation': 0.1
        }
    )

class RefinerConfig(BaseModel):
    max_iter: int = 100
    optimization_method: Literal['SLSQP', 'L-BFGS-B'] = 'SLSQP'
    verbose: bool = False