"""
Multi-Asset Optimal Stopping Algorithm - Mathematical Components

This package contains the core mathematical algorithms for multi-asset optimal stopping
with budget constraints as described in the accompanying LaTeX document. 

Mathematical Components:
- StateGrid: Manages discretized state space
- ValueFunction: Stores and manages value function over state space  
- PortfolioOptimizer: Handles constrained portfolio optimization
- ContinuationValueComputer: Computes continuation values using finite differences
- BackwardInductionSolver: Core backward induction algorithm implementation
- PayoffFunctions: Mathematical payoff function definitions
- Utility functions: Parameter generation, correlation matrices, validation tools

Example Usage:
    from src.algorithm import BackwardInductionSolver, PayoffFunctions, GridParameters
    import numpy as np
    
    # Create payoff functions
    payoffs = {
        0: PayoffFunctions.american_call(1.0),
        1: PayoffFunctions.american_put(1.0)
    }
    
    # Set up algorithm
    solver = BackwardInductionSolver(
        n_assets=2,
        mu=np.array([0.05, 0.03]),
        sigma=np.array([0.2, 0.15]),
        correlation=np.array([[1.0, 0.3], [0.3, 1.0]]),
        payoff_functions=payoffs
    )
    
    # Solve using backward induction
    value_function, policies, stopping_regions = solver.solve()
"""

from .grid import StateGrid, ValueFunction, GridParameters
from .optimization import PortfolioOptimizer, finite_difference_gradient  
from .continuation import ContinuationValueComputer, InterpolationHelper
from .backward_induction import BackwardInductionSolver
from .utils import (
    PayoffFunctions, 
    CorrelationMatrixGenerator, 
    ParameterGenerator,
    ResultAnalyzer,
    ValidationTools
)

__all__ = [
    # Core algorithm classes
    'BackwardInductionSolver',
    
    # Grid and state management
    'StateGrid',
    'ValueFunction', 
    'GridParameters',
    
    # Mathematical optimization components
    'PortfolioOptimizer',
    'ContinuationValueComputer',
    'InterpolationHelper',
    
    # Mathematical utility functions and classes
    'PayoffFunctions',
    'CorrelationMatrixGenerator',
    'ParameterGenerator', 
    'ResultAnalyzer',
    'ValidationTools',
    
    # Mathematical functions
    'finite_difference_gradient'
]

__version__ = '1.0.0'
__author__ = 'Lianghao Chen'
__description__ = 'Multi-Asset Optimal Stopping Mathematical Core'