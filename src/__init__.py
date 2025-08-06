"""
Multi-Asset Optimal Stopping with Budget Constraints

This package provides a complete implementation of multi-asset optimal stopping
algorithms with budget constraints, implementing the mathematical framework
described in the accompanying LaTeX document.

Package Structure:
- src.algorithm: Core mathematical algorithms and components
- src.solver: High-level solver interface and configuration
- src.example: Usage examples and demonstrations

Quick Start:
    from src import MultiAssetOptimalStoppingSolver, SolverConfig
    from src.algorithm import PayoffFunctions
    import numpy as np
    
    # Create solver
    config = SolverConfig(n_assets=2, max_time=1.0)
    solver = MultiAssetOptimalStoppingSolver(config)
    
    # Set market parameters  
    mu = np.array([0.05, 0.03])
    sigma = np.array([0.2, 0.15])
    correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
    solver.set_market_parameters(mu, sigma, correlation)
    
    # Set payoff functions
    payoffs = {
        0: PayoffFunctions.american_call(1.0),
        1: PayoffFunctions.american_put(1.0)
    }
    solver.set_payoff_functions(payoffs)
    
    # Solve and get results
    results = solver.solve()
    value = solver.get_value(t=0.5, wealth=1.0, prices=[1.1, 0.9], active_set={0, 1})
"""

from .solver import MultiAssetOptimalStoppingSolver, SolverConfig

__all__ = [
    'MultiAssetOptimalStoppingSolver',
    'SolverConfig'
]

__version__ = '1.0.0'
__author__ = 'Lianghao Chen'
__description__ = 'Multi-Asset Optimal Stopping with Budget Constraints'