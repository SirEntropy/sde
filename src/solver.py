"""
Multi-Asset Optimal Stopping Solver

This module provides a high-level interface to the multi-asset optimal stopping
algorithm described in the LaTeX document. It orchestrates all components and
provides a clean API for users.
"""

import numpy as np
from typing import Dict, Set, Tuple, Optional, Callable, List
import logging
import time
from dataclasses import dataclass, field

from .algorithm.grid import GridParameters
from .algorithm.backward_induction import BackwardInductionSolver
from .algorithm.utils import PayoffFunctions, ParameterGenerator, ResultAnalyzer, ValidationTools


@dataclass
class SolverConfig:
    """Configuration for the optimal stopping solver."""
    
    # Problem parameters
    n_assets: int = 3
    max_time: float = 1.0
    risk_free_rate: float = 0.023
    
    # Grid parameters
    grid_params: GridParameters = field(default_factory=GridParameters)
    
    # Numerical parameters
    tolerance: float = 1e-6
    max_iterations: int = 10
    
    # Computational parameters
    use_parallel: bool = False
    use_numba: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_progress: bool = True


class MultiAssetOptimalStoppingSolver:
    """
    Main solver class for multi-asset optimal stopping problems.
    
    This class implements the backward induction algorithm with policy iteration
    described in the LaTeX document. It provides a high-level interface for
    setting up and solving multi-asset optimal stopping problems.
    
    Example usage:
        # Create solver
        solver = MultiAssetOptimalStoppingSolver(config)
        
        # Set market parameters
        solver.set_market_parameters(mu, sigma, correlation)
        
        # Set payoff functions
        payoffs = {0: PayoffFunctions.american_call(1.0), 
                   1: PayoffFunctions.american_put(1.0)}
        solver.set_payoff_functions(payoffs)
        
        # Solve
        results = solver.solve()
        
        # Analyze results
        value = solver.get_value(t=0.5, wealth=1.0, prices=[1.1, 0.9], 
                                active_set={0, 1})
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        """Initialize the solver with given configuration."""
        self.config = config or SolverConfig()
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level.upper()))
        self.logger = logging.getLogger(__name__)
        
        # Problem parameters (to be set by user)
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.correlation: Optional[np.ndarray] = None
        self.payoff_functions: Optional[Dict[int, Callable]] = None
        
        # Internal solver
        self._solver: Optional[BackwardInductionSolver] = None
        self._solved: bool = False
        
        # Results
        self.results: Optional[Dict] = None
        self.analyzer: Optional[ResultAnalyzer] = None
        
        self.logger.info(f"Initialized MultiAssetOptimalStoppingSolver with {self.config.n_assets} assets")
    
    def set_market_parameters(self, mu: np.ndarray, sigma: np.ndarray, 
                             correlation: np.ndarray) -> None:
        """
        Set market parameters for asset dynamics.
        
        Args:
            mu: Drift vector (n_assets,)
            sigma: Volatility vector (n_assets,)
            correlation: Correlation matrix (n_assets, n_assets)
        """
        if len(mu) != self.config.n_assets:
            raise ValueError(f"mu must have {self.config.n_assets} elements")
        if len(sigma) != self.config.n_assets:
            raise ValueError(f"sigma must have {self.config.n_assets} elements")
        if correlation.shape != (self.config.n_assets, self.config.n_assets):
            raise ValueError(f"correlation must be {self.config.n_assets} x {self.config.n_assets}")
        
        # Validate correlation matrix
        if not np.allclose(correlation, correlation.T):
            raise ValueError("Correlation matrix must be symmetric")
        
        eigenvals = np.linalg.eigvals(correlation)
        if np.any(eigenvals <= 0):
            raise ValueError("Correlation matrix must be positive definite")
        
        self.mu = mu.copy()
        self.sigma = sigma.copy()
        self.correlation = correlation.copy()
        
        self.logger.info("Market parameters set successfully")
        self._solved = False  # Reset solution status
    
    def set_payoff_functions(self, payoff_functions: Dict[int, Callable]) -> None:
        """
        Set payoff functions G_i(x, s_i) for each asset.
        
        Args:
            payoff_functions: Dict mapping asset index to payoff function
        """
        for asset_id in payoff_functions.keys():
            if not (0 <= asset_id < self.config.n_assets):
                raise ValueError(f"Asset ID {asset_id} out of range [0, {self.config.n_assets})")
        
        self.payoff_functions = payoff_functions.copy()
        self.logger.info(f"Payoff functions set for {len(payoff_functions)} assets")
        self._solved = False
    
    def generate_default_parameters(self, seed: int = 42) -> None:
        """Generate reasonable default parameters for testing."""
        mu, sigma, correlation = ParameterGenerator.generate_market_parameters(
            self.config.n_assets, seed
        )
        self.set_market_parameters(mu, sigma, correlation)
        
        payoffs = ParameterGenerator.generate_payoff_functions(self.config.n_assets)
        self.set_payoff_functions(payoffs)
        
        self.logger.info("Default parameters generated")
    
    def solve(self) -> Dict:
        """
        Solve the multi-asset optimal stopping problem.
        
        Returns:
            Dictionary containing solution results
        """
        if not self._validate_parameters():
            raise ValueError("Parameters not properly set. Call set_market_parameters() and set_payoff_functions()")
        
        self.logger.info("Starting multi-asset optimal stopping solution")
        start_time = time.time()
        
        # Create internal solver
        self._solver = BackwardInductionSolver(
            n_assets=self.config.n_assets,
            mu=self.mu,
            sigma=self.sigma,
            correlation=self.correlation,
            payoff_functions=self.payoff_functions,
            grid_params=self.config.grid_params,
            risk_free_rate=self.config.risk_free_rate,
            max_time=self.config.max_time
        )
        
        # Solve using backward induction
        value_function, optimal_policies, stopping_regions = self._solver.solve(
            parallel=self.config.use_parallel
        )
        
        solve_time = time.time() - start_time
        self.logger.info(f"Solution completed in {solve_time:.2f} seconds")
        
        # Validate solution
        validation_results = self._validate_solution()
        
        # Package results
        self.results = {
            'value_function': value_function,
            'optimal_policies': optimal_policies,
            'stopping_regions': stopping_regions,
            'solve_time': solve_time,
            'validation': validation_results,
            'config': self.config,
            'parameters': {
                'mu': self.mu,
                'sigma': self.sigma,
                'correlation': self.correlation
            }
        }
        
        # Create analyzer
        self.analyzer = ResultAnalyzer(self._solver)
        
        self._solved = True
        self.logger.info("Solution package created successfully")
        
        return self.results
    
    def get_value(self, t: float, wealth: float, prices: List[float], 
                  active_set: Set[int]) -> float:
        """
        Get value function at specified state.
        
        Args:
            t: Time
            wealth: Current wealth
            prices: Asset prices
            active_set: Set of active (not yet stopped) assets
            
        Returns:
            Value function value
        """
        if not self._solved:
            raise RuntimeError("Must call solve() first")
        
        return self._solver.get_value(t, wealth, np.array(prices), active_set)
    
    def get_optimal_policy(self, t: float, wealth: float, prices: List[float],
                          active_set: Set[int]) -> Tuple[np.ndarray, Optional[int]]:
        """
        Get optimal policy at specified state.
        
        Args:
            t: Time
            wealth: Current wealth  
            prices: Asset prices
            active_set: Set of active assets
            
        Returns:
            (portfolio_weights, asset_to_stop)
        """
        if not self._solved:
            raise RuntimeError("Must call solve() first")
        
        return self._solver.get_optimal_policy(t, wealth, np.array(prices), active_set)
    
    def simulate_path(self, initial_wealth: float, initial_prices: np.ndarray,
                     n_steps: int = 100, seed: int = 42) -> Dict:
        """
        Simulate an optimal path using the computed policy.
        
        Args:
            initial_wealth: Starting wealth
            initial_prices: Starting asset prices
            n_steps: Number of simulation steps
            seed: Random seed
            
        Returns:
            Dictionary with simulation results
        """
        if not self._solved:
            raise RuntimeError("Must call solve() first")
        
        np.random.seed(seed)
        
        # Initialize arrays
        times = np.linspace(0, self.config.max_time, n_steps)
        dt = times[1] - times[0]
        
        wealth_path = [initial_wealth]
        price_paths = [initial_prices.copy()]
        portfolio_path = []
        stopping_decisions = []
        active_set = set(range(self.config.n_assets))
        
        current_wealth = initial_wealth
        current_prices = initial_prices.copy()
        
        for i, t in enumerate(times[:-1]):
            if len(active_set) == 0:
                break
            
            # Get optimal policy
            portfolio, stop_asset = self.get_optimal_policy(
                t, current_wealth, current_prices.tolist(), active_set
            )
            
            portfolio_path.append(portfolio.copy())
            stopping_decisions.append(stop_asset)
            
            # Execute stopping decision
            if stop_asset is not None:
                # Realize payoff
                payoff = self.payoff_functions[stop_asset](current_wealth, current_prices[stop_asset])
                current_wealth += payoff
                active_set.remove(stop_asset)
                self.logger.info(f"Stopped asset {stop_asset} at t={t:.3f}, payoff={payoff:.4f}")
            
            # Simulate next step (simplified)
            if i < len(times) - 2:  # Not the last step
                # Generate random shocks
                dW = np.random.normal(0, np.sqrt(dt), self.config.n_assets)
                
                # Update prices
                for j in range(self.config.n_assets):
                    price_change = (self.mu[j] * current_prices[j] * dt + 
                                  self.sigma[j] * current_prices[j] * dW[j])
                    current_prices[j] += price_change
                    current_prices[j] = max(current_prices[j], 0.01)  # Floor at 1 cent
                
                # Update wealth based on portfolio
                wealth_change = current_wealth * (
                    self.config.risk_free_rate * (1 - np.sum(portfolio)) * dt +
                    np.sum(portfolio * self.mu) * dt +
                    np.sum(portfolio * self.sigma * dW)
                )
                current_wealth += wealth_change
                current_wealth = max(current_wealth, 0.01)  # Floor at 1 cent
            
            wealth_path.append(current_wealth)
            price_paths.append(current_prices.copy())
        
        return {
            'times': times[:len(wealth_path)],
            'wealth_path': np.array(wealth_path),
            'price_paths': np.array(price_paths),
            'portfolio_path': portfolio_path,
            'stopping_decisions': stopping_decisions,
            'final_wealth': current_wealth,
            'final_active_set': active_set
        }
    
    def _validate_parameters(self) -> bool:
        """Validate that all required parameters are set."""
        return (self.mu is not None and 
                self.sigma is not None and 
                self.correlation is not None and 
                self.payoff_functions is not None)
    
    def _validate_solution(self) -> Dict:
        """Validate the computed solution."""
        if self._solver is None:
            return {'valid': False, 'reason': 'No solver available'}
        
        try:
            # Check boundary conditions
            boundary_ok = ValidationTools.check_boundary_conditions(self._solver)
            
            # Check monotonicity
            monotonicity = ValidationTools.check_monotonicity(self._solver)
            
            return {
                'valid': boundary_ok and all(monotonicity.values()),
                'boundary_conditions': boundary_ok,
                'monotonicity': monotonicity
            }
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def plot_results(self, active_set: Optional[Set[int]] = None, save_dir: Optional[str] = None):
        """Plot key results."""
        if not self._solved or self.analyzer is None:
            raise RuntimeError("Must call solve() first")
        
        if active_set is None:
            active_set = set(range(self.config.n_assets))
        
        # Plot value function
        self.analyzer.plot_value_function(
            active_set, t_idx=0, 
            save_path=f"{save_dir}/value_function.png" if save_dir else None
        )
        
        # Plot stopping regions for each asset
        for asset_id in active_set:
            self.analyzer.plot_stopping_regions(
                asset_id, active_set, t_idx=0,
                save_path=f"{save_dir}/stopping_region_{asset_id}.png" if save_dir else None
            )
    
    def export_results(self, filename: str):
        """Export results to file."""
        if not self._solved or self.analyzer is None:
            raise RuntimeError("Must call solve() first")
        
        self.analyzer.export_results(filename)
        self.logger.info(f"Results exported to {filename}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics of the solution."""
        if not self._solved:
            raise RuntimeError("Must call solve() first")
        
        return {
            'n_assets': self.config.n_assets,
            'grid_size': (self.config.grid_params.n_time_steps,
                         self.config.grid_params.n_wealth_points,
                         self.config.grid_params.n_price_points),
            'solve_time': self.results['solve_time'],
            'validation_passed': self.results['validation']['valid'],
            'n_active_sets': len(self._solver.grid.active_sets),
            'max_value': self._get_max_value(),
            'parameters': {
                'mu_range': (float(np.min(self.mu)), float(np.max(self.mu))),
                'sigma_range': (float(np.min(self.sigma)), float(np.max(self.sigma))),
                'correlation_range': (float(np.min(self.correlation)), float(np.max(self.correlation)))
            }
        }
    
    def _get_max_value(self) -> float:
        """Get maximum value across all states."""
        max_val = 0.0
        for active_key, value_array in self._solver.value_function.values.items():
            max_val = max(max_val, float(np.max(value_array)))
        return max_val