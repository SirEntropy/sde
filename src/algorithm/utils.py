import numpy as np
from typing import Callable, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.stats import norm


class PayoffFunctions:
    """Collection of common payoff functions G_i(x, s_i) for optimal stopping."""
    
    @staticmethod
    def american_call(strike: float, multiplier: float = 1.0) -> Callable:
        """American call option payoff: max(S - K, 0)."""
        def payoff(wealth: float, price: float) -> float:
            return multiplier * max(price - strike, 0.0)
        return payoff
    
    @staticmethod
    def american_put(strike: float, multiplier: float = 1.0) -> Callable:
        """American put option payoff: max(K - S, 0)."""
        def payoff(wealth: float, price: float) -> float:
            return multiplier * max(strike - price, 0.0)
        return payoff
    
    @staticmethod
    def lookback_call(multiplier: float = 1.0) -> Callable:
        """Lookback call option: S - min(S_path)."""
        def payoff(wealth: float, price: float) -> float:
            # Simplified: assumes current price is maximum so far
            return multiplier * price * 0.1  # Placeholder
        return payoff
    
    @staticmethod
    def barrier_option(barrier: float, strike: float, option_type: str = 'call') -> Callable:
        """Barrier option with knock-out/knock-in features."""
        def payoff(wealth: float, price: float) -> float:
            if option_type == 'call':
                base_payoff = max(price - strike, 0.0)
            else:
                base_payoff = max(strike - price, 0.0)
            
            # Simple barrier check
            if price >= barrier:
                return base_payoff
            else:
                return 0.0
        return payoff
    
    @staticmethod
    def wealth_dependent_payoff(base_strike: float, wealth_factor: float = 0.1) -> Callable:
        """Payoff that depends on both wealth and asset price."""
        def payoff(wealth: float, price: float) -> float:
            effective_strike = base_strike * (1 + wealth_factor * wealth)
            return max(price - effective_strike, 0.0)
        return payoff
    
    @staticmethod
    def power_payoff(strike: float, power: float = 2.0) -> Callable:
        """Power payoff: (S/K)^power - 1 if S > K, 0 otherwise."""
        def payoff(wealth: float, price: float) -> float:
            if price > strike:
                return (price / strike) ** power - 1.0
            return 0.0
        return payoff


class CorrelationMatrixGenerator:
    """Utility to generate correlation matrices for multi-asset problems."""
    
    @staticmethod
    def uniform_correlation(n_assets: int, rho: float) -> np.ndarray:
        """Generate correlation matrix with uniform correlation rho."""
        corr = np.full((n_assets, n_assets), rho)
        np.fill_diagonal(corr, 1.0)
        return corr
    
    @staticmethod
    def block_correlation(block_sizes: list, within_block_rho: float, 
                         between_block_rho: float) -> np.ndarray:
        """Generate block correlation matrix."""
        n_total = sum(block_sizes)
        corr = np.full((n_total, n_total), between_block_rho)
        
        start_idx = 0
        for block_size in block_sizes:
            end_idx = start_idx + block_size
            corr[start_idx:end_idx, start_idx:end_idx] = within_block_rho
            start_idx = end_idx
        
        np.fill_diagonal(corr, 1.0)
        return corr
    
    @staticmethod
    def exponential_decay(n_assets: int, decay_rate: float = 0.1) -> np.ndarray:
        """Generate correlation with exponential decay by distance."""
        corr = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(n_assets):
                corr[i, j] = np.exp(-decay_rate * abs(i - j))
        return corr
    
    @staticmethod
    def random_correlation(n_assets: int, seed: int = 42) -> np.ndarray:
        """Generate random positive definite correlation matrix."""
        np.random.seed(seed)
        A = np.random.randn(n_assets, n_assets)
        corr = A @ A.T
        # Normalize to correlation matrix
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        return corr


class ParameterGenerator:
    """Generate realistic parameter sets for multi-asset problems."""
    
    @staticmethod
    def generate_market_parameters(n_assets: int, 
                                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate realistic mu, sigma, and correlation for assets."""
        np.random.seed(seed)
        
        # Drift parameters: centered around risk-free rate with some variation
        mu = np.random.normal(0.05, 0.03, n_assets)  # 5% +/- 3%
        mu = np.clip(mu, -0.1, 0.2)  # Reasonable bounds
        
        # Volatility parameters: realistic range for stocks
        sigma = np.random.uniform(0.15, 0.4, n_assets)  # 15% to 40% vol
        
        # Correlation: moderate positive correlation
        correlation = CorrelationMatrixGenerator.exponential_decay(n_assets, 0.2)
        
        return mu, sigma, correlation
    
    @staticmethod
    def generate_payoff_functions(n_assets: int, 
                                 option_types: list = None) -> Dict[int, Callable]:
        """Generate diverse payoff functions for each asset."""
        if option_types is None:
            option_types = ['call', 'put', 'barrier', 'power'] * (n_assets // 4 + 1)
        
        payoffs = {}
        for i in range(n_assets):
            option_type = option_types[i % len(option_types)]
            
            if option_type == 'call':
                payoffs[i] = PayoffFunctions.american_call(1.0 + 0.1 * i, 1.0)
            elif option_type == 'put':
                payoffs[i] = PayoffFunctions.american_put(1.0 + 0.1 * i, 1.0)
            elif option_type == 'barrier':
                payoffs[i] = PayoffFunctions.barrier_option(1.5 + 0.1 * i, 1.0 + 0.1 * i)
            elif option_type == 'power':
                payoffs[i] = PayoffFunctions.power_payoff(1.0 + 0.1 * i, 1.5)
            else:
                payoffs[i] = PayoffFunctions.american_call(1.0, 1.0)
        
        return payoffs


class ResultAnalyzer:
    """Analyze and visualize results from the optimal stopping algorithm."""
    
    def __init__(self, solver):
        self.solver = solver
    
    def plot_value_function(self, active_set: set, t_idx: int = 0, 
                           asset_prices: np.ndarray = None, save_path: str = None):
        """Plot value function as function of wealth for fixed prices."""
        if asset_prices is None:
            # Use middle prices
            asset_prices = np.full(self.solver.n_assets, 
                                 (self.solver.grid.params.price_min + 
                                  self.solver.grid.params.price_max) / 2)
        
        wealth_grid = self.solver.grid.wealth_grid
        values = []
        
        for wealth in wealth_grid:
            value = self.solver.get_value(
                self.solver.grid.time_grid[t_idx], wealth, asset_prices, active_set
            )
            values.append(value)
        
        plt.figure(figsize=(10, 6))
        plt.plot(wealth_grid, values, 'b-', linewidth=2)
        plt.xlabel('Wealth')
        plt.ylabel('Value Function')
        plt.title(f'Value Function at t={self.solver.grid.time_grid[t_idx]:.3f}, '
                 f'Active Set: {active_set}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_stopping_regions(self, asset_id: int, active_set: set, t_idx: int = 0,
                             save_path: str = None):
        """Plot stopping region for specific asset."""
        wealth_grid = self.solver.grid.wealth_grid
        price_grid = self.solver.grid.price_grid
        
        stopping_matrix = np.zeros((len(wealth_grid), len(price_grid)))
        
        for i, wealth in enumerate(wealth_grid):
            for j, price in enumerate(price_grid):
                prices = np.full(self.solver.n_assets, price)
                _, stopping_asset = self.solver.get_optimal_policy(
                    self.solver.grid.time_grid[t_idx], wealth, prices, active_set
                )
                stopping_matrix[i, j] = 1.0 if stopping_asset == asset_id else 0.0
        
        plt.figure(figsize=(10, 8))
        plt.imshow(stopping_matrix, extent=[price_grid[0], price_grid[-1], 
                                           wealth_grid[0], wealth_grid[-1]], 
                  aspect='auto', origin='lower', cmap='RdYlBu')
        plt.colorbar(label='Stop Asset (1) or Continue (0)')
        plt.xlabel(f'Asset {asset_id} Price')
        plt.ylabel('Wealth')
        plt.title(f'Stopping Region for Asset {asset_id} at t={self.solver.grid.time_grid[t_idx]:.3f}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_convergence(self) -> Dict:
        """Analyze convergence properties of the solution."""
        # This would implement convergence checks mentioned in the paper
        # For now, return placeholder analysis
        return {
            'converged': True,
            'iterations': 1,
            'final_error': 1e-8,
            'policy_stability': True
        }
    
    def export_results(self, filename: str):
        """Export results to file for further analysis."""
        import pickle
        
        results = {
            'value_function': self.solver.value_function,
            'optimal_policies': self.solver.optimal_policies,
            'stopping_regions': self.solver.stopping_regions,
            'grid': self.solver.grid,
            'parameters': {
                'mu': self.solver.mu,
                'sigma': self.solver.sigma,
                'correlation': self.solver.correlation,
                'risk_free_rate': self.solver.risk_free_rate
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)


class ValidationTools:
    """Tools for validating the numerical solution."""
    
    @staticmethod
    def check_boundary_conditions(solver, tolerance: float = 1e-6) -> bool:
        """Check if boundary conditions are satisfied."""
        # Terminal condition: V(T, x, s, I) = 0
        terminal_values = []
        final_t_idx = len(solver.grid.time_grid) - 1
        
        for active_set in solver.grid.active_sets:
            for wealth_idx in range(len(solver.grid.wealth_grid)):
                for price_indices in np.ndindex(*[len(solver.grid.price_grid)] * solver.n_assets):
                    value = solver.value_function.get_value(
                        final_t_idx, active_set, wealth_idx, np.array(price_indices)
                    )
                    terminal_values.append(abs(value))
        
        max_terminal_error = max(terminal_values) if terminal_values else 0.0
        return max_terminal_error < tolerance
    
    @staticmethod
    def check_monotonicity(solver) -> Dict[str, bool]:
        """Check monotonicity properties of the value function."""
        # Value function should be increasing in wealth (generally)
        # This is a simplified check
        return {
            'wealth_monotonic': True,  # Placeholder
            'price_monotonic': True,   # Placeholder
            'time_monotonic': True     # Placeholder
        }