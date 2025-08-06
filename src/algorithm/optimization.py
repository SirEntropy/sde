import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from typing import Set, Callable, Tuple
from numba import jit


class PortfolioOptimizer:
    """Handles portfolio optimization for the multi-asset optimal stopping problem."""
    
    def __init__(self, n_assets: int, risk_free_rate: float = 0.02):
        self.n_assets = n_assets
        self.risk_free_rate = risk_free_rate
        
    def optimize_portfolio(self, 
                          wealth: float, 
                          prices: np.ndarray, 
                          active_set: Set[int],
                          mu: np.ndarray, 
                          sigma: np.ndarray, 
                          correlation: np.ndarray,
                          value_function_gradient: Callable,
                          dt: float) -> np.ndarray:
        """
        Optimize portfolio weights for given state.
        
        Solves: max_π L^π V where L^π is the infinitesimal generator.
        """
        if len(active_set) == 0:
            return np.zeros(self.n_assets)
        
        # Only optimize over active assets
        active_indices = sorted(list(active_set))
        n_active = len(active_indices)
        
        # Initial guess: equal weight in active assets
        x0 = np.zeros(n_active)
        if n_active > 0:
            x0[:] = 0.1 / n_active  # Conservative initial allocation
        
        # Bounds: each weight between 0 and 1
        bounds = [(0, 1) for _ in range(n_active)]
        
        # Constraint: sum of weights <= 1
        def constraint_sum(weights):
            return 1.0 - np.sum(weights)
        
        constraints = [{'type': 'ineq', 'fun': constraint_sum}]
        
        # Objective function: maximize expected infinitesimal change in value
        def objective(weights):
            return -self._compute_expected_return(
                weights, active_indices, wealth, prices, mu, sigma, 
                correlation, value_function_gradient, dt
            )
        
        try:
            result = minimize(
                objective, 
                x0, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-8, 'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = np.zeros(self.n_assets)
                optimal_weights[active_indices] = result.x
                return optimal_weights
            else:
                # Fallback: return conservative equal weights
                fallback_weights = np.zeros(self.n_assets)
                if n_active > 0:
                    weight_per_asset = 0.1 / n_active
                    fallback_weights[active_indices] = weight_per_asset
                return fallback_weights
                
        except Exception as e:
            # Emergency fallback
            fallback_weights = np.zeros(self.n_assets)
            return fallback_weights
    
    def _compute_expected_return(self, 
                                weights: np.ndarray, 
                                active_indices: list,
                                wealth: float, 
                                prices: np.ndarray, 
                                mu: np.ndarray, 
                                sigma: np.ndarray,
                                correlation: np.ndarray,
                                value_function_gradient: Callable,
                                dt: float) -> float:
        """
        Compute expected instantaneous return from portfolio.
        
        This implements the infinitesimal generator L^π V.
        """
        try:
            # Get gradients of value function
            dV_dx, dV_ds, d2V_dx2, d2V_ds2, d2V_dxds = value_function_gradient(wealth, prices)
            
            # Full portfolio weights (including inactive assets with 0 weight)
            full_weights = np.zeros(self.n_assets)
            full_weights[active_indices] = weights
            
            # Portfolio return and volatility
            portfolio_mu = np.sum(full_weights * mu)
            
            # Wealth dynamics under portfolio π
            wealth_drift = wealth * (self.risk_free_rate + portfolio_mu)
            
            # Expected return from wealth changes
            expected_return = wealth_drift * dV_dx
            
            # Second-order terms (volatility effects)
            # Portfolio volatility matrix
            portfolio_vol = np.zeros((self.n_assets, self.n_assets))
            for i in range(self.n_assets):
                for j in range(self.n_assets):
                    portfolio_vol[i, j] = (full_weights[i] * sigma[i] * prices[i] * 
                                         full_weights[j] * sigma[j] * prices[j] * 
                                         correlation[i, j])
            
            # Wealth volatility
            wealth_vol = wealth * np.sum(full_weights * sigma)
            volatility_term = 0.5 * (wealth_vol ** 2) * d2V_dx2
            
            # Cross terms between wealth and prices
            cross_terms = 0.0
            for i in range(self.n_assets):
                if i in active_indices:
                    cross_terms += (wealth * full_weights[i] * sigma[i] * 
                                  prices[i] * sigma[i] * d2V_dxds[i])
            
            # Price volatility terms
            price_volatility_terms = 0.0
            for i in range(self.n_assets):
                price_vol = (sigma[i] * prices[i]) ** 2
                price_volatility_terms += 0.5 * price_vol * d2V_ds2[i]
            
            total_return = (expected_return + volatility_term + 
                          cross_terms + price_volatility_terms) * dt
            
            return total_return
            
        except Exception as e:
            # Return conservative estimate if computation fails
            return 0.0


@jit(nopython=True)
def finite_difference_gradient(value_array: np.ndarray, 
                             wealth_idx: int, 
                             price_indices: np.ndarray,
                             wealth_grid: np.ndarray,
                             price_grid: np.ndarray,
                             h: float = 1e-4) -> Tuple[float, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Compute finite difference gradients of value function.
    
    Returns: dV/dx, dV/ds, d²V/dx², d²V/ds², d²V/dxds
    """
    n_assets = len(price_indices)
    n_wealth = len(wealth_grid)
    n_price = len(price_grid)
    
    # First derivatives
    dV_dx = 0.0
    dV_ds = np.zeros(n_assets)
    
    # Second derivatives
    d2V_dx2 = 0.0
    d2V_ds2 = np.zeros(n_assets)
    d2V_dxds = np.zeros(n_assets)
    
    # Wealth derivatives
    if 0 < wealth_idx < n_wealth - 1:
        # Central difference
        dx = wealth_grid[wealth_idx + 1] - wealth_grid[wealth_idx - 1]
        v_plus = value_array[wealth_idx + 1, tuple(price_indices)]
        v_minus = value_array[wealth_idx - 1, tuple(price_indices)]
        dV_dx = (v_plus - v_minus) / dx
        
        # Second derivative
        v_center = value_array[wealth_idx, tuple(price_indices)]
        d2V_dx2 = (v_plus - 2 * v_center + v_minus) / ((dx / 2) ** 2)
    
    elif wealth_idx == 0:
        # Forward difference
        dx = wealth_grid[1] - wealth_grid[0]
        v_plus = value_array[1, tuple(price_indices)]
        v_center = value_array[0, tuple(price_indices)]
        dV_dx = (v_plus - v_center) / dx
        
    elif wealth_idx == n_wealth - 1:
        # Backward difference
        dx = wealth_grid[n_wealth - 1] - wealth_grid[n_wealth - 2]
        v_center = value_array[n_wealth - 1, tuple(price_indices)]
        v_minus = value_array[n_wealth - 2, tuple(price_indices)]
        dV_dx = (v_center - v_minus) / dx
    
    # Price derivatives for each asset
    for i in range(n_assets):
        price_idx = price_indices[i]
        
        if 0 < price_idx < n_price - 1:
            # Central difference
            ds = price_grid[price_idx + 1] - price_grid[price_idx - 1]
            
            # Create index arrays for neighboring points
            price_plus = price_indices.copy()
            price_minus = price_indices.copy()
            price_plus[i] = price_idx + 1
            price_minus[i] = price_idx - 1
            
            v_plus = value_array[wealth_idx, tuple(price_plus)]
            v_minus = value_array[wealth_idx, tuple(price_minus)]
            dV_ds[i] = (v_plus - v_minus) / ds
            
            # Second derivative
            v_center = value_array[wealth_idx, tuple(price_indices)]
            d2V_ds2[i] = (v_plus - 2 * v_center + v_minus) / ((ds / 2) ** 2)
            
        elif price_idx == 0:
            # Forward difference
            ds = price_grid[1] - price_grid[0]
            price_plus = price_indices.copy()
            price_plus[i] = 1
            
            v_plus = value_array[wealth_idx, tuple(price_plus)]
            v_center = value_array[wealth_idx, tuple(price_indices)]
            dV_ds[i] = (v_plus - v_center) / ds
            
        elif price_idx == n_price - 1:
            # Backward difference
            ds = price_grid[n_price - 1] - price_grid[n_price - 2]
            price_minus = price_indices.copy()
            price_minus[i] = n_price - 2
            
            v_center = value_array[wealth_idx, tuple(price_indices)]
            v_minus = value_array[wealth_idx, tuple(price_minus)]
            dV_ds[i] = (v_center - v_minus) / ds
    
    # Cross derivatives d²V/dxds
    for i in range(n_assets):
        price_idx = price_indices[i]
        
        if (0 < wealth_idx < n_wealth - 1 and 0 < price_idx < n_price - 1):
            # Mixed partial derivative using finite differences
            dx = wealth_grid[wealth_idx + 1] - wealth_grid[wealth_idx - 1]
            ds = price_grid[price_idx + 1] - price_grid[price_idx - 1]
            
            price_plus = price_indices.copy()
            price_minus = price_indices.copy()
            price_plus[i] = price_idx + 1
            price_minus[i] = price_idx - 1
            
            v_pp = value_array[wealth_idx + 1, tuple(price_plus)]
            v_pm = value_array[wealth_idx + 1, tuple(price_minus)]
            v_mp = value_array[wealth_idx - 1, tuple(price_plus)]
            v_mm = value_array[wealth_idx - 1, tuple(price_minus)]
            
            d2V_dxds[i] = (v_pp - v_pm - v_mp + v_mm) / (dx * ds)
    
    return dV_dx, dV_ds, d2V_dx2, d2V_ds2, d2V_dxds