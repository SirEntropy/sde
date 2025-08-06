import numpy as np
from typing import Set, Tuple
from scipy.interpolate import RegularGridInterpolator
from .grid import StateGrid, ValueFunction
from .optimization import finite_difference_gradient


class ContinuationValueComputer:
    """Computes continuation values for the optimal stopping problem."""
    
    def __init__(self, grid: StateGrid, dt: float):
        self.grid = grid
        self.dt = dt
        
    def compute_continuation(self,
                           wealth: float,
                           prices: np.ndarray,
                           active_set: Set[int],
                           portfolio: np.ndarray,
                           mu: np.ndarray,
                           sigma: np.ndarray,
                           correlation: np.ndarray,
                           value_function: ValueFunction,
                           t_idx: int,
                           risk_free_rate: float = 0.02) -> float:
        """
        Compute continuation value V_cont for given state and portfolio.
        
        This implements the PDE: ∂V/∂t + L^π V = 0
        where L^π is the infinitesimal generator under portfolio π.
        """
        if t_idx >= len(self.grid.time_grid) - 1:
            return 0.0  # Terminal condition
        
        # Get current value function array for this active set
        active_key = tuple(sorted(active_set))
        if active_key not in value_function.values:
            return 0.0
        
        current_values = value_function.values[active_key][t_idx + 1]  # Next time step
        
        # Get grid indices
        wealth_idx, price_indices = self.grid.get_state_indices(wealth, prices)
        
        # Compute gradients using finite differences
        try:
            dV_dx, dV_ds, d2V_dx2, d2V_ds2, d2V_dxds = finite_difference_gradient(
                current_values, wealth_idx, price_indices,
                self.grid.wealth_grid, self.grid.price_grid
            )
        except:
            # Fallback to simple interpolation if gradient computation fails
            return self._simple_interpolation(current_values, wealth_idx, price_indices)
        
        # Compute drift terms
        wealth_drift = self._compute_wealth_drift(wealth, portfolio, mu, risk_free_rate)
        price_drifts = self._compute_price_drifts(prices, mu)
        
        # Compute diffusion terms
        wealth_volatility = self._compute_wealth_volatility(wealth, portfolio, sigma)
        price_volatilities = self._compute_price_volatilities(prices, sigma)
        cross_volatilities = self._compute_cross_volatilities(
            wealth, prices, portfolio, sigma, correlation
        )
        
        # Assemble the infinitesimal generator L^π V
        drift_term = wealth_drift * dV_dx + np.sum(price_drifts * dV_ds)
        
        volatility_term = (0.5 * wealth_volatility**2 * d2V_dx2 + 
                          0.5 * np.sum(price_volatilities**2 * d2V_ds2))
        
        cross_term = np.sum(cross_volatilities * d2V_dxds)
        
        # Total infinitesimal change
        infinitesimal_change = drift_term + volatility_term + cross_term
        
        # Current value plus expected change over dt
        current_value = self._interpolate_value(current_values, wealth_idx, price_indices)
        continuation_value = current_value + infinitesimal_change * self.dt
        
        return max(continuation_value, 0.0)  # Ensure non-negative
    
    def _compute_wealth_drift(self, wealth: float, portfolio: np.ndarray, 
                             mu: np.ndarray, risk_free_rate: float) -> float:
        """Compute drift of wealth process."""
        portfolio_return = np.sum(portfolio * mu)
        cash_weight = 1.0 - np.sum(portfolio)
        total_return = cash_weight * risk_free_rate + portfolio_return
        return wealth * total_return
    
    def _compute_price_drifts(self, prices: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Compute drift of price processes."""
        return prices * mu
    
    def _compute_wealth_volatility(self, wealth: float, portfolio: np.ndarray, 
                                  sigma: np.ndarray) -> float:
        """Compute volatility of wealth process."""
        portfolio_vol = np.sqrt(np.sum((portfolio * sigma)**2))
        return wealth * portfolio_vol
    
    def _compute_price_volatilities(self, prices: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Compute volatilities of price processes."""
        return prices * sigma
    
    def _compute_cross_volatilities(self, wealth: float, prices: np.ndarray,
                                   portfolio: np.ndarray, sigma: np.ndarray,
                                   correlation: np.ndarray) -> np.ndarray:
        """Compute cross-volatilities between wealth and prices."""
        n_assets = len(prices)
        cross_vols = np.zeros(n_assets)
        
        wealth_vol = wealth * np.sum(portfolio * sigma)
        
        for i in range(n_assets):
            price_vol = prices[i] * sigma[i]
            # Correlation between wealth and asset i
            wealth_asset_corr = np.sum(portfolio * sigma * correlation[i, :])
            cross_vols[i] = wealth_vol * price_vol * wealth_asset_corr / np.sum(portfolio * sigma)
        
        return cross_vols
    
    def _interpolate_value(self, value_array: np.ndarray, wealth_idx: int, 
                          price_indices: np.ndarray) -> float:
        """Interpolate value function at given indices."""
        try:
            # Simple linear interpolation as fallback
            return value_array[wealth_idx, tuple(price_indices)]
        except:
            return 0.0
    
    def _simple_interpolation(self, value_array: np.ndarray, wealth_idx: int,
                             price_indices: np.ndarray) -> float:
        """Simple interpolation when gradient computation fails."""
        try:
            return value_array[wealth_idx, tuple(price_indices)]
        except:
            return 0.0


class InterpolationHelper:
    """Helper class for more sophisticated interpolation methods."""
    
    def __init__(self, grid: StateGrid):
        self.grid = grid
        
    def create_interpolator(self, value_array: np.ndarray) -> RegularGridInterpolator:
        """Create scipy interpolator for value function."""
        points = [self.grid.wealth_grid] + [self.grid.price_grid] * self.grid.n_assets
        
        return RegularGridInterpolator(
            points, 
            value_array,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
    
    def interpolate_value(self, interpolator: RegularGridInterpolator,
                         wealth: float, prices: np.ndarray) -> float:
        """Interpolate value at continuous state."""
        point = np.concatenate([[wealth], prices])
        return float(interpolator(point))
    
    def interpolate_gradient(self, interpolator: RegularGridInterpolator,
                           wealth: float, prices: np.ndarray, 
                           h: float = 1e-6) -> Tuple[float, np.ndarray]:
        """Compute gradient using interpolator."""
        n_assets = len(prices)
        
        # Wealth gradient
        dV_dx = (interpolator([wealth + h] + list(prices)) - 
                 interpolator([wealth - h] + list(prices))) / (2 * h)
        
        # Price gradients
        dV_ds = np.zeros(n_assets)
        for i in range(n_assets):
            prices_plus = prices.copy()
            prices_minus = prices.copy()
            prices_plus[i] += h
            prices_minus[i] -= h
            
            dV_ds[i] = (interpolator([wealth] + list(prices_plus)) - 
                       interpolator([wealth] + list(prices_minus))) / (2 * h)
        
        return dV_dx, dV_ds