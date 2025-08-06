import numpy as np
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass
from itertools import combinations


@dataclass
class GridParameters:
    """Configuration for computational grids."""
    n_time_steps: int = 100
    n_wealth_points: int = 50
    n_price_points: int = 30
    wealth_min: float = 0.1
    wealth_max: float = 10.0
    price_min: float = 0.5
    price_max: float = 2.0


class StateGrid:
    """Manages discretized state space for the optimal stopping problem."""
    
    def __init__(self, n_assets: int, params: GridParameters):
        self.n_assets = n_assets
        self.params = params
        
        # Time grid
        self.time_grid = np.linspace(0, 1, params.n_time_steps)
        self.dt = self.time_grid[1] - self.time_grid[0]
        
        # Wealth grid (log-spaced for better resolution near 0)
        self.wealth_grid = np.logspace(
            np.log10(params.wealth_min), 
            np.log10(params.wealth_max), 
            params.n_wealth_points
        )
        
        # Price grid for each asset
        self.price_grid = np.linspace(
            params.price_min, 
            params.price_max, 
            params.n_price_points
        )
        
        # Generate all possible active sets
        self.active_sets = self._generate_active_sets()
        
    def _generate_active_sets(self) -> List[Set[int]]:
        """Generate all possible active sets (subsets of {1, ..., N})."""
        assets = set(range(self.n_assets))
        active_sets = []
        
        # Include all non-empty subsets
        for r in range(1, self.n_assets + 1):
            for subset in combinations(assets, r):
                active_sets.append(set(subset))
                
        return active_sets
    
    def get_state_indices(self, wealth: float, prices: np.ndarray) -> Tuple[int, np.ndarray]:
        """Find nearest grid indices for given state."""
        wealth_idx = np.argmin(np.abs(self.wealth_grid - wealth))
        price_indices = np.array([
            np.argmin(np.abs(self.price_grid - price)) 
            for price in prices
        ])
        return wealth_idx, price_indices
    
    def get_state_values(self, wealth_idx: int, price_indices: np.ndarray) -> Tuple[float, np.ndarray]:
        """Get actual state values from grid indices."""
        wealth = self.wealth_grid[wealth_idx]
        prices = np.array([self.price_grid[idx] for idx in price_indices])
        return wealth, prices
    
    def is_valid_state(self, wealth_idx: int, price_indices: np.ndarray) -> bool:
        """Check if state indices are within grid bounds."""
        if not (0 <= wealth_idx < len(self.wealth_grid)):
            return False
        if not all(0 <= idx < len(self.price_grid) for idx in price_indices):
            return False
        return True


class ValueFunction:
    """Stores and manages the value function over the state space."""
    
    def __init__(self, grid: StateGrid):
        self.grid = grid
        self.n_time = len(grid.time_grid)
        self.n_wealth = len(grid.wealth_grid)
        self.n_price = len(grid.price_grid)
        self.n_assets = grid.n_assets
        
        # Value function: V[t][active_set_idx][wealth_idx][price_indices]
        # Using dictionary for active sets to handle variable-sized tuples
        self.values = {}
        self.portfolios = {}  # Store optimal portfolios
        self.stopping_regions = {}  # Store stopping decisions
        
        self._initialize()
    
    def _initialize(self):
        """Initialize value function with terminal conditions."""
        for active_set in self.grid.active_sets:
            active_key = tuple(sorted(active_set))
            
            # Initialize value function array
            shape = (self.n_time, self.n_wealth) + (self.n_price,) * self.n_assets
            self.values[active_key] = np.zeros(shape)
            self.portfolios[active_key] = np.zeros(shape + (self.n_assets,))
            self.stopping_regions[active_key] = np.zeros(shape, dtype=int)
    
    def get_value(self, t_idx: int, active_set: Set[int], 
                  wealth_idx: int, price_indices: np.ndarray) -> float:
        """Get value function at specific state."""
        active_key = tuple(sorted(active_set))
        if active_key not in self.values:
            return 0.0
        
        index = (t_idx, wealth_idx) + tuple(price_indices)
        return self.values[active_key][index]
    
    def set_value(self, t_idx: int, active_set: Set[int], 
                  wealth_idx: int, price_indices: np.ndarray, value: float):
        """Set value function at specific state."""
        active_key = tuple(sorted(active_set))
        if active_key in self.values:
            index = (t_idx, wealth_idx) + tuple(price_indices)
            self.values[active_key][index] = value
    
    def get_portfolio(self, t_idx: int, active_set: Set[int], 
                     wealth_idx: int, price_indices: np.ndarray) -> np.ndarray:
        """Get optimal portfolio at specific state."""
        active_key = tuple(sorted(active_set))
        if active_key not in self.portfolios:
            return np.zeros(self.n_assets)
        
        index = (t_idx, wealth_idx) + tuple(price_indices)
        return self.portfolios[active_key][index]
    
    def set_portfolio(self, t_idx: int, active_set: Set[int], 
                     wealth_idx: int, price_indices: np.ndarray, portfolio: np.ndarray):
        """Set optimal portfolio at specific state."""
        active_key = tuple(sorted(active_set))
        if active_key in self.portfolios:
            index = (t_idx, wealth_idx) + tuple(price_indices)
            self.portfolios[active_key][index] = portfolio