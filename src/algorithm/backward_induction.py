import numpy as np
from typing import Set, Dict, Tuple, Callable, Optional
import logging
from concurrent.futures import ProcessPoolExecutor
from .grid import StateGrid, ValueFunction, GridParameters
from .optimization import PortfolioOptimizer
from .continuation import ContinuationValueComputer


class BackwardInductionSolver:
    """Implements the backward induction algorithm from the LaTeX document."""
    
    def __init__(self, 
                 n_assets: int,
                 mu: np.ndarray,
                 sigma: np.ndarray,
                 correlation: np.ndarray,
                 payoff_functions: Dict[int, Callable],
                 grid_params: Optional[GridParameters] = None,
                 risk_free_rate: float = 0.02,
                 max_time: float = 1.0):
        """
        Initialize the backward induction solver.
        
        Args:
            n_assets: Number of risky assets
            mu: Drift parameters for each asset
            sigma: Volatility parameters for each asset  
            correlation: Correlation matrix between assets
            payoff_functions: Dict mapping asset index to payoff function G_i(x, s_i)
            grid_params: Grid discretization parameters
            risk_free_rate: Risk-free interest rate
            max_time: Maximum time horizon
        """
        self.n_assets = n_assets
        self.mu = mu
        self.sigma = sigma
        self.correlation = correlation
        self.payoff_functions = payoff_functions
        self.risk_free_rate = risk_free_rate
        self.max_time = max_time
        
        # Initialize grid
        self.grid_params = grid_params or GridParameters()
        self.grid = StateGrid(n_assets, self.grid_params)
        self.dt = self.max_time / self.grid_params.n_time_steps
        
        # Initialize components
        self.optimizer = PortfolioOptimizer(n_assets, risk_free_rate)
        self.continuation_computer = ContinuationValueComputer(self.grid, self.dt)
        
        # Results storage
        self.value_function: Optional[ValueFunction] = None
        self.optimal_policies: Dict = {}
        self.stopping_regions: Dict = {}
        
        # Convergence parameters
        self.tolerance = 1e-6
        self.max_iterations = 10
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def solve(self, parallel: bool = False) -> Tuple[ValueFunction, Dict, Dict]:
        """
        Main solving routine implementing Algorithm 1 from the paper.
        
        Returns:
            value_function: The computed value function
            optimal_policies: Optimal portfolio policies
            stopping_regions: Optimal stopping regions
        """
        self.logger.info("Starting backward induction algorithm")
        self.logger.info(f"Grid size: {self.grid_params.n_time_steps} × "
                        f"{self.grid_params.n_wealth_points} × "
                        f"{self.grid_params.n_price_points}^{self.n_assets}")
        
        # Initialize value function
        self.value_function = ValueFunction(self.grid)
        
        # Backward induction loop (Algorithm 1, lines 7-88)
        import time
        start_time = time.time()
        total_time_steps = len(self.grid.time_grid) - 1
        
        for step_counter, t_idx in enumerate(reversed(range(len(self.grid.time_grid) - 1))):
            current_time = self.grid.time_grid[t_idx]
            progress = ((step_counter + 1) / total_time_steps) * 100
            
            # Time estimation
            if step_counter > 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed / step_counter * total_time_steps
                remaining = estimated_total - elapsed
                self.logger.info(f"Processing time step {t_idx}/{len(self.grid.time_grid)-1} "
                               f"(t={current_time:.3f}) - Progress: {progress:.1f}% "
                               f"- ETA: {remaining/60:.1f} minutes")
            else:
                self.logger.info(f"Processing time step {t_idx}/{len(self.grid.time_grid)-1} "
                               f"(t={current_time:.3f}) - Progress: {progress:.1f}%")
            
            step_start_time = time.time()
            if parallel:
                self._solve_time_step_parallel(t_idx)
            else:
                self._solve_time_step_sequential(t_idx)
            
            step_duration = time.time() - step_start_time
            self.logger.info(f"Completed time step {t_idx} in {step_duration:.1f} seconds")
        
        self.logger.info("Backward induction completed")
        return self.value_function, self.optimal_policies, self.stopping_regions
    
    def _solve_time_step_sequential(self, t_idx: int):
        """Solve single time step sequentially over all active sets and states."""
        total_active_sets = len([s for s in self.grid.active_sets if len(s) > 0])
        total_wealth_points = len(self.grid.wealth_grid)
        total_price_combinations = len(self.grid.price_grid) ** self.n_assets
        
        self.logger.info(f"Time step {t_idx}: Processing {total_active_sets} active sets, "
                        f"{total_wealth_points} wealth points, {total_price_combinations} price combinations")
        
        processed_sets = 0
        # Loop over all active sets (Algorithm 1, line 8)
        for active_set_idx, active_set in enumerate(self.grid.active_sets):
            if len(active_set) == 0:
                continue
            
            processed_sets += 1
            self.logger.info(f"  Active set {processed_sets}/{total_active_sets}: {sorted(list(active_set))}")
            
            processed_states = 0
            total_states_this_set = total_wealth_points * total_price_combinations
            
            # Loop over all states (Algorithm 1, line 9)
            for wealth_idx in range(len(self.grid.wealth_grid)):
                if wealth_idx % 5 == 0:  # Log every 5th wealth point
                    self.logger.info(f"    Wealth point {wealth_idx+1}/{total_wealth_points}")
                
                for price_indices in np.ndindex(*[len(self.grid.price_grid)] * self.n_assets):
                    price_indices = np.array(price_indices)
                    processed_states += 1
                    
                    if processed_states % 1000 == 0:  # Log every 1000 states
                        progress = (processed_states / total_states_this_set) * 100
                        self.logger.info(f"    Progress: {processed_states}/{total_states_this_set} states ({progress:.1f}%)")
                    
                    self._solve_single_state(t_idx, active_set, wealth_idx, price_indices)
            
            self.logger.info(f"  Completed active set {processed_sets}/{total_active_sets}")
    
    def _solve_time_step_parallel(self, t_idx: int):
        """Solve single time step with parallel processing over active sets."""
        with ProcessPoolExecutor() as executor:
            futures = []
            
            for active_set in self.grid.active_sets:
                if len(active_set) > 0:
                    future = executor.submit(self._solve_active_set, t_idx, active_set)
                    futures.append(future)
            
            # Wait for all to complete
            for future in futures:
                future.result()
    
    def _solve_active_set(self, t_idx: int, active_set: Set[int]):
        """Solve all states for a given active set."""
        for wealth_idx in range(len(self.grid.wealth_grid)):
            for price_indices in np.ndindex(*[len(self.grid.price_grid)] * self.n_assets):
                price_indices = np.array(price_indices)
                self._solve_single_state(t_idx, active_set, wealth_idx, price_indices)
    
    def _solve_single_state(self, t_idx: int, active_set: Set[int], 
                           wealth_idx: int, price_indices: np.ndarray):
        """
        Solve the HJB equation at a single state point.
        
        Implements Algorithm 1, lines 71-85.
        """
        try:
            # Get current state values
            wealth, prices = self.grid.get_state_values(wealth_idx, price_indices)
            
            # Step 1: Solve portfolio optimization (Algorithm 1, line 72)
            optimal_portfolio = self._optimize_portfolio(wealth, prices, active_set, t_idx)
            
            # Step 2: Compute continuation value (Algorithm 1, line 76)
            continuation_value = self._compute_continuation_value(
                wealth, prices, active_set, optimal_portfolio, t_idx
            )
            
            # Step 3: Compute stopping values (Algorithm 1, lines 80-82)
            stopping_values = self._compute_stopping_values(
                wealth, prices, active_set, t_idx
            )
        except Exception as e:
            self.logger.error(f"Error at state (t={t_idx}, W={wealth_idx}, P={price_indices}, A={active_set}): {e}")
            # Set default values to continue
            optimal_portfolio = np.zeros(self.n_assets)
            continuation_value = 0.0
            stopping_values = {}
        
        # Step 4: Optimal decision (Algorithm 1, line 85)
        max_stopping_value = max(stopping_values.values()) if stopping_values else 0.0
        
        # Ensure scalar values for comparison
        if isinstance(continuation_value, np.ndarray):
            if continuation_value.size == 1:
                continuation_value = float(continuation_value.item())
            else:
                continuation_value = float(np.mean(continuation_value))  # Fallback for multi-element arrays
        elif continuation_value is None:
            continuation_value = 0.0
        else:
            continuation_value = float(continuation_value)
            
        if isinstance(max_stopping_value, np.ndarray):
            if max_stopping_value.size == 1:
                max_stopping_value = float(max_stopping_value.item())
            else:
                max_stopping_value = float(np.mean(max_stopping_value))  # Fallback for multi-element arrays
        elif max_stopping_value is None:
            max_stopping_value = 0.0
        else:
            max_stopping_value = float(max_stopping_value)
            
        optimal_value = max(continuation_value, max_stopping_value)
        
        # Store results
        self.value_function.set_value(t_idx, active_set, wealth_idx, price_indices, optimal_value)
        self.value_function.set_portfolio(t_idx, active_set, wealth_idx, price_indices, optimal_portfolio)
        
        # Store stopping decision
        best_stopping_asset = None
        if max_stopping_value > continuation_value:
            best_stopping_asset = max(stopping_values.keys(), key=stopping_values.get)
        
        # Store in results dictionaries
        state_key = (t_idx, tuple(sorted(active_set)), wealth_idx, tuple(price_indices))
        self.optimal_policies[state_key] = optimal_portfolio
        self.stopping_regions[state_key] = best_stopping_asset
    
    def _optimize_portfolio(self, wealth: float, prices: np.ndarray, 
                           active_set: Set[int], t_idx: int) -> np.ndarray:
        """Step 1: Portfolio optimization using scipy.optimize.minimize."""
        
        def value_gradient(w, p):
            """Gradient function for portfolio optimization."""
            # Get next time step value function
            if t_idx + 1 >= len(self.grid.time_grid):
                return 0.0, np.zeros(self.n_assets), 0.0, np.zeros(self.n_assets), np.zeros(self.n_assets)
            
            w_idx, p_idx = self.grid.get_state_indices(w, p)
            active_key = tuple(sorted(active_set))
            
            if active_key in self.value_function.values:
                next_values = self.value_function.values[active_key][t_idx + 1]
                try:
                    from .optimization import finite_difference_gradient
                    return finite_difference_gradient(
                        next_values, w_idx, p_idx,
                        self.grid.wealth_grid, self.grid.price_grid
                    )
                except:
                    return 0.0, np.zeros(self.n_assets), 0.0, np.zeros(self.n_assets), np.zeros(self.n_assets)
            
            return 0.0, np.zeros(self.n_assets), 0.0, np.zeros(self.n_assets), np.zeros(self.n_assets)
        
        try:
            portfolio = self.optimizer.optimize_portfolio(
                wealth, prices, active_set, self.mu, self.sigma, 
                self.correlation, value_gradient, self.dt
            )
            return portfolio
        except Exception as e:
            # Log the problematic optimization and return zero portfolio
            self.logger.warning(f"Portfolio optimization failed at W={wealth:.3f}, P={prices}, A={active_set}: {e}")
            return np.zeros(self.n_assets)
    
    def _compute_continuation_value(self, wealth: float, prices: np.ndarray,
                                   active_set: Set[int], portfolio: np.ndarray,
                                   t_idx: int) -> float:
        """Step 2: Compute continuation value using finite differences."""
        return self.continuation_computer.compute_continuation(
            wealth, prices, active_set, portfolio, self.mu, self.sigma,
            self.correlation, self.value_function, t_idx, self.risk_free_rate
        )
    
    def _compute_stopping_values(self, wealth: float, prices: np.ndarray,
                                active_set: Set[int], t_idx: int) -> Dict[int, float]:
        """Step 3: Compute stopping values for each asset in active set."""
        stopping_values = {}
        
        for asset_i in active_set:
            # Compute payoff G_i(x, s_i)
            if asset_i in self.payoff_functions:
                payoff = self.payoff_functions[asset_i](wealth, prices[asset_i])
            else:
                # Default payoff: some function of wealth and price
                payoff = wealth * max(0, prices[asset_i] - 1.0)  # Call option style
            
            # Compute value after stopping asset i: V(t, x, s, I\{i})
            remaining_set = active_set - {asset_i}
            
            if len(remaining_set) > 0:
                remaining_value = self._get_value_after_stopping(
                    wealth, prices, remaining_set, t_idx
                )
            else:
                remaining_value = 0.0  # No assets left
            
            stopping_values[asset_i] = payoff + remaining_value
        
        return stopping_values
    
    def _get_value_after_stopping(self, wealth: float, prices: np.ndarray,
                                 remaining_set: Set[int], t_idx: int) -> float:
        """Get value function after removing an asset from active set."""
        if len(remaining_set) == 0:
            return 0.0
        
        wealth_idx, price_indices = self.grid.get_state_indices(wealth, prices)
        return self.value_function.get_value(t_idx, remaining_set, wealth_idx, price_indices)
    
    def get_optimal_policy(self, t: float, wealth: float, prices: np.ndarray, 
                          active_set: Set[int]) -> Tuple[np.ndarray, Optional[int]]:
        """
        Get optimal policy (portfolio and stopping decision) for given state.
        
        Returns:
            portfolio: Optimal portfolio weights
            stopping_asset: Asset to stop (None to continue)
        """
        if self.value_function is None:
            raise RuntimeError("Must call solve() first")
        
        # Find nearest time index
        t_idx = np.argmin(np.abs(self.grid.time_grid - t))
        wealth_idx, price_indices = self.grid.get_state_indices(wealth, prices)
        
        state_key = (t_idx, tuple(sorted(active_set)), wealth_idx, tuple(price_indices))
        
        portfolio = self.optimal_policies.get(state_key, np.zeros(self.n_assets))
        stopping_asset = self.stopping_regions.get(state_key, None)
        
        return portfolio, stopping_asset
    
    def get_value(self, t: float, wealth: float, prices: np.ndarray, 
                  active_set: Set[int]) -> float:
        """Get value function at given state."""
        if self.value_function is None:
            raise RuntimeError("Must call solve() first")
        
        t_idx = np.argmin(np.abs(self.grid.time_grid - t))
        wealth_idx, price_indices = self.grid.get_state_indices(wealth, prices)
        
        return self.value_function.get_value(t_idx, active_set, wealth_idx, price_indices)