# Multi-Asset Optimal Stopping with Budget Constraints

A Python implementation of multi-asset optimal stopping algorithms with budget constraints, using backward induction with policy iteration to solve the Hamilton-Jacobi-Bellman equation.

## 🚀 Features

- **Complete Mathematical Framework**: Implements the algorithm described in the accompanying LaTeX document
- **Real-World Applications**: American options pricing using live market data from Yahoo Finance
- **High Performance**: Optimized with NumPy, SciPy, and Numba for fast computation
- **Flexible Architecture**: Modular design supporting various payoff functions and market scenarios
- **Comprehensive Analysis**: Built-in visualization and Monte Carlo simulation capabilities

## 📁 Project Structure

```
├── src/                          # Main package
│   ├── algorithm/               # Mathematical core algorithms
│   │   ├── grid.py             # State space discretization
│   │   ├── optimization.py     # Portfolio optimization
│   │   ├── continuation.py     # Continuation value computation
│   │   ├── backward_induction.py # Main algorithm implementation
│   │   └── utils.py            # Mathematical utilities
│   ├── solver.py               # High-level solver interface
├── examples/                    # Real-world examples
│   └── options.py              # American options with yfinance data
├── latex/                      # Mathematical documentation
│   └── pseudo_code.tex         # Algorithm specification
└── requirements.txt            # Package dependencies
```

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sde
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

### Basic Usage

```python
from src import MultiAssetOptimalStoppingSolver, SolverConfig
from src.algorithm import PayoffFunctions
import numpy as np

# Configure solver
config = SolverConfig(n_assets=2, max_time=0.25)
solver = MultiAssetOptimalStoppingSolver(config)

# Set market parameters
mu = np.array([0.05, 0.03])          # Drift rates
sigma = np.array([0.2, 0.15])        # Volatilities  
correlation = np.array([[1.0, 0.3], [0.3, 1.0]])
solver.set_market_parameters(mu, sigma, correlation)

# Define payoff functions (American options)
payoffs = {
    0: PayoffFunctions.american_call(strike=1.0),
    1: PayoffFunctions.american_put(strike=1.0)
}
solver.set_payoff_functions(payoffs)

# Solve the optimal stopping problem
results = solver.solve()

# Query the solution
value = solver.get_value(t=0.1, wealth=1.0, prices=[1.1, 0.9], active_set={0, 1})
portfolio, stop_asset = solver.get_optimal_policy(t=0.1, wealth=1.0, prices=[1.1, 0.9], active_set={0, 1})

print(f"Value: {value:.4f}")
print(f"Optimal portfolio: {portfolio}")
print(f"Stop asset: {stop_asset}")
```

### Real-World Example with Market Data

```bash
# Basic run
python examples/options.py

# Generate comprehensive reports
python examples/options.py --reports

# Run multiple asset scenarios
python examples/options.py --multiple
```

This example:
- Fetches real market data for popular stocks (AAPL, GOOGL, MSFT)
- Estimates market parameters from historical data
- Solves optimal stopping for American call/put options
- Runs Monte Carlo simulations
- Generates visualizations and analysis

## 📊 Algorithm Overview

The implementation follows the mathematical framework:

1. **Problem Setup**: Multi-asset price dynamics with correlation
   ```
   dS_i(t) = μ_i S_i(t) dt + σ_i S_i(t) dW_i(t)
   ```

2. **State Variables**:
   - Wealth: X(t)
   - Asset prices: S(t) = (S₁(t), ..., Sₙ(t))
   - Active set: I(t) ⊆ {1, ..., N}

3. **HJB Equation**:
   ```
   ∂V/∂t + sup_π L^π V + max_i [G_i(x,s_i) - V(t,x,s,I\{i})]⁺ = 0
   ```

4. **Backward Induction Algorithm**:
   - Grid-based discretization
   - Portfolio optimization at each state
   - Continuation value computation via finite differences
   - Optimal stopping decision

## 🔧 Configuration Options

### Grid Parameters
```python
from src.algorithm import GridParameters

grid_params = GridParameters(
    n_time_steps=50,      # Time discretization
    n_wealth_points=30,   # Wealth grid points
    n_price_points=25,    # Price grid points per asset
    wealth_min=0.1,       # Minimum wealth
    wealth_max=5.0,       # Maximum wealth
    price_min=0.5,        # Minimum asset price
    price_max=2.0         # Maximum asset price
)
```

### Solver Configuration
```python
config = SolverConfig(
    n_assets=3,                    # Number of assets
    max_time=1.0,                 # Time horizon (years)
    risk_free_rate=0.05,          # Risk-free interest rate
    grid_params=grid_params,      # Grid configuration
    use_parallel=True,            # Enable parallel processing
    tolerance=1e-6                # Convergence tolerance
)
```

### Payoff Functions
```python
# Available payoff types
payoffs = {
    0: PayoffFunctions.american_call(strike=100, multiplier=1.0),
    1: PayoffFunctions.american_put(strike=100, multiplier=1.0),
    2: PayoffFunctions.barrier_option(barrier=110, strike=100),
    3: PayoffFunctions.power_payoff(strike=100, power=1.5),
    4: PayoffFunctions.wealth_dependent_payoff(base_strike=100, wealth_factor=0.1)
}
```

## 📈 Analysis and Visualization

The solver provides comprehensive analysis tools:

```python
# Get solution summary
summary = solver.get_summary()
print(f"Solve time: {summary['solve_time']:.2f}s")
print(f"Max value: {summary['max_value']:.4f}")

# Run Monte Carlo simulation
simulation = solver.simulate_path(
    initial_wealth=1.0,
    initial_prices=np.array([1.0, 1.0, 1.0]),
    n_steps=100
)

# Generate plots
solver.plot_results(active_set={0, 1, 2})

# Export results for further analysis
solver.export_results('results.pkl')
```

## 🧪 Testing and Validation

The implementation includes built-in validation:

- **Boundary conditions**: Terminal condition V(T,x,s,I) = 0
- **Monotonicity checks**: Value function properties
- **Convergence criteria**: ||V^(k+1) - V^(k)||_∞ < ε
- **Policy stability**: Portfolio convergence

## ⚡ Performance Tips

1. **Grid Size**: Start with smaller grids for testing, increase for production
2. **Parallel Processing**: Enable for problems with many assets
3. **NumPy/Numba**: Ensure proper installation for optimized performance
4. **Memory Management**: Monitor memory usage for large state spaces

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
- Check the examples in `examples/`
- Review the mathematical specification in `latex/`
- Open an issue on GitHub

---

**Note**: This implementation is for research and educational purposes. For production trading systems, additional considerations for numerical stability, market microstructure, and regulatory requirements may be necessary.
