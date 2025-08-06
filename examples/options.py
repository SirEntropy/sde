"""
Real-World American Options Example using yfinance Data

This example demonstrates the multi-asset optimal stopping algorithm using
real market data for popular stocks. It fetches historical price data,
estimates parameters, and computes optimal stopping strategies for 
American-style options.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
import json
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

# Suppress yfinance progress bars and warnings
import logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Import our solver
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import MultiAssetOptimalStoppingSolver, SolverConfig
from src.algorithm import PayoffFunctions, GridParameters, CorrelationMatrixGenerator


class MarketDataProcessor:
    """Process real market data for the optimal stopping algorithm."""
    
    def __init__(self, tickers: List[str], period: str = "2y"):
        """
        Initialize with list of stock tickers.
        
        Args:
            tickers: List of stock symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
            period: Data period ('1y', '2y', '5y', 'max')
        """
        self.tickers = tickers
        self.period = period
        self.data = None
        self.returns = None
        self.parameters = {}
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical price data from Yahoo Finance."""
        print(f"Fetching data for {self.tickers} over {self.period}...")
        
        try:
            # Download data for all tickers with progress disabled to avoid noise
            data = yf.download(self.tickers, period=self.period, interval='1d', 
                             progress=False)
            
            if data.empty:
                raise ValueError("No data downloaded from Yahoo Finance")
            
            # Handle different column structures based on number of tickers
            if len(self.tickers) == 1:
                # Single ticker: columns are simple like ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                if 'Adj Close' in data.columns:
                    adj_close = data[['Adj Close']]
                    adj_close.columns = self.tickers
                else:
                    # Fallback to Close if Adj Close not available
                    adj_close = data[['Close']]
                    adj_close.columns = self.tickers
            else:
                # Multiple tickers: columns are MultiIndex like ('Adj Close', 'AAPL'), ('Adj Close', 'GOOGL'), etc.
                if 'Adj Close' in data.columns.levels[0]:
                    adj_close = data['Adj Close']
                elif 'Close' in data.columns.levels[0]:
                    # Fallback to Close if Adj Close not available
                    adj_close = data['Close']
                else:
                    raise ValueError("No price data (Close/Adj Close) found in downloaded data")
            
            # Store data
            self.data = adj_close.dropna()
            
            if self.data.empty:
                raise ValueError("No valid price data after removing NaN values")
            
            print(f"Downloaded data from {self.data.index[0]} to {self.data.index[-1]}")
            print(f"Data shape: {self.data.shape}")
            
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("This might be due to:")
            print("- Internet connectivity issues")
            print("- Yahoo Finance API limitations")
            print("- Invalid ticker symbols")
            print("- Market holidays or weekends")
            raise
    
    def compute_returns(self) -> pd.DataFrame:
        """Compute daily returns."""
        if self.data is None or self.data.empty:
            raise ValueError("Must fetch data first")
        
        # Compute log returns
        self.returns = np.log(self.data / self.data.shift(1)).dropna()
        
        print(f"Computed returns for {len(self.returns)} trading days")
        return self.returns
    
    def estimate_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate drift (mu), volatility (sigma), and correlation parameters.
        
        Returns:
            mu: Annualized drift vector
            sigma: Annualized volatility vector  
            correlation: Correlation matrix
        """
        if self.returns is None or self.returns.empty:
            self.compute_returns()
        
        # Annualization factor (252 trading days per year)
        ann_factor = 252
        
        # Estimate parameters
        mu = self.returns.mean().values * ann_factor
        sigma = self.returns.std().values * np.sqrt(ann_factor)
        correlation = self.returns.corr().values
        
        # Store parameters
        self.parameters = {
            'mu': mu,
            'sigma': sigma, 
            'correlation': correlation,
            'tickers': self.tickers,
            'current_prices': self.data.iloc[-1].values
        }
        
        print("\nEstimated Parameters:")
        print("-" * 40)
        for i, ticker in enumerate(self.tickers):
            print(f"{ticker:6}: μ={mu[i]:6.1%}, σ={sigma[i]:6.1%}, Price=${self.parameters['current_prices'][i]:6.2f}")
        
        print(f"\nCorrelation Matrix:")
        corr_df = pd.DataFrame(correlation, index=self.tickers, columns=self.tickers)
        print(corr_df.round(3))
        
        return mu, sigma, correlation
    
    def plot_price_history(self, save_path: str = None):
        """Plot historical price data."""
        if self.data is None or self.data.empty:
            raise ValueError("Must fetch data first")
        
        # Normalize prices to start at 100 for comparison
        normalized_data = self.data / self.data.iloc[0] * 100
        
        plt.figure(figsize=(12, 8))
        
        for ticker in self.tickers:
            plt.plot(normalized_data.index, normalized_data[ticker], 
                    label=ticker, linewidth=2)
        
        plt.title('Normalized Stock Price History', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Price (Start = 100)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_returns_correlation(self, save_path: str = None):
        """Plot correlation heatmap of returns."""
        if self.returns is None or self.returns.empty:
            self.compute_returns()
        
        plt.figure(figsize=(10, 8))
        
        corr_matrix = self.returns.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, fmt='.3f')
        
        plt.title('Stock Returns Correlation Matrix', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class AmericanOptionsExample:
    """Complete example using real market data for American options."""
    
    def __init__(self, tickers: List[str], 
                 strikes: Dict[str, float] = None,
                 option_types: Dict[str, str] = None,
                 time_to_expiry: float = 0.25,
                 generate_reports: bool = False,
                 results_dir: str = "examples/results"):
        """
        Initialize American options example.
        
        Args:
            tickers: Stock symbols
            strikes: Strike prices for each ticker (default: current price)
            option_types: 'call' or 'put' for each ticker
            time_to_expiry: Time to expiration in years (default: 3 months)
            generate_reports: Whether to generate and save comprehensive reports
            results_dir: Directory to save results and reports
        """
        self.tickers = tickers
        self.strikes = strikes or {}
        self.option_types = option_types or {}
        self.time_to_expiry = time_to_expiry
        self.generate_reports = generate_reports
        
        # Setup results directory
        self.results_dir = Path(results_dir)
        if self.generate_reports:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
        # Create timestamped run directory for this analysis
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"run_{self.run_timestamp}"
        if self.generate_reports:
            self.run_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_processor = MarketDataProcessor(tickers)
        self.solver = None
        self.results = None
        self.simulation_results = None
        
        # Report data collection
        self.report_data = {
            'metadata': {
                'timestamp': self.run_timestamp,
                'tickers': tickers,
                'time_to_expiry': time_to_expiry
            }
        }
        
    def run_complete_example(self):
        """Run the complete American options example."""
        print("="*80)
        print("AMERICAN OPTIONS OPTIMAL STOPPING - REAL MARKET DATA EXAMPLE")
        print("="*80)
        
        if self.generate_reports:
            print(f"Reports will be saved to: {self.run_dir}")
        
        try:
            # Step 1: Fetch and process market data
            print("\n1. FETCHING MARKET DATA")
            print("-" * 40)
            self.data_processor.fetch_data()
            mu, sigma, correlation = self.data_processor.estimate_parameters()
            
            # Store market data for reporting
            self.report_data['market_data'] = {
                'mu': mu.tolist(),
                'sigma': sigma.tolist(),
                'correlation': correlation.tolist(),
                'current_prices': self.data_processor.parameters['current_prices'].tolist(),
                'data_period': self.data_processor.period,
                'data_shape': list(self.data_processor.data.shape),
                'start_date': str(self.data_processor.data.index[0].date()),
                'end_date': str(self.data_processor.data.index[-1].date())
            }
            
            # Step 2: Set up option parameters
            print("\n2. SETTING UP OPTION PARAMETERS")
            print("-" * 40)
            current_prices = self.data_processor.parameters['current_prices']
            self._setup_option_parameters(current_prices)
            
            # Step 3: Configure and solve
            print("\n3. CONFIGURING OPTIMAL STOPPING SOLVER")
            print("-" * 40)
            self._configure_solver(mu, sigma, correlation, current_prices)
            
            print("\n4. SOLVING OPTIMAL STOPPING PROBLEM")
            print("-" * 40)
            self.results = self.solver.solve()
            
            # Store solver results for reporting
            summary = self.solver.get_summary()
            self.report_data['solver_results'] = summary
            
            # Step 4: Analyze results
            print("\n5. ANALYZING RESULTS")
            print("-" * 40)
            analysis_results = self._analyze_results()
            self.report_data['analysis'] = analysis_results
            
            # Step 5: Run simulations
            print("\n6. RUNNING MONTE CARLO SIMULATIONS")
            print("-" * 40)
            simulation_results = self._run_simulations(current_prices)
            self.report_data['simulations'] = simulation_results
            
            # Step 6: Visualize results
            print("\n7. GENERATING VISUALIZATIONS")
            print("-" * 40)
            self._create_visualizations()
            
            # Step 7: Generate reports if requested
            if self.generate_reports:
                print("\n8. GENERATING COMPREHENSIVE REPORTS")
                print("-" * 40)
                self._generate_reports()
            
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE")
            if self.generate_reports:
                print(f"All reports saved to: {self.run_dir}")
            print("="*80)
            
        except Exception as e:
            print(f"\nError during analysis: {e}")
            if self.generate_reports:
                self._save_error_report(str(e))
            raise
    
    def _setup_option_parameters(self, current_prices: np.ndarray):
        """Setup option strikes and types."""
        n_assets = len(self.tickers)
        
        # Set default strikes (at-the-money)
        for i, ticker in enumerate(self.tickers):
            if ticker not in self.strikes:
                self.strikes[ticker] = current_prices[i]
        
        # Set default option types (alternate call/put)
        for i, ticker in enumerate(self.tickers):
            if ticker not in self.option_types:
                self.option_types[ticker] = 'call' if i % 2 == 0 else 'put'
        
        print("Option Setup:")
        option_setup = []
        for i, ticker in enumerate(self.tickers):
            option_type = self.option_types[ticker]
            strike = self.strikes[ticker]
            current = current_prices[i]
            moneyness = current / strike
            
            option_info = {
                'ticker': ticker,
                'type': option_type,
                'strike': float(strike),
                'current_price': float(current),
                'moneyness': float(moneyness)
            }
            option_setup.append(option_info)
            
            print(f"  {ticker:6}: {option_type.upper():4} option, "
                  f"Strike=${strike:6.2f}, Current=${current:6.2f}, "
                  f"Moneyness={moneyness:.3f}")
        
        # Store for reporting
        self.report_data['options'] = option_setup
    
    def _configure_solver(self, mu: np.ndarray, sigma: np.ndarray, 
                         correlation: np.ndarray, current_prices: np.ndarray):
        """Configure the optimal stopping solver."""
        n_assets = len(self.tickers)
        
        # Configure grid (optimized for demonstration)
        grid_params = GridParameters(
            n_time_steps=15,     # Reduced from 30
            n_wealth_points=12,  # Reduced from 20
            n_price_points=12,   # Reduced from 20
            wealth_min=0.5,
            wealth_max=3.0,
            price_min=0.7 * min(current_prices),
            price_max=1.5 * max(current_prices)
        )
        
        # Configure solver
        config = SolverConfig(
            n_assets=n_assets,
            max_time=self.time_to_expiry,
            risk_free_rate=0.05,  # 5% risk-free rate
            grid_params=grid_params,
            use_parallel=False,  # Keep False for demonstration
            log_level="INFO"
        )
        
        # Create solver
        self.solver = MultiAssetOptimalStoppingSolver(config)
        
        # Set market parameters
        self.solver.set_market_parameters(mu, sigma, correlation)
        
        # Create payoff functions based on option types
        payoff_functions = {}
        for i, ticker in enumerate(self.tickers):
            strike = self.strikes[ticker]
            option_type = self.option_types[ticker]
            
            if option_type.lower() == 'call':
                payoff_functions[i] = PayoffFunctions.american_call(strike, 1.0)
            else:
                payoff_functions[i] = PayoffFunctions.american_put(strike, 1.0)
        
        self.solver.set_payoff_functions(payoff_functions)
        
        print(f"Solver configured with {n_assets} assets")
        print(f"Grid size: {grid_params.n_time_steps} × {grid_params.n_wealth_points} × {grid_params.n_price_points}")
        print(f"Time horizon: {self.time_to_expiry} years")
    
    def _analyze_results(self):
        """Analyze the optimization results."""
        if not self.results:
            return {}
        
        summary = self.solver.get_summary()
        
        print(f"Solution Summary:")
        print(f"  Solve time: {summary['solve_time']:.2f} seconds")
        print(f"  Validation passed: {summary['validation_passed']}")
        print(f"  Maximum value: {summary['max_value']:.4f}")
        
        # Test some specific scenarios
        current_prices = self.data_processor.parameters['current_prices']
        test_scenarios = [
            (0.0, 1.0, "Start"),
            (self.time_to_expiry * 0.5, 1.0, "Mid-point"),
            (self.time_to_expiry * 0.9, 1.0, "Near expiry")
        ]
        
        print(f"\nValue Function at Key Points:")
        print(f"{'Scenario':<12} {'Time':<6} {'Wealth':<7} {'Value':<8} {'Action'}")
        print("-" * 60)
        
        active_set = set(range(len(self.tickers)))
        scenario_results = []
        
        for t, wealth, description in test_scenarios:
            try:
                value = self.solver.get_value(t, wealth, current_prices.tolist(), active_set)
                portfolio, stop_asset = self.solver.get_optimal_policy(t, wealth, current_prices.tolist(), active_set)
                
                action = f"Stop {self.tickers[stop_asset]}" if stop_asset is not None else "Continue"
                print(f"{description:<12} {t:<6.2f} {wealth:<7.2f} {value:<8.4f} {action}")
                
                scenario_results.append({
                    'scenario': description,
                    'time': float(t),
                    'wealth': float(wealth),
                    'value': float(value),
                    'stop_asset': stop_asset,
                    'stop_ticker': self.tickers[stop_asset] if stop_asset is not None else None,
                    'action': action,
                    'portfolio': portfolio.tolist() if portfolio is not None else None
                })
                
            except Exception as e:
                error_msg = str(e)[:50]
                print(f"{description:<12} {t:<6.2f} {wealth:<7.2f} {'ERROR':<8} {error_msg}...")
                scenario_results.append({
                    'scenario': description,
                    'time': float(t),
                    'wealth': float(wealth),
                    'error': error_msg
                })
        
        return {
            'scenarios': scenario_results,
            'summary': summary
        }
    
    def _run_simulations(self, current_prices: np.ndarray, n_simulations: int = 100):
        """Run Monte Carlo simulations."""
        print(f"Running {n_simulations} Monte Carlo paths...")
        
        final_wealths = []
        stopping_stats = {ticker: 0 for ticker in self.tickers}
        successful_sims = []
        failed_sims = []
        
        for i in range(n_simulations):
            try:
                sim_result = self.solver.simulate_path(
                    initial_wealth=1.0,
                    initial_prices=current_prices,
                    n_steps=50,
                    seed=42 + i
                )
                
                final_wealths.append(sim_result['final_wealth'])
                successful_sims.append({
                    'simulation_id': i,
                    'final_wealth': float(sim_result['final_wealth']),
                    'final_active_set': list(sim_result['final_active_set']),
                    'stopping_decisions': sim_result['stopping_decisions']
                })
                
                # Count stopping decisions
                for decision in sim_result['stopping_decisions']:
                    if decision is not None:
                        ticker = self.tickers[decision]
                        stopping_stats[ticker] += 1
                        
            except Exception as e:
                print(f"  Simulation {i+1} failed: {e}")
                failed_sims.append({'simulation_id': i, 'error': str(e)})
                continue
        
        simulation_summary = {}
        if final_wealths:
            wealth_stats = {
                'mean': float(np.mean(final_wealths)),
                'std': float(np.std(final_wealths)),
                'min': float(np.min(final_wealths)),
                'max': float(np.max(final_wealths)),
                'median': float(np.median(final_wealths)),
                'q25': float(np.percentile(final_wealths, 25)),
                'q75': float(np.percentile(final_wealths, 75))
            }
            
            stopping_frequencies = {}
            for ticker, count in stopping_stats.items():
                frequency = count / len(final_wealths)
                stopping_frequencies[ticker] = {
                    'count': count,
                    'frequency': float(frequency)
                }
            
            simulation_summary = {
                'n_simulations': n_simulations,
                'n_successful': len(final_wealths),
                'n_failed': len(failed_sims),
                'wealth_statistics': wealth_stats,
                'stopping_frequencies': stopping_frequencies,
                'successful_simulations': successful_sims[:10],  # Store first 10 detailed results
                'failed_simulations': failed_sims
            }
            
            print(f"\nSimulation Results ({len(final_wealths)} successful paths):")
            print(f"  Mean final wealth: {wealth_stats['mean']:.4f}")
            print(f"  Std final wealth:  {wealth_stats['std']:.4f}")
            print(f"  Min final wealth:  {wealth_stats['min']:.4f}")
            print(f"  Max final wealth:  {wealth_stats['max']:.4f}")
            
            print(f"\nStopping Frequencies:")
            for ticker, stats in stopping_frequencies.items():
                print(f"  {ticker}: {stats['count']:3d} times ({stats['frequency']:.1%})")
        else:
            print("  All simulations failed")
            simulation_summary = {
                'n_simulations': n_simulations,
                'n_successful': 0,
                'n_failed': len(failed_sims),
                'failed_simulations': failed_sims
            }
        
        self.simulation_results = simulation_summary
        return simulation_summary
    
    def _create_visualizations(self):
        """Create visualizations of the results."""
        try:
            save_plots = self.generate_reports
            
            # Plot market data
            print("  Creating price history plot...")
            save_path = str(self.run_dir / "price_history.png") if save_plots else None
            self.data_processor.plot_price_history(save_path=save_path)
            
            # Plot correlation matrix
            print("  Creating correlation heatmap...")
            save_path = str(self.run_dir / "correlation_matrix.png") if save_plots else None
            self.data_processor.plot_returns_correlation(save_path=save_path)
            
            # Plot simulation results if available
            if self.simulation_results and self.simulation_results.get('n_successful', 0) > 0:
                print("  Creating simulation results plots...")
                self._plot_simulation_results()
            
            # Plot value function (if solver succeeded)
            if self.solver and hasattr(self.solver, 'analyzer') and self.solver.analyzer:
                print("  Creating value function plots...")
                active_set = set(range(len(self.tickers)))
                if save_plots:
                    self.solver.plot_results(active_set, save_dir=str(self.run_dir))
                else:
                    self.solver.plot_results(active_set)
            
        except Exception as e:
            print(f"  Visualization error: {e}")
    
    def _plot_simulation_results(self):
        """Plot Monte Carlo simulation results."""
        if not self.simulation_results or self.simulation_results['n_successful'] == 0:
            return
        
        # Extract data for plotting
        successful_sims = self.simulation_results['successful_simulations']
        final_wealths = [sim['final_wealth'] for sim in successful_sims]
        
        if not final_wealths:
            return
        
        # Create wealth distribution plot
        plt.figure(figsize=(12, 5))
        
        # Histogram of final wealths
        plt.subplot(1, 2, 1)
        plt.hist(final_wealths, bins=min(20, len(final_wealths)//5), 
                alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(final_wealths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(final_wealths):.3f}')
        plt.xlabel('Final Wealth')
        plt.ylabel('Frequency')
        plt.title('Distribution of Final Wealth (Monte Carlo)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Stopping frequency bar chart
        plt.subplot(1, 2, 2)
        stopping_freq = self.simulation_results['stopping_frequencies']
        tickers = list(stopping_freq.keys())
        frequencies = [stopping_freq[ticker]['frequency'] for ticker in tickers]
        
        plt.bar(tickers, frequencies, alpha=0.7)
        plt.xlabel('Asset')
        plt.ylabel('Stopping Frequency')
        plt.title('Asset Stopping Frequencies')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.generate_reports:
            plt.savefig(self.run_dir / "simulation_results.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _generate_reports(self):
        """Generate comprehensive reports and save to results directory."""
        print("  Generating JSON summary report...")
        self._save_json_report()
        
        print("  Generating detailed analysis report...")
        self._save_detailed_report()
        
        print("  Generating market data report...")
        self._save_market_data()
        
        print("  Saving solver results...")
        self._save_solver_results()
        
        print("  Generating executive summary...")
        self._save_executive_summary()
    
    def _save_json_report(self):
        """Save comprehensive JSON report with all results."""
        report_file = self.run_dir / "comprehensive_report.json"
        
        # Ensure all data is JSON serializable
        json_data = self._make_json_serializable(self.report_data)
        
        with open(report_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"    JSON report saved: {report_file}")
    
    def _save_detailed_report(self):
        """Save detailed markdown report."""
        report_file = self.run_dir / "detailed_analysis.md"
        
        with open(report_file, 'w') as f:
            f.write(self._generate_markdown_report())
        
        print(f"    Markdown report saved: {report_file}")
    
    def _save_market_data(self):
        """Save market data as CSV files."""
        # Save price data
        if hasattr(self.data_processor, 'data') and not self.data_processor.data.empty:
            price_file = self.run_dir / "market_prices.csv"
            self.data_processor.data.to_csv(price_file)
            
            # Save returns data
            if hasattr(self.data_processor, 'returns') and not self.data_processor.returns.empty:
                returns_file = self.run_dir / "market_returns.csv"
                self.data_processor.returns.to_csv(returns_file)
        
        # Save parameters as CSV
        if 'market_data' in self.report_data:
            params_file = self.run_dir / "market_parameters.csv"
            params_df = pd.DataFrame({
                'ticker': self.tickers,
                'mu': self.report_data['market_data']['mu'],
                'sigma': self.report_data['market_data']['sigma'],
                'current_price': self.report_data['market_data']['current_prices']
            })
            params_df.to_csv(params_file, index=False)
        
        print(f"    Market data saved to CSV files")
    
    def _save_solver_results(self):
        """Save solver results using pickle for later analysis."""
        if self.solver and self.results:
            results_file = self.run_dir / "solver_results.pkl"
            
            solver_data = {
                'results': self.results,
                'solver_config': self.solver.config,
                'tickers': self.tickers,
                'run_timestamp': self.run_timestamp
            }
            
            # Try to save value function if available (it's stored in the internal solver)
            if hasattr(self.solver, '_solver') and hasattr(self.solver._solver, 'value_function'):
                solver_data['value_function'] = self.solver._solver.value_function
            elif hasattr(self.solver, 'value_function'):
                solver_data['value_function'] = self.solver.value_function
            
            with open(results_file, 'wb') as f:
                pickle.dump(solver_data, f)
            
            print(f"    Solver results saved: {results_file}")
    
    def _save_executive_summary(self):
        """Save executive summary report."""
        summary_file = self.run_dir / "executive_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(self._generate_executive_summary())
        
        print(f"    Executive summary saved: {summary_file}")
    
    def _save_error_report(self, error_message: str):
        """Save error report when analysis fails."""
        error_file = self.run_dir / "error_report.txt"
        
        with open(error_file, 'w') as f:
            f.write(f"Multi-Asset Optimal Stopping Analysis - Error Report\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Timestamp: {self.run_timestamp}\n")
            f.write(f"Tickers: {self.tickers}\n")
            f.write(f"Time to Expiry: {self.time_to_expiry}\n\n")
            f.write(f"Error Message:\n{error_message}\n\n")
            
            if hasattr(self, 'report_data'):
                f.write("Available Data:\n")
                for key in self.report_data.keys():
                    f.write(f"- {key}\n")
        
        print(f"    Error report saved: {error_file}")
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj
    
    def _generate_markdown_report(self) -> str:
        """Generate detailed markdown report."""
        md = []
        md.append("# Multi-Asset Optimal Stopping Analysis Report")
        md.append("=" * 50)
        md.append("")
        
        # Metadata
        md.append("## Analysis Metadata")
        md.append(f"- **Timestamp**: {self.run_timestamp}")
        md.append(f"- **Assets**: {', '.join(self.tickers)}")
        md.append(f"- **Time to Expiry**: {self.time_to_expiry} years")
        md.append("")
        
        # Market Data
        if 'market_data' in self.report_data:
            md.append("## Market Data Summary")
            market_data = self.report_data['market_data']
            md.append(f"- **Data Period**: {market_data['data_period']}")
            md.append(f"- **Date Range**: {market_data['start_date']} to {market_data['end_date']}")
            md.append(f"- **Data Points**: {market_data['data_shape'][0]} days")
            md.append("")
            
            md.append("### Estimated Parameters")
            for i, ticker in enumerate(self.tickers):
                mu = market_data['mu'][i]
                sigma = market_data['sigma'][i]
                price = market_data['current_prices'][i]
                md.append(f"- **{ticker}**: μ={mu:.1%}, σ={sigma:.1%}, Price=${price:.2f}")
            md.append("")
        
        # Options Setup
        if 'options' in self.report_data:
            md.append("## Options Configuration")
            for option in self.report_data['options']:
                md.append(f"- **{option['ticker']}**: {option['type'].upper()} option, "
                         f"Strike=${option['strike']:.2f}, Moneyness={option['moneyness']:.3f}")
            md.append("")
        
        # Solver Results
        if 'solver_results' in self.report_data:
            solver_results = self.report_data['solver_results']
            md.append("## Solver Performance")
            md.append(f"- **Solve Time**: {solver_results['solve_time']:.2f} seconds")
            md.append(f"- **Grid Size**: {solver_results['grid_size']}")
            md.append(f"- **Validation Passed**: {solver_results['validation_passed']}")
            md.append(f"- **Maximum Value**: {solver_results['max_value']:.4f}")
            md.append("")
        
        # Analysis Results
        if 'analysis' in self.report_data and 'scenarios' in self.report_data['analysis']:
            md.append("## Value Function Analysis")
            md.append("| Scenario | Time | Wealth | Value | Action |")
            md.append("|----------|------|--------|-------|--------|")
            
            for scenario in self.report_data['analysis']['scenarios']:
                if 'error' not in scenario:
                    md.append(f"| {scenario['scenario']} | {scenario['time']:.2f} | "
                             f"{scenario['wealth']:.2f} | {scenario['value']:.4f} | {scenario['action']} |")
            md.append("")
        
        # Simulation Results
        if 'simulations' in self.report_data and self.report_data['simulations']['n_successful'] > 0:
            sim_data = self.report_data['simulations']
            md.append("## Monte Carlo Simulation Results")
            md.append(f"- **Simulations Run**: {sim_data['n_simulations']}")
            md.append(f"- **Successful**: {sim_data['n_successful']}")
            md.append(f"- **Failed**: {sim_data['n_failed']}")
            md.append("")
            
            if 'wealth_statistics' in sim_data:
                wealth_stats = sim_data['wealth_statistics']
                md.append("### Final Wealth Statistics")
                md.append(f"- **Mean**: {wealth_stats['mean']:.4f}")
                md.append(f"- **Std Dev**: {wealth_stats['std']:.4f}")
                md.append(f"- **Median**: {wealth_stats['median']:.4f}")
                md.append(f"- **Range**: [{wealth_stats['min']:.4f}, {wealth_stats['max']:.4f}]")
                md.append("")
            
            if 'stopping_frequencies' in sim_data:
                md.append("### Stopping Frequencies")
                for ticker, stats in sim_data['stopping_frequencies'].items():
                    md.append(f"- **{ticker}**: {stats['count']} times ({stats['frequency']:.1%})")
                md.append("")
        
        return "\n".join(md)
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary."""
        summary = []
        summary.append("MULTI-ASSET OPTIMAL STOPPING - EXECUTIVE SUMMARY")
        summary.append("=" * 55)
        summary.append("")
        summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Assets Analyzed: {', '.join(self.tickers)}")
        summary.append(f"Time Horizon: {self.time_to_expiry} years")
        summary.append("")
        
        # Key findings
        summary.append("KEY FINDINGS:")
        summary.append("-" * 15)
        
        if 'solver_results' in self.report_data:
            solver_results = self.report_data['solver_results']
            summary.append(f"✓ Algorithm solved successfully in {solver_results['solve_time']:.1f} seconds")
            summary.append(f"✓ Maximum portfolio value: {solver_results['max_value']:.4f}")
        
        if 'simulations' in self.report_data and self.report_data['simulations']['n_successful'] > 0:
            sim_data = self.report_data['simulations']
            wealth_stats = sim_data.get('wealth_statistics', {})
            if wealth_stats:
                summary.append(f"✓ Monte Carlo mean return: {wealth_stats['mean']:.4f} ({(wealth_stats['mean']-1)*100:+.1f}%)")
                summary.append(f"✓ Success rate: {sim_data['n_successful']}/{sim_data['n_simulations']} simulations")
        
        # Risk assessment
        if 'market_data' in self.report_data:
            market_data = self.report_data['market_data']
            max_vol = max(market_data['sigma'])
            summary.append(f"✓ Maximum asset volatility: {max_vol:.1%}")
        
        summary.append("")
        summary.append("RECOMMENDATIONS:")
        summary.append("-" * 15)
        
        # Generate basic recommendations based on results
        if 'simulations' in self.report_data:
            sim_data = self.report_data['simulations']
            if sim_data.get('n_successful', 0) > 0:
                stopping_freq = sim_data.get('stopping_frequencies', {})
                if stopping_freq:
                    most_stopped = max(stopping_freq.keys(), key=lambda x: stopping_freq[x]['frequency'])
                    freq = stopping_freq[most_stopped]['frequency']
                    summary.append(f"• Asset {most_stopped} shows highest stopping frequency ({freq:.1%})")
                    summary.append(f"• Consider position sizing based on optimal stopping probabilities")
        
        summary.append("• Monitor market conditions for parameter changes")
        summary.append("• Regular rebalancing recommended based on optimal policies")
        summary.append("")
        
        return "\n".join(summary)


def main(generate_reports: bool = False):
    """Run the main American options example."""
    
    # Define popular tech stocks
    TECH_TICKERS = ['AAPL', 'GOOGL', 'MSFT']
    FINANCIAL_TICKERS = ['JPM', 'BAC', 'WFC'] 
    MIXED_TICKERS = ['AAPL', 'JPM', 'TSLA']
    
    # Choose ticker set
    tickers = TECH_TICKERS  # Change this to test different combinations
    
    # Create and run example
    example = AmericanOptionsExample(
        tickers=tickers,
        time_to_expiry=0.25,  # 3 months
        generate_reports=generate_reports
    )
    
    try:
        example.run_complete_example()
        return example
        
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
        return None
        
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_with_reports():
    """Convenience function to run example with report generation enabled."""
    return main(generate_reports=True)


def run_multiple_scenarios():
    """Run multiple scenarios and generate comparative reports."""
    scenarios = [
        {'name': 'Tech Stocks', 'tickers': ['AAPL', 'GOOGL', 'MSFT']},
        {'name': 'Financial Stocks', 'tickers': ['JPM', 'BAC', 'WFC']},
        {'name': 'Mixed Portfolio', 'tickers': ['AAPL', 'JPM', 'TSLA']}
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"RUNNING SCENARIO: {scenario['name']}")
        print(f"{'='*80}")
        
        example = AmericanOptionsExample(
            tickers=scenario['tickers'],
            time_to_expiry=0.25,
            generate_reports=True
        )
        
        try:
            example.run_complete_example()
            results[scenario['name']] = example
            
        except Exception as e:
            print(f"Scenario {scenario['name']} failed: {e}")
            results[scenario['name']] = None
    
    # Generate comparative summary
    _generate_comparative_report(results)
    
    return results


def _generate_comparative_report(scenario_results: Dict):
    """Generate a comparative report across multiple scenarios."""
    results_dir = Path("examples/results")
    results_dir.mkdir(exist_ok=True)
    
    comparison_file = results_dir / f"comparative_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(comparison_file, 'w') as f:
        f.write("# Multi-Scenario Comparative Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Scenario Summary\n")
        f.write("| Scenario | Assets | Status | Solve Time | Max Value |\n")
        f.write("|----------|--------|---------|------------|----------|\n")
        
        for scenario_name, example in scenario_results.items():
            if example and hasattr(example, 'report_data') and 'solver_results' in example.report_data:
                solver_results = example.report_data['solver_results']
                assets = ", ".join(example.tickers)
                status = "✓ Success" if solver_results['validation_passed'] else "✗ Failed"
                solve_time = solver_results['solve_time']
                max_value = solver_results['max_value']
                
                f.write(f"| {scenario_name} | {assets} | {status} | {solve_time:.2f}s | {max_value:.4f} |\n")
            else:
                f.write(f"| {scenario_name} | - | ✗ Error | - | - |\n")
        
        f.write("\n## Key Insights\n")
        
        # Generate insights based on successful scenarios
        successful_scenarios = {k: v for k, v in scenario_results.items() if v is not None}
        
        if successful_scenarios:
            f.write("- **Performance Comparison**: ")
            solve_times = []
            for name, example in successful_scenarios.items():
                if hasattr(example, 'report_data') and 'solver_results' in example.report_data:
                    solve_times.append((name, example.report_data['solver_results']['solve_time']))
            
            if solve_times:
                fastest = min(solve_times, key=lambda x: x[1])
                f.write(f"Fastest solver: {fastest[0]} ({fastest[1]:.2f}s)\n")
        
        f.write("\n## Recommendations\n")
        f.write("- Review individual scenario reports for detailed analysis\n")
        f.write("- Consider asset correlation effects on stopping strategies\n")
        f.write("- Monitor solver performance across different asset types\n")
    
    print(f"\nComparative analysis saved to: {comparison_file}")
    return comparison_file


if __name__ == "__main__":
    import sys
    
    # Check command line arguments for report generation
    generate_reports = '--reports' in sys.argv or '-r' in sys.argv
    multiple_scenarios = '--multiple' in sys.argv or '-m' in sys.argv
    
    if multiple_scenarios:
        print("Running multiple scenarios with report generation...")
        results = run_multiple_scenarios()
        successful_count = sum(1 for r in results.values() if r is not None)
        print(f"\n{'='*80}")
        print(f"Multiple scenario analysis completed!")
        print(f"Successful scenarios: {successful_count}/{len(results)}")
        print(f"Results saved in examples/results/")
        print("="*80)
    
    else:
        if generate_reports:
            print("Running with comprehensive report generation enabled...")
            print("Use --multiple or -m to run multiple asset scenarios")
        
        # Run the example
        result = main(generate_reports=generate_reports)
        
        if result:
            print("\n" + "="*80)
            print("Example completed successfully!")
            print("The solver has computed optimal stopping strategies for American options")
            print("using real market data.")
            
            if generate_reports:
                print(f"Comprehensive reports saved to: {result.run_dir}")
                print("\nGenerated files:")
                print("- comprehensive_report.json  (Complete data)")
                print("- detailed_analysis.md       (Formatted report)")
                print("- executive_summary.txt      (Key findings)")
                print("- market_prices.csv          (Historical data)")
                print("- market_parameters.csv      (Estimated parameters)")
                print("- solver_results.pkl         (Full solver state)")
                print("- *.png                      (Visualization plots)")
            else:
                print("Use --reports or -r flag to generate comprehensive reports")
                print("Use --multiple or -m to run multiple asset scenarios")
            
            print("="*80)
        else:
            print("\nExample failed. Please check the error messages above.")
    
    print("\nUsage options:")
    print("  python examples/options.py              # Run basic example")
    print("  python examples/options.py --reports    # Run with full reports")
    print("  python examples/options.py --multiple   # Run multiple scenarios")