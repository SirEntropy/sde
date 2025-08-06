# Multi-Asset Optimal Stopping Analysis Report
==================================================

## Analysis Metadata
- **Timestamp**: 20250805_215900
- **Assets**: AAPL, GOOGL, MSFT
- **Time to Expiry**: 0.25 years

## Market Data Summary
- **Data Period**: 2y
- **Date Range**: 2023-08-07 to 2025-08-05
- **Data Points**: 501 days

### Estimated Parameters
- **AAPL**: μ=6.9%, σ=27.2%, Price=$202.92
- **GOOGL**: μ=20.1%, σ=29.6%, Price=$194.67
- **MSFT**: μ=24.4%, σ=22.5%, Price=$527.75

## Options Configuration
- **AAPL**: CALL option, Strike=$202.92, Moneyness=1.000
- **GOOGL**: PUT option, Strike=$194.67, Moneyness=1.000
- **MSFT**: CALL option, Strike=$527.75, Moneyness=1.000

## Solver Performance
- **Solve Time**: 941.76 seconds
- **Grid Size**: (15, 12, 12)
- **Validation Passed**: True
- **Maximum Value**: 910.9810

## Value Function Analysis
| Scenario | Time | Wealth | Value | Action |
|----------|------|--------|-------|--------|
| Start | 0.00 | 1.00 | 758.3142 | Continue |
| Mid-point | 0.12 | 1.00 | 730.2693 | Continue |
| Near expiry | 0.23 | 1.00 | 713.6495 | Continue |

## Monte Carlo Simulation Results
- **Simulations Run**: 100
- **Successful**: 100
- **Failed**: 0

### Final Wealth Statistics
- **Mean**: 10.6815
- **Std Dev**: 26.9187
- **Median**: 1.0207
- **Range**: [1.0038, 180.7610]

### Stopping Frequencies
- **AAPL**: 0 times (0.0%)
- **GOOGL**: 20 times (20.0%)
- **MSFT**: 2 times (2.0%)
