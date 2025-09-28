# QNV Solver - Production-Grade Implementation

## Overview

This directory contains a **production-grade implementation** of the Quadratic Normal Volatility (QNV) model solver using the confluent Heun equation mapping. The implementation features optimal numerical schemes, comprehensive diagnostics, and intelligent method selection.

## Key Features

### 🚀 **Production-Ready Solver**
- **High-order PDE solver** with BDF2 and Crank-Nicolson schemes
- **Stable Heun series** with Miller's algorithm and convergence control
- **Adaptive time stepping** with error control
- **Intelligent method selection** based on parameter analysis

### 📊 **Comprehensive Diagnostics**
- **Multi-panel visualization** with 9 diagnostic plots
- **Volatility smile analysis** (local and implied)
- **Parameter sensitivity studies**
- **Time decay analysis**
- **Convergence diagnostics**

### 🔬 **Mathematical Rigor**
- **Correct Heun parameter computation** matching paper equations
- **Proper boundary conditions** for call options
- **Energy estimation** based on boundary conditions
- **Polynomial case detection** for special parameter regimes

## Files

### Core Implementation
- **`qnv_solver.py`** - Main solver implementation with all classes and methods
- **`qnv_usage_example.py`** - Comprehensive usage examples and demonstrations

### Documentation
- **`main.tex`** - Complete mathematical derivation (LaTeX source)
- **`main.pdf`** - Compiled mathematical paper
- **`README.md`** - This documentation file

### Generated Analysis
- **`qnv_test_case_*_comprehensive.png`** - Comprehensive diagnostic dashboards for test cases
- **`volatility_smile_analysis.png`** - Volatility smile analysis
- **`parameter_sensitivity.png`** - Parameter sensitivity studies
- **`time_decay_analysis.png`** - Time decay analysis
- **`comprehensive_dashboard.png`** - Full diagnostic dashboard

## Quick Start

### Basic Usage

```python
from qnv_solver import QNVParameters, AdaptiveSolver

# Define QNV parameters
params = QNVParameters(
    a=0.00001,    # quadratic coefficient
    b=0.01,       # linear coefficient
    c=0.2,        # constant coefficient
    r=0.02,       # risk-free rate
    S0=100,       # spot price
    F=100,        # forward price
    T=0.25        # time to maturity
)

# Create solver
solver = AdaptiveSolver(params)

# Price options
K = 100  # strike price
price = solver.solve(K)
print(f"Call price: {price:.4f}")
```

### Comprehensive Analysis

```python
from qnv_solver import QNVVisualizer

# Create visualizer
visualizer = QNVVisualizer(params)

# Generate comprehensive dashboard
fig = visualizer.create_comprehensive_dashboard()
```

## Test Suite

Run the comprehensive test suite:

```bash
python3 qnv_solver.py
```

This will:
- Test 5 different parameter regimes
- Generate comprehensive diagnostic plots
- Display Heun parameters and polynomial case detection
- Show option pricing results across strikes

## Usage Examples

Run the enhanced usage examples:

```bash
python3 qnv_usage_example.py
```

This includes:
1. **Basic option pricing** across different strikes
2. **Volatility smile analysis** with local and implied volatilities
3. **Parameter sensitivity studies** for coefficients a and b
4. **Time decay analysis** for different strikes
5. **Comprehensive diagnostic dashboard** with 9 analysis panels

## Solver Methods

### Automatic Method Selection

The solver automatically selects the optimal method based on parameters:

- **`heun_series`** - For symmetric cases (r≈0) or short maturities
- **`pde_high_order`** - For general cases requiring numerical PDE solution
- **`analytic_approx`** - For quick approximations using Black-Scholes with local vol

### Manual Method Selection

```python
# Force specific method
price = solver.solve(K, method='pde_high_order')
price = solver.solve(K, method='heun_series')
price = solver.solve(K, method='analytic_approx')
```

## Numerical Schemes

### PDE Solver
- **BDF2** (2nd-order Backward Differentiation Formula) - Most stable
- **Crank-Nicolson** - Balanced accuracy and stability
- **Implicit Euler** - Conservative fallback
- **4th-order compact spatial discretization**
- **Adaptive time stepping** with error control

### Heun Series
- **Miller's algorithm** for backward recurrence (more stable)
- **Horner's scheme** for polynomial evaluation
- **Convergence control** with coefficient rescaling
- **Polynomial case detection** for special parameters

## Parameter Validation

The solver includes comprehensive parameter validation:

```python
# Invalid parameters will raise exceptions
try:
    params = QNVParameters(a=-0.1, b=0.01, c=0.2, r=0.02, S0=100, F=100, T=0.25)
except ValueError as e:
    print(f"Parameter error: {e}")
```

## Visualization Features

### Comprehensive Dashboard (9 panels)
1. **Volatility Smile** - Local volatility vs strike
2. **Option Prices** - Call prices across strikes
3. **Implied Volatility Smile** - Implied vols vs strike
4. **Potential Landscape** - Quantum potential V(t)
5. **Heun Parameters** - γ, δ, ε, α_H, q values
6. **Heun Solutions** - Series solution plots
7. **Parameter Sensitivity** - Price vs parameter changes
8. **Time Decay** - Price evolution over time
9. **Convergence Analysis** - Error vs grid resolution

### Individual Analysis Plots
- Volatility smile analysis
- Parameter sensitivity studies
- Time decay analysis
- Comprehensive diagnostic dashboards

## Mathematical Background

The implementation is based on the mathematical derivation in `main.pdf`, which shows:

1. **QNV Model**: σ(S) = aS² + bS + c
2. **Confluent Heun Mapping**: Transforms the Black-Scholes PDE to the confluent Heun equation
3. **Parameter Relationships**: Exact formulas connecting QNV parameters to Heun parameters
4. **Special Cases**: Polynomial solutions, symmetric cases, and asymptotic behavior

## Performance Characteristics

- **High accuracy** with 4th-order spatial discretization
- **Stable time stepping** with adaptive control
- **Efficient linear algebra** using optimized scipy routines
- **Intelligent method selection** for optimal performance
- **Comprehensive error handling** and validation

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

## License

This implementation is part of the Quadratic Volatility Paper research project.
