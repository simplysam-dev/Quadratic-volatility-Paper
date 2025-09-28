# QNV Solver: Production-Grade Implementation

This directory contains a complete production-grade implementation of the Quadratic Normal Volatility (QNV) model solver using the confluent Heun equation mapping derived in the accompanying paper.

## Files Overview

### Core Implementation
- **`qnv_solver.py`** - Main solver implementation with all core classes and methods
- **`qnv_usage_example.py`** - Comprehensive usage examples and demonstrations

### Documentation
- **`main.tex`** - Complete LaTeX paper with mathematical derivation
- **`main.pdf`** - Compiled PDF of the mathematical paper

### Generated Outputs
- **`qnv_test_case_*.png`** - Diagnostic plots from synthetic test cases
- **`qnv_usage_example.png`** - Comprehensive diagnostic dashboard
- **`qnv_smile_comparison.png`** - Volatility smile analysis comparison

## Key Features

### 🎯 **Automatic Method Selection**
The solver automatically chooses the optimal solution method based on parameter analysis:
- **Heun Series**: For rapidly converging cases
- **Polynomial Solutions**: When exact solutions exist
- **Pöschl-Teller**: For symmetric cases (r≈0)
- **PDE Finite Difference**: For general cases requiring numerical solution

### 🔧 **Core Classes**

#### `QNVParameters`
```python
params = QNVParameters(
    a=0.00001,    # quadratic coefficient (controls smile convexity)
    b=0.01,       # linear coefficient (controls skew)
    c=0.2,        # constant coefficient (base volatility level)
    r=0.02,       # risk-free rate
    S0=100,       # current spot price
    F=100         # forward price
)
```

#### `QNVSolver`
Main solver class that:
- Converts QNV parameters to Heun equation parameters
- Implements three-term recurrence relation for series solutions
- Detects polynomial termination conditions
- Provides comprehensive error handling and validation

#### `FiniteDifferenceSolver`
High-performance PDE solver using:
- Crank-Nicolson scheme for maximum stability
- Adaptive time stepping
- Optimized tridiagonal matrix solving
- Production-ready boundary condition handling

#### `AdaptiveSolverSelector`
Intelligent method recommendation based on:
- Parameter regime analysis
- Convergence condition checking
- Special case detection (symmetric, polynomial)

#### `QNVVisualizer`
Comprehensive diagnostic visualization:
- Volatility smile plots
- Quantum potential landscape
- Heun function solutions
- Parameter sensitivity analysis
- Convergence diagnostics

## Quick Start

### Basic Usage
```python
from qnv_solver import QNVParameters, QNVSolver, AdaptiveSolverSelector

# Define model parameters
params = QNVParameters(a=0.00001, b=0.01, c=0.2, r=0.02, S0=100, F=100)

# Initialize solver
solver = QNVSolver(params)

# Get method recommendation
selector = AdaptiveSolverSelector(params)
recommendation = selector.recommend_method()
print(f"Recommended method: {recommendation['primary']}")

# Solve using recommended method
if recommendation['primary'] == 'heun_series':
    y, solution = solver.solve_heun_series(y_max=0.5, n_terms=30)
```

### Option Pricing
```python
from qnv_solver import FiniteDifferenceSolver

# Initialize PDE solver
pde_solver = FiniteDifferenceSolver(params, S_min=50, S_max=150, n_points=200)

# Price European call option
prices = pde_solver.price_european_call(K=100, T=0.25)
```

### Diagnostic Analysis
```python
from qnv_solver import QNVVisualizer

# Create comprehensive diagnostic plots
visualizer = QNVVisualizer(solver)
fig = visualizer.create_diagnostic_dashboard()
plt.savefig('qnv_analysis.png', dpi=300, bbox_inches='tight')
```

## Mathematical Foundation

The solver implements the complete mathematical framework from the paper:

1. **QNV Model**: σ(S) = aS² + bS + c
2. **Confluent Heun Mapping**: Automatic conversion to Heun equation parameters
3. **Series Solutions**: Three-term recurrence relation implementation
4. **Polynomial Cases**: Detection and exact solution computation
5. **Symmetric Limit**: Pöschl-Teller reduction when r→0

## Test Results

The synthetic test suite validates the solver across different parameter regimes:

- ✅ **Symmetric Case (r=0)**: Correctly reduces to Pöschl-Teller
- ✅ **Asymmetric Case (r>0)**: Handles δ-ε asymmetry properly
- ✅ **Near-Polynomial**: Detects special parameter conditions
- ✅ **Convergence**: Automatic rescaling prevents overflow
- ✅ **Visualization**: Comprehensive diagnostic plots generated

## Production Features

- **Error Handling**: Comprehensive validation and graceful failure modes
- **Logging**: Detailed progress tracking and warning system
- **Performance**: Optimized algorithms for production use
- **Scalability**: Handles large parameter ranges and grid sizes
- **Documentation**: Extensive docstrings and usage examples

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

## Usage Examples

Run the comprehensive test suite:
```bash
python3 qnv_solver.py
```

Run usage examples:
```bash
python3 qnv_usage_example.py
```

## Mathematical Validation

The implementation has been validated against:
- Theoretical convergence conditions
- Special case reductions (Pöschl-Teller limit)
- Parameter sensitivity analysis
- Volatility smile structure verification
- Heun parameter mapping accuracy

This production-grade solver provides a complete bridge from mathematical theory to practical implementation, enabling both research and commercial applications of the QNV model.
