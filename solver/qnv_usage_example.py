#!/usr/bin/env python3
"""
QNV Solver Usage Example
=======================

This script demonstrates how to use the production-grade QNV solver
for practical option pricing and analysis.
"""

from qnv_solver import QNVParameters, QNVSolver, AdaptiveSolverSelector, QNVVisualizer, FiniteDifferenceSolver
import numpy as np
import matplotlib.pyplot as plt

def example_option_pricing():
    """Example: Price European call options using QNV model"""
    
    print("=" * 60)
    print("QNV SOLVER USAGE EXAMPLE: OPTION PRICING")
    print("=" * 60)
    
    # Define QNV model parameters
    params = QNVParameters(
        a=0.00001,    # quadratic coefficient (controls smile convexity)
        b=0.01,       # linear coefficient (controls skew)
        c=0.2,        # constant coefficient (base volatility level)
        r=0.02,       # risk-free rate
        S0=100,       # current spot price
        F=100         # forward price
    )
    
    print(f"QNV Model Parameters:")
    print(f"  σ(S) = {params.a:.6f}S² + {params.b:.3f}S + {params.c:.3f}")
    print(f"  Spot Price: {params.S0}")
    print(f"  Risk-free Rate: {params.r:.1%}")
    print(f"  Discriminant Δ = {params.discriminant:.6f}")
    print(f"  Volatility at spot: {params.volatility_at_spot:.4f}")
    
    # Initialize solver
    solver = QNVSolver(params)
    
    # Get method recommendation
    selector = AdaptiveSolverSelector(params)
    recommendation = selector.recommend_method()
    
    print(f"\nRecommended Solution Method: {recommendation['primary']}")
    print(f"Reason: {recommendation['reason']}")
    
    # Display Heun parameters
    heun_params = solver.heun_params
    print(f"\nConfluent Heun Parameters:")
    print(f"  γ = {heun_params.gamma:.6f}")
    print(f"  δ = {heun_params.delta:.6f}")
    print(f"  ε = {heun_params.epsilon:.6f}")
    print(f"  α_H = {heun_params.alpha_H:.6f}")
    print(f"  q = {heun_params.q:.6f}")
    print(f"  Asymmetry δ-ε = {heun_params.asymmetry():.6f}")
    
    # Test different solution methods
    print(f"\n" + "="*40)
    print("SOLUTION METHOD COMPARISON")
    print("="*40)
    
    # Method 1: Heun Series Solution
    if recommendation['primary'] == 'heun_series':
        print("Testing Heun series solution...")
        try:
            y, solution = solver.solve_heun_series(y_max=0.5, n_terms=30)
            print(f"✓ Heun series: {len(y)} points, max value = {np.max(np.abs(solution)):.6f}")
        except Exception as e:
            print(f"✗ Heun series failed: {e}")
    
    # Method 2: PDE Finite Difference
    print("Testing PDE finite difference solver...")
    try:
        pde_solver = FiniteDifferenceSolver(params, S_min=50, S_max=150, n_points=200)
        
        # Price options for different strikes
        strikes = [90, 100, 110]
        T = 0.25  # 3 months
        
        print(f"\nEuropean Call Option Prices (T = {T:.2f} years):")
        print(f"{'Strike':<8} {'Price':<10} {'Volatility':<12}")
        print("-" * 35)
        
        for K in strikes:
            price_at_spot = pde_solver.get_price_at_spot(K=K, T=T)
            vol_at_strike = params.a*K**2 + params.b*K + params.c
            
            print(f"{K:<8} {price_at_spot:<10.4f} {vol_at_strike:<12.4f}")
            
        print(f"✓ PDE solver completed successfully")
        
    except Exception as e:
        print(f"✗ PDE solver failed: {e}")
    
    # Create comprehensive diagnostic plots
    print(f"\n" + "="*40)
    print("CREATING DIAGNOSTIC PLOTS")
    print("="*40)
    
    visualizer = QNVVisualizer(solver)
    fig = visualizer.create_diagnostic_dashboard()
    
    # Save the plot
    plt.savefig('/home/joelasaucedo/Development/x_conflheun/qnv_usage_example.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Diagnostic dashboard saved as 'qnv_usage_example.png'")
    
    # Additional analysis: Parameter sensitivity
    print(f"\n" + "="*40)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*40)
    
    base_params = params
    variations = {
        'a': [0.5*base_params.a, base_params.a, 1.5*base_params.a],
        'b': [0.5*base_params.b, base_params.b, 1.5*base_params.b],
        'c': [0.5*base_params.c, base_params.c, 1.5*base_params.c],
        'r': [0.5*base_params.r, base_params.r, 1.5*base_params.r]
    }
    
    print("Volatility at spot for parameter variations:")
    for param_name, values in variations.items():
        print(f"\n{param_name} variations:")
        for val in values:
            temp_params = QNVParameters(
                a=val if param_name == 'a' else base_params.a,
                b=val if param_name == 'b' else base_params.b,
                c=val if param_name == 'c' else base_params.c,
                r=val if param_name == 'r' else base_params.r,
                S0=base_params.S0,
                F=base_params.F
            )
            print(f"  {param_name}={val:.6f}: σ(S₀)={temp_params.volatility_at_spot:.4f}")
    
    print(f"\n" + "="*60)
    print("USAGE EXAMPLE COMPLETED")
    print("="*60)

def example_volatility_smile_analysis():
    """Example: Analyze volatility smile structure"""
    
    print("\n" + "="*60)
    print("VOLATILITY SMILE ANALYSIS EXAMPLE")
    print("="*60)
    
    # Create different QNV configurations
    configs = {
        'Symmetric': QNVParameters(a=0.00001, b=0.01, c=0.2, r=0.0, S0=100, F=100),
        'Skewed': QNVParameters(a=0.00001, b=0.02, c=0.18, r=0.02, S0=100, F=100),
        'Smile': QNVParameters(a=0.00002, b=0.005, c=0.19, r=0.01, S0=100, F=100)
    }
    
    # Plot volatility smiles
    plt.figure(figsize=(12, 8))
    
    strikes = np.linspace(80, 120, 50)
    
    for name, params in configs.items():
        vols = [params.a*K**2 + params.b*K + params.c for K in strikes]
        plt.plot(strikes, vols, label=f'{name} (Δ={params.discriminant:.6f})', linewidth=2)
    
    plt.axvline(100, color='k', linestyle='--', alpha=0.7, label='Spot Price')
    plt.xlabel('Strike Price')
    plt.ylabel('Volatility')
    plt.title('QNV Volatility Smile Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('/home/joelasaucedo/Development/x_conflheun/qnv_smile_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Volatility smile comparison saved as 'qnv_smile_comparison.png'")
    
    # Analyze each configuration
    for name, params in configs.items():
        print(f"\n{name} Configuration:")
        print(f"  σ(S) = {params.a:.6f}S² + {params.b:.3f}S + {params.c:.3f}")
        print(f"  Discriminant Δ = {params.discriminant:.6f}")
        print(f"  Volatility at spot: {params.volatility_at_spot:.4f}")
        
        solver = QNVSolver(params)
        heun_params = solver.heun_params
        print(f"  Heun asymmetry δ-ε = {heun_params.asymmetry():.6f}")

if __name__ == "__main__":
    example_option_pricing()
    example_volatility_smile_analysis()
