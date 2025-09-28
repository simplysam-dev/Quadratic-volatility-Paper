#!/usr/bin/env python3
"""
ENHANCED QNV SOLVER USAGE EXAMPLE
=================================

This example demonstrates the production-grade QNV solver with:
- Multiple solution methods
- Comprehensive diagnostics
- Volatility smile analysis
- Parameter sensitivity studies
- Real-world option pricing scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
from qnv_solver import QNVParameters, AdaptiveSolver, QNVVisualizer, QNVSolver
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_basic_option_pricing():
    """Example 1: Basic option pricing with QNV model"""
    print("\n" + "="*60)
    print("EXAMPLE 1: BASIC OPTION PRICING")
    print("="*60)
    
    # Define QNV parameters for a realistic market scenario
    params = QNVParameters(
        a=0.00001,    # Small quadratic term (weak smile)
        b=0.01,       # Linear term
        c=0.2,        # Base volatility level
        r=0.02,       # 2% risk-free rate
        S0=100,       # Spot price
        F=100,        # Forward price
        T=0.25        # 3 months to maturity
    )
    
    print(f"QNV Model Parameters:")
    print(f"  Spot Price (S₀): {params.S0}")
    print(f"  Risk-free Rate (r): {params.r:.1%}")
    print(f"  Time to Maturity (T): {params.T:.2f} years")
    print(f"  Discriminant (Δ): {params.discriminant:.6f}")
    print(f"  ATM Volatility: {params.sigma_atm:.3f}")
    
    # Create solver
    solver = AdaptiveSolver(params)
    
    # Price options across different strikes
    strikes = np.array([80, 90, 100, 110, 120])
    print(f"\nOption Pricing Results:")
    print(f"{'Strike':<8} {'Price':<10} {'Local Vol':<12} {'Method':<15}")
    print("-" * 50)
    
    for K in strikes:
        try:
            price = solver.solve(K)
            local_vol = params.volatility_at_strike(K)
            method = solver._recommend_method()
            print(f"{K:<8.0f} {price:<10.4f} {local_vol:<12.4f} {method:<15}")
        except Exception as e:
            print(f"{K:<8.0f} {'ERROR':<10} {'ERROR':<12} {'ERROR':<15}")
    
    return params, solver

def example_2_volatility_smile_analysis():
    """Example 2: Comprehensive volatility smile analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 2: VOLATILITY SMILE ANALYSIS")
    print("="*60)
    
    # Create parameters with strong volatility smile
    params = QNVParameters(
        a=0.0001,     # Larger quadratic term (stronger smile)
        b=-0.005,     # Negative linear term (skew)
        c=0.25,       # Higher base volatility
        r=0.03,       # 3% risk-free rate
        S0=100,       # Spot price
        F=100,        # Forward price
        T=0.5         # 6 months to maturity
    )
    
    print(f"Strong Smile Parameters:")
    print(f"  a = {params.a:.2e} (quadratic coefficient)")
    print(f"  b = {params.b:.3f} (linear coefficient)")
    print(f"  c = {params.c:.3f} (constant coefficient)")
    print(f"  ATM Volatility: {params.sigma_atm:.3f}")
    
    # Create visualizer for comprehensive analysis
    visualizer = QNVVisualizer(params)
    
    # Generate strikes for smile analysis
    strikes = np.linspace(70, 130, 20)
    
    # Compute prices and implied volatilities
    prices = []
    local_vols = []
    implied_vols = []
    
    for K in strikes:
        try:
            price = visualizer.adaptive_solver.solve(K)
            local_vol = params.volatility_at_strike(K)
            implied_vol = visualizer._compute_implied_volatility(K, price)
            
            prices.append(price)
            local_vols.append(local_vol)
            implied_vols.append(implied_vol)
        except:
            prices.append(np.nan)
            local_vols.append(np.nan)
            implied_vols.append(np.nan)
    
    # Create volatility smile plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Local volatility smile
    ax1.plot(strikes, local_vols, 'r-', linewidth=2, label='Local Volatility')
    ax1.axvline(params.S0, color='k', linestyle='--', alpha=0.7, label='Spot')
    ax1.set_xlabel('Strike K')
    ax1.set_ylabel('Volatility')
    ax1.set_title('Local Volatility Smile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Implied volatility smile
    ax2.plot(strikes, implied_vols, 'b-', linewidth=2, label='Implied Volatility')
    ax2.axvline(params.S0, color='k', linestyle='--', alpha=0.7, label='Spot')
    ax2.set_xlabel('Strike K')
    ax2.set_ylabel('Implied Volatility')
    ax2.set_title('Implied Volatility Smile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/joelasaucedo/Development/Quadratic-volatility-Paper/solver/volatility_smile_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Volatility smile analysis saved as volatility_smile_analysis.png")
    
    return params, visualizer

def example_3_parameter_sensitivity():
    """Example 3: Parameter sensitivity analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 3: PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Base parameters
    base_params = QNVParameters(
        a=0.00005, b=0.01, c=0.2, r=0.02, S0=100, F=100, T=0.25
    )
    
    K = 100  # ATM option
    
    # Sensitivity to parameter 'a' (quadratic coefficient)
    a_values = np.linspace(0.5 * base_params.a, 2.0 * base_params.a, 10)
    prices_a = []
    
    print("Sensitivity to quadratic coefficient (a):")
    print(f"{'a':<12} {'Price':<10} {'ATM Vol':<10}")
    print("-" * 35)
    
    for a in a_values:
        try:
            temp_params = QNVParameters(a=a, b=base_params.b, c=base_params.c,
                                      r=base_params.r, S0=base_params.S0, F=base_params.F, T=base_params.T)
            temp_solver = AdaptiveSolver(temp_params)
            price = temp_solver.solve(K)
            atm_vol = temp_params.sigma_atm
            
            prices_a.append(price)
            print(f"{a:<12.2e} {price:<10.4f} {atm_vol:<10.4f}")
        except:
            prices_a.append(np.nan)
            print(f"{a:<12.2e} {'ERROR':<10} {'ERROR':<10}")
    
    # Sensitivity to parameter 'b' (linear coefficient)
    b_values = np.linspace(0.5 * base_params.b, 2.0 * base_params.b, 10)
    prices_b = []
    
    print(f"\nSensitivity to linear coefficient (b):")
    print(f"{'b':<12} {'Price':<10} {'ATM Vol':<10}")
    print("-" * 35)
    
    for b in b_values:
        try:
            temp_params = QNVParameters(a=base_params.a, b=b, c=base_params.c,
                                      r=base_params.r, S0=base_params.S0, F=base_params.F, T=base_params.T)
            temp_solver = AdaptiveSolver(temp_params)
            price = temp_solver.solve(K)
            atm_vol = temp_params.sigma_atm
            
            prices_b.append(price)
            print(f"{b:<12.3f} {price:<10.4f} {atm_vol:<10.4f}")
        except:
            prices_b.append(np.nan)
            print(f"{b:<12.3f} {'ERROR':<10} {'ERROR':<10}")
    
    # Create sensitivity plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sensitivity to 'a'
    ax1.plot(a_values, prices_a, 'ro-', linewidth=2, markersize=6)
    ax1.set_xlabel('Parameter a')
    ax1.set_ylabel('ATM Call Price')
    ax1.set_title('Sensitivity to Quadratic Coefficient (a)')
    ax1.grid(True, alpha=0.3)
    
    # Sensitivity to 'b'
    ax2.plot(b_values, prices_b, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Parameter b')
    ax2.set_ylabel('ATM Call Price')
    ax2.set_title('Sensitivity to Linear Coefficient (b)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/joelasaucedo/Development/Quadratic-volatility-Paper/solver/parameter_sensitivity.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Parameter sensitivity analysis saved as parameter_sensitivity.png")
    
    return base_params

def example_4_time_decay_analysis():
    """Example 4: Time decay analysis for different strikes"""
    print("\n" + "="*60)
    print("EXAMPLE 4: TIME DECAY ANALYSIS")
    print("="*60)
    
    # Base parameters
    params = QNVParameters(
        a=0.00005, b=0.01, c=0.2, r=0.02, S0=100, F=100, T=1.0
    )
    
    # Time points from 1 year to 1 day
    T_values = np.linspace(0.01, 1.0, 20)
    strikes = [90, 100, 110]  # OTM, ATM, ITM
    
    print("Time Decay Analysis:")
    print(f"{'Time':<8} {'OTM(90)':<10} {'ATM(100)':<10} {'ITM(110)':<10}")
    print("-" * 45)
    
    # Store results for plotting
    results = {K: [] for K in strikes}
    
    for T in T_values:
        temp_params = QNVParameters(a=params.a, b=params.b, c=params.c,
                                  r=params.r, S0=params.S0, F=params.F, T=T)
        temp_solver = AdaptiveSolver(temp_params)
        
        prices_row = []
        for K in strikes:
            try:
                price = temp_solver.solve(K)
                results[K].append(price)
                prices_row.append(price)
            except:
                results[K].append(np.nan)
                prices_row.append(np.nan)
        
        print(f"{T:<8.2f} {prices_row[0]:<10.4f} {prices_row[1]:<10.4f} {prices_row[2]:<10.4f}")
    
    # Create time decay plot
    plt.figure(figsize=(10, 6))
    
    for K in strikes:
        plt.plot(T_values, results[K], 'o-', linewidth=2, markersize=4, label=f'Strike {K}')
    
    plt.xlabel('Time to Maturity (years)')
    plt.ylabel('Call Price')
    plt.title('Time Decay Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()  # Reverse x-axis to show decay
    
    plt.tight_layout()
    plt.savefig('/home/joelasaucedo/Development/Quadratic-volatility-Paper/solver/time_decay_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Time decay analysis saved as time_decay_analysis.png")
    
    return params

def example_5_comprehensive_dashboard():
    """Example 5: Create comprehensive diagnostic dashboard"""
    print("\n" + "="*60)
    print("EXAMPLE 5: COMPREHENSIVE DIAGNOSTIC DASHBOARD")
    print("="*60)
    
    # Create parameters for comprehensive analysis
    params = QNVParameters(
        a=0.00008,    # Moderate quadratic term
        b=-0.002,     # Slight negative skew
        c=0.22,       # Realistic base volatility
        r=0.025,      # 2.5% risk-free rate
        S0=100,       # Spot price
        F=100,        # Forward price
        T=0.3         # 3.6 months to maturity
    )
    
    print(f"Comprehensive Analysis Parameters:")
    print(f"  a = {params.a:.2e} (quadratic)")
    print(f"  b = {params.b:.3f} (linear)")
    print(f"  c = {params.c:.3f} (constant)")
    print(f"  r = {params.r:.1%} (risk-free rate)")
    print(f"  T = {params.T:.2f} years (maturity)")
    print(f"  Δ = {params.discriminant:.6f} (discriminant)")
    print(f"  σ_ATM = {params.sigma_atm:.3f} (ATM volatility)")
    
    # Create visualizer
    visualizer = QNVVisualizer(params)
    
    # Generate strikes for analysis
    strikes = [80, 90, 100, 110, 120]
    
    # Create comprehensive dashboard
    print(f"\nCreating comprehensive diagnostic dashboard...")
    fig = visualizer.create_comprehensive_dashboard(strikes)
    
    plt.savefig('/home/joelasaucedo/Development/Quadratic-volatility-Paper/solver/comprehensive_dashboard.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comprehensive dashboard saved as comprehensive_dashboard.png")
    
    # Display Heun parameters
    solver = QNVSolver(params)
    heun_params = solver.heun_params
    
    print(f"\nHeun Equation Parameters:")
    print(f"  γ = {heun_params.gamma:.6f}")
    print(f"  δ = {heun_params.delta:.6f}")
    print(f"  ε = {heun_params.epsilon:.6f}")
    print(f"  α_H = {heun_params.alpha_H:.6f}")
    print(f"  q = {heun_params.q:.6f}")
    print(f"  Asymmetry δ-ε = {heun_params.asymmetry():.6f}")
    
    # Test polynomial case detection
    poly_degree = solver.check_polynomial_case()
    if poly_degree is not None:
        print(f"✓ Polynomial solution detected (degree {poly_degree})")
    else:
        print("✗ No polynomial solution detected")
    
    return params, visualizer

def main():
    """Run all examples"""
    print("QNV SOLVER - ENHANCED USAGE EXAMPLES")
    print("====================================")
    
    try:
        # Run all examples
        params1, solver1 = example_1_basic_option_pricing()
        params2, visualizer2 = example_2_volatility_smile_analysis()
        params3 = example_3_parameter_sensitivity()
        params4 = example_4_time_decay_analysis()
        params5, visualizer5 = example_5_comprehensive_dashboard()
        
        print(f"\n{'='*60}")
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Generated files:")
        print(f"  - volatility_smile_analysis.png")
        print(f"  - parameter_sensitivity.png")
        print(f"  - time_decay_analysis.png")
        print(f"  - comprehensive_dashboard.png")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        logger.error("Example execution failed", exc_info=True)

if __name__ == "__main__":
    main()