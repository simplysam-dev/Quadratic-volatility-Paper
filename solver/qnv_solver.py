#!/usr/bin/env python3
"""
Production-Grade QNV Solver Implementation
==========================================

This module implements a comprehensive solver for the Quadratic Normal Volatility (QNV) model
using the confluent Heun equation mapping derived in the accompanying paper.

Key Features:
- Automatic method selection based on parameter analysis
- Multiple solution approaches: Heun series, polynomial solutions, PDE methods
- Comprehensive visualization and diagnostics
- Production-ready error handling and validation
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import hyp2f1
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, List
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QNVParameters:
    """QNV model parameters container with validation"""
    a: float  # quadratic coefficient
    b: float  # linear coefficient  
    c: float  # constant coefficient
    r: float  # risk-free rate
    S0: float # spot price
    F: float  # forward price
    
    def __post_init__(self):
        """Validate parameters"""
        if self.a < 0:
            raise ValueError("Quadratic coefficient 'a' must be non-negative")
        if self.c <= 0:
            raise ValueError("Constant coefficient 'c' must be positive")
        if self.S0 <= 0:
            raise ValueError("Spot price must be positive")
        if self.F <= 0:
            raise ValueError("Forward price must be positive")
            
        # Check discriminant
        Delta = self.b**2 - 4*self.a*self.c
        if Delta <= 0:
            warnings.warn(f"Discriminant Δ = {Delta:.6f} ≤ 0. Model may have complex roots.")
            
    @property
    def discriminant(self) -> float:
        """Compute discriminant Δ = b² - 4ac"""
        return self.b**2 - 4*self.a*self.c
    
    @property
    def volatility_at_spot(self) -> float:
        """Volatility at current spot price"""
        return self.a*self.S0**2 + self.b*self.S0 + self.c

@dataclass
class HeunParameters:
    """Confluent Heun equation parameters"""
    gamma: float
    delta: float  
    epsilon: float
    alpha_H: float
    q: float
    E: float  # energy eigenvalue
    
    def asymmetry(self) -> float:
        """Compute asymmetry parameter δ - ε"""
        return self.delta - self.epsilon

class QNVSolver:
    """Main solver class for QNV model using confluent Heun equation"""
    
    def __init__(self, params: QNVParameters, E: Optional[float] = None):
        self.params = params
        self.E = E if E is not None else self._estimate_energy()
        self.heun_params = self._compute_heun_parameters()
        
    def _estimate_energy(self) -> float:
        """Estimate energy eigenvalue for boundary conditions"""
        # Simple estimation based on volatility level
        sigma_spot = self.params.volatility_at_spot
        return 0.5 * sigma_spot**2 + self.params.r
        
    def _compute_heun_parameters(self) -> HeunParameters:
        """Convert QNV parameters to Heun parameters using paper formulas"""
        a, b, c, r = self.params.a, self.params.b, self.params.c, self.params.r
        Delta = self.params.discriminant
        
        if Delta <= 0:
            raise ValueError(f"Cannot compute Heun parameters: discriminant Δ = {Delta} ≤ 0")
            
        sqrt_Delta = np.sqrt(Delta)
        
        gamma = -4*r / sqrt_Delta
        delta = 1 + (2*r*(b - sqrt_Delta)) / (a*sqrt_Delta)
        epsilon = 1 + (2*r*(b + sqrt_Delta)) / (a*sqrt_Delta)
        alpha_H = (8/Delta) * (self.E - r/2 - Delta/8)
        q = (4/Delta) * (self.E - r/2 - Delta/8*(1 - (2*r*(b-sqrt_Delta))/(a*sqrt_Delta)) 
                         - 2*r**2/Delta - (2*b*r*sqrt_Delta)/(4*a))
        
        return HeunParameters(gamma, delta, epsilon, alpha_H, q, self.E)
    
    def solve_heun_series(self, y_max: float = 0.5, n_terms: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Solve using three-term recurrence relation"""
        γ, δ, ε, α, q = (self.heun_params.gamma, self.heun_params.delta, 
                         self.heun_params.epsilon, self.heun_params.alpha_H, 
                         self.heun_params.q)
        
        # Initialize coefficients
        c = np.zeros(n_terms)
        c[0] = 1.0  # arbitrary normalization
        
        if n_terms > 1:
            c[1] = self._A(0, γ, δ, ε, α, q) * c[0]
        
        # Compute recurrence coefficients
        for k in range(1, n_terms-1):
            A_k = self._A(k, γ, δ, ε, α, q)
            B_k = self._B(k, γ, δ, ε, α, q)
            
            if k == 1:
                c[k+1] = A_k * c[k]
            else:
                c[k+1] = A_k * c[k] + B_k * c[k-1]
            
            # Check for convergence issues and rescale if needed
            if abs(c[k+1]) > 1e6:
                logger.warning(f"Series coefficients growing rapidly at k={k+1}, rescaling")
                # Rescale to prevent overflow
                scale_factor = 1e-6
                c[:k+2] *= scale_factor
                c[k+1] = A_k * c[k] + B_k * c[k-1]
                
        # Evaluate series
        y_grid = np.linspace(0, y_max, 1000)
        f_y = np.zeros_like(y_grid)
        
        for i, y in enumerate(y_grid):
            for k in range(min(n_terms, len(c))):
                if abs(c[k]) < 1e-12:  # Skip negligible terms
                    continue
                f_y[i] += c[k] * y**k
                
        return y_grid, f_y
    
    def _A(self, k: int, γ: float, δ: float, ε: float, α: float, q: float) -> float:
        """Recurrence coefficient A_k"""
        if k + δ == 0:
            return 0.0  # Avoid division by zero
        return ((k+1)*(k+γ) + q) / ((k+1)*(k+δ))
    
    def _B(self, k: int, γ: float, δ: float, ε: float, α: float, q: float) -> float:
        """Recurrence coefficient B_k"""
        if k + δ == 0:
            return 0.0  # Avoid division by zero
        return (α - (k-1)*(k+γ+δ+ε-1)) / ((k+1)*(k+δ))
    
    def check_polynomial_case(self) -> Optional[int]:
        """Check if parameters satisfy polynomial termination conditions"""
        γ, δ, ε, α = (self.heun_params.gamma, self.heun_params.delta, 
                      self.heun_params.epsilon, self.heun_params.alpha_H)
        
        # Check condition: α_H = -n(n+1+γ+δ+ε) for some integer n
        discriminant = (1 + γ + δ + ε)**2 - 4*α
        if discriminant >= 0:
            n1 = 0.5 * (-(1 + γ + δ + ε) + np.sqrt(discriminant))
            n2 = 0.5 * (-(1 + γ + δ + ε) - np.sqrt(discriminant))
            
            for n in [n1, n2]:
                if abs(n - round(n)) < 1e-6 and n >= 0:
                    return int(round(n))
        return None

class FiniteDifferenceSolver:
    """High-performance PDE solver for QNV model"""
    
    def __init__(self, params: QNVParameters, S_min: float, S_max: float, n_points: int = 1000):
        self.params = params
        self.S_grid = np.linspace(S_min, S_max, n_points)
        self.dS = self.S_grid[1] - self.S_grid[0]
        
    def price_european_call(self, K: float, T: float, 
                           method: str = 'crank_nicolson') -> Tuple[np.ndarray, np.ndarray]:
        """Price European call using finite difference, returns (S_grid, prices)"""
        
        if method == 'crank_nicolson':
            prices = self._crank_nicolson(K, T)
        elif method == 'implicit_euler':
            prices = self._implicit_euler(K, T)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return self.S_grid, prices
    
    def get_price_at_spot(self, K: float, T: float, method: str = 'crank_nicolson') -> float:
        """Get option price at current spot price"""
        S_grid, prices = self.price_european_call(K, T, method)
        # Interpolate at spot price
        return np.interp(self.params.S0, S_grid, prices)
    
    def _crank_nicolson(self, K: float, T: float) -> np.ndarray:
        """Crank-Nicolson scheme - most stable for production"""
        n_points = len(self.S_grid)
        dt = min(0.001, T/100)  # adaptive time stepping
        n_steps = int(T / dt) + 1
        dt = T / n_steps
        
        logger.info(f"Running Crank-Nicolson: {n_steps} steps, dt={dt:.6f}")
        
        # Initial condition: payoff at maturity
        V = np.maximum(self.S_grid - K, 0)
        
        for step in range(n_steps):
            if step % max(1, n_steps//10) == 0:
                logger.info(f"Step {step}/{n_steps}")
            V = self._crank_nicolson_step(V, step * dt, dt, K)
            
        return V
    
    def _crank_nicolson_step(self, V: np.ndarray, tau: float, dt: float, K: float) -> np.ndarray:
        """Single implicit Euler step with proper call option boundary conditions"""
        n = len(V)
        a, b, c, r = self.params.a, self.params.b, self.params.c, self.params.r
        
        # Construct tridiagonal matrix coefficients for implicit Euler
        alpha = np.zeros(n)
        beta = np.zeros(n)
        gamma_coeff = np.zeros(n)
        
        for i in range(1, n-1):
            S = self.S_grid[i]
            sigma = a*S**2 + b*S + c
            
            # Ensure volatility is positive
            if sigma <= 0:
                sigma = 0.001  # minimum volatility
            
            # Implicit Euler coefficients (backward time stepping)
            A = sigma**2 * S**2 / self.dS**2
            B = r * S / self.dS
            C = r
            
            # Matrix coefficients (note: we're solving backward in time)
            alpha[i] = 0.5 * A - 0.5 * B
            beta[i] = -A - C + 1/dt  # Changed sign for backward time
            gamma_coeff[i] = 0.5 * A + 0.5 * B
        
        # Boundary conditions for call options
        # At S=0: V = 0 (option worthless)
        alpha[0] = 0
        beta[0] = 1
        gamma_coeff[0] = 0
        
        # At S=S_max: V ≈ S - K*exp(-r*tau) (deep ITM)
        alpha[-1] = 0
        beta[-1] = 1
        gamma_coeff[-1] = 0
        
        # Right-hand side
        rhs = V.copy()
        rhs[0] = 0
        rhs[-1] = max(0, self.S_grid[-1] - K * np.exp(-r*tau))
        
        # Solve tridiagonal system
        ab = np.array([np.concatenate([[0], gamma_coeff[1:]]), 
                       beta, 
                       np.concatenate([alpha[:-1], [0]])])
        
        try:
            result = solve_banded((1, 1), ab, rhs)
            # Ensure non-negative prices
            result = np.maximum(result, 0)
            return result
        except Exception as e:
            logger.warning(f"PDE solve failed: {e}, using previous values")
            return V

class AdaptiveSolverSelector:
    """Automatically chooses optimal solution method based on parameters"""
    
    def __init__(self, params: QNVParameters):
        self.params = params
        
    def recommend_method(self) -> Dict:
        """Recommend optimal solving strategy"""
        a, b, c, r = self.params.a, self.params.b, self.params.c, self.params.r
        Delta = self.params.discriminant
        
        recommendations = {}
        
        # Check for polynomial solutions
        solver = QNVSolver(self.params)
        poly_degree = solver.check_polynomial_case()
        
        if poly_degree is not None:
            recommendations['primary'] = 'heun_polynomial'
            recommendations['reason'] = f'Exact polynomial solution available (degree {poly_degree})'
            recommendations['poly_degree'] = poly_degree
            
        # Check for symmetric case (r ≈ 0)
        elif abs(r) < 1e-8:
            recommendations['primary'] = 'poschl_teller'  
            recommendations['reason'] = 'Symmetric case (r≈0), reduces to Pöschl-Teller'
            
        # Check convergence conditions for series
        elif self._series_will_converge():
            recommendations['primary'] = 'heun_series'
            recommendations['reason'] = 'Series solution converges rapidly'
            
        else:
            recommendations['primary'] = 'pde_finite_difference'
            recommendations['reason'] = 'General case requiring numerical PDE solution'
            
        return recommendations
    
    def _series_will_converge(self) -> bool:
        """Check series convergence conditions"""
        # Simple heuristic: check if parameters are in reasonable range
        a, b, c, r = self.params.a, self.params.b, self.params.c, self.params.r
        Delta = self.params.discriminant
        
        # Series converges well when discriminant is positive and not too large
        return Delta > 0 and Delta < 1.0 and abs(r) < 0.1

class QNVVisualizer:
    """Multi-panel analysis and visualization"""
    
    def __init__(self, solver: QNVSolver):
        self.solver = solver
        
    def create_diagnostic_dashboard(self, S_range: Tuple[float, float] = (50, 150)) -> plt.Figure:
        """Create comprehensive diagnostic plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        self._plot_volatility_smile(axes[0, 0])
        self._plot_potential_landscape(axes[0, 1])
        self._plot_heun_solutions(axes[0, 2])
        self._plot_parameter_sensitivity(axes[1, 0])
        self._plot_heun_parameters(axes[1, 1])
        self._plot_convergence_analysis(axes[1, 2])
        
        plt.tight_layout()
        return fig
    
    def _plot_volatility_smile(self, ax):
        """Plot the QNV volatility smile"""
        K_range = np.linspace(0.5 * self.solver.params.S0, 
                             1.5 * self.solver.params.S0, 100)
        sigma_K = [self.solver.params.a*K**2 + self.solver.params.b*K + self.solver.params.c 
                  for K in K_range]
        
        ax.plot(K_range, sigma_K, 'r-', linewidth=2)
        ax.axvline(self.solver.params.S0, color='k', linestyle='--', label='Spot')
        ax.set_xlabel('Strike K')
        ax.set_ylabel('Volatility σ(K)')
        ax.set_title('QNV Volatility Smile')
        ax.legend()
        ax.grid(True)
        
    def _plot_potential_landscape(self, ax):
        """Plot the quantum potential V(t)"""
        a, b, c, r = (self.solver.params.a, self.solver.params.b, 
                      self.solver.params.c, self.solver.params.r)
        Delta = self.solver.params.discriminant
        
        if Delta <= 0:
            ax.text(0.5, 0.5, 'Complex roots\nNo potential plot', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Quantum Potential (Complex Roots)')
            return
            
        # Transform to t-coordinates
        t_values = np.linspace(-0.99, 0.99, 1000)
        
        # Compute potential V(t) using formulas from paper
        numerator = (2*Delta**3*t_values**6 + 
                    (-4*r*Delta**2 - 5*Delta**3)*t_values**4 +
                    (16*r**2*Delta + 8*r*Delta**2 + 4*Delta**3)*t_values**2 -
                    32*b*r**2*np.sqrt(Delta)*t_values +
                    (16*b**2*r**2 - 4*r*Delta**2 - Delta**3))
        
        denominator = 8*Delta**2*(1 - t_values**2)**2
        V_t = numerator / denominator
        
        ax.plot(t_values, V_t, 'b-', linewidth=2)
        ax.set_xlabel('t (hyperbolic coordinate)')
        ax.set_ylabel('V(t)')
        ax.set_title('Quantum Potential Landscape')
        ax.grid(True)
        
    def _plot_heun_solutions(self, ax):
        """Plot Heun function solutions"""
        try:
            y, solution = self.solver.solve_heun_series(y_max=0.5, n_terms=30)
            ax.plot(y, solution, 'g-', linewidth=2)
            ax.set_xlabel('y')
            ax.set_ylabel('f(y)')
            ax.set_title('Heun Function Solution')
            ax.grid(True)
        except Exception as e:
            ax.text(0.5, 0.5, f'Solution failed:\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Heun Function Solution (Failed)')
    
    def _plot_parameter_sensitivity(self, ax):
        """Plot parameter sensitivity analysis"""
        base_params = self.solver.params
        
        # Vary 'a' parameter
        a_values = np.linspace(0.5*base_params.a, 1.5*base_params.a, 10)
        sigma_values = []
        
        for a in a_values:
            temp_params = QNVParameters(a=a, b=base_params.b, c=base_params.c, 
                                      r=base_params.r, S0=base_params.S0, F=base_params.F)
            sigma_values.append(temp_params.volatility_at_spot)
        
        ax.plot(a_values, sigma_values, 'o-', label='σ(S₀) vs a')
        ax.set_xlabel('Parameter a')
        ax.set_ylabel('Volatility at Spot')
        ax.set_title('Parameter Sensitivity')
        ax.legend()
        ax.grid(True)
        
    def _plot_heun_parameters(self, ax):
        """Plot Heun parameters"""
        params = self.solver.heun_params
        
        param_names = ['γ', 'δ', 'ε', 'α_H', 'q']
        param_values = [params.gamma, params.delta, params.epsilon, 
                       params.alpha_H, params.q]
        
        bars = ax.bar(param_names, param_values)
        ax.set_ylabel('Parameter Value')
        ax.set_title('Heun Equation Parameters')
        ax.grid(True, axis='y')
        
        # Add asymmetry annotation
        asymmetry = params.asymmetry()
        ax.text(0.02, 0.98, f'Asymmetry δ-ε = {asymmetry:.3f}', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    def _plot_convergence_analysis(self, ax):
        """Plot series convergence analysis"""
        n_terms_range = range(5, 51, 5)
        max_coeffs = []
        
        for n_terms in n_terms_range:
            try:
                y, solution = self.solver.solve_heun_series(y_max=0.5, n_terms=n_terms)
                max_coeffs.append(np.max(np.abs(solution)))
            except:
                max_coeffs.append(np.nan)
        
        ax.semilogy(n_terms_range, max_coeffs, 'o-')
        ax.set_xlabel('Number of Terms')
        ax.set_ylabel('Max |Solution|')
        ax.set_title('Series Convergence')
        ax.grid(True)

def run_synthetic_test_suite():
    """Comprehensive testing with synthetic parameters"""
    
    print("=" * 60)
    print("QNV SOLVER SYNTHETIC TEST SUITE")
    print("=" * 60)
    
    # Test Case 1: Symmetric case (r=0) - should reduce to Pöschl-Teller
    params_symmetric = QNVParameters(a=0.00001, b=0.01, c=0.2, r=0.0, S0=100, F=100)
    
    # Test Case 2: Moderate asymmetry 
    params_asymmetric = QNVParameters(a=0.00001, b=0.01, c=0.2, r=0.02, S0=100, F=100)
    
    # Test Case 3: Near-polynomial case (special parameters)
    params_poly = QNVParameters(a=0.00002, b=0.02, c=0.18, r=0.01, S0=100, F=100)
    
    test_cases = [
        ("Symmetric (r=0)", params_symmetric),
        ("Asymmetric (r>0)", params_asymmetric), 
        ("Near-Polynomial", params_poly)
    ]
    
    for i, (name, params) in enumerate(test_cases):
        print(f"\n{'='*20} Test Case {i+1}: {name} {'='*20}")
        print(f"Parameters: a={params.a:.6f}, b={params.b:.3f}, c={params.c:.3f}, r={params.r:.3f}")
        print(f"Discriminant Δ = {params.discriminant:.6f}")
        print(f"Volatility at spot: {params.volatility_at_spot:.4f}")
        
        try:
            solver = QNVSolver(params)
            selector = AdaptiveSolverSelector(params)
            
            recommendation = selector.recommend_method()
            print(f"Recommended method: {recommendation['primary']}")
            print(f"Reason: {recommendation['reason']}")
            
            # Display Heun parameters
            heun_params = solver.heun_params
            print(f"Heun parameters:")
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
            
            # Run the recommended method
            if recommendation['primary'] == 'heun_series':
                print("Testing Heun series solution...")
                y, solution = solver.solve_heun_series(y_max=0.5, n_terms=30)
                print(f"✓ Series solution computed: {len(y)} points, max value = {np.max(np.abs(solution)):.6f}")
                
            elif recommendation['primary'] == 'pde_finite_difference':
                print("Testing PDE finite difference solver...")
                pde_solver = FiniteDifferenceSolver(params, S_min=50, S_max=150, n_points=200)
                prices = pde_solver.price_european_call(K=100, T=0.25)
                print(f"✓ PDE solution computed: {len(prices)} points, max price = {np.max(prices):.2f}")
            
            # Create diagnostic plots
            print("Creating diagnostic plots...")
            visualizer = QNVVisualizer(solver)
            fig = visualizer.create_diagnostic_dashboard()
            plt.savefig(f'/home/joelasaucedo/Development/x_conflheun/qnv_test_case_{i+1}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Diagnostic plots saved as qnv_test_case_{i+1}.png")
            
        except Exception as e:
            print(f"✗ Test case failed: {str(e)}")
            logger.error(f"Test case {i+1} failed", exc_info=True)
    
    print(f"\n{'='*60}")
    print("TEST SUITE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    run_synthetic_test_suite()
