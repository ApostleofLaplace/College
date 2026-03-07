import numpy as np
import matplotlib.pyplot as plt

def kb_logspace(xmin, xmax, alpha=1.2):
    """
    Produces a set of logarithmically spaced points in [xmin,xmax].  Points 
    are related as:
            x[i+1] = alpha*x[i]
    """
    x_logspace = [xmin]
    x_current = alpha*xmin 
    while x_current < xmax:
        x_logspace.append(x_current)
        x_current = x_current*alpha 
    return np.array([int(x) for x in x_logspace])

# Part (a): Analytical solution

# dN/dt = -μN with N(0) = N0
# Solution: N(t) = N0 * exp(-μt)

# Part (b): How to calculate μ from results

# From N(T) = N0 * exp(-μT), we get:
# ln(N(T)/N0) = -μT
# Therefore: μ = -ln(N(T)/N0) / T

def calculate_mu_from_results(N0, NT, T):
    """Calculate decay constant from initial and final values."""
    return -np.log(NT / N0) / T

# Part (c) and (d): Euler method

# Euler step: N_{t+Δt} = N_t + Δt * dN/dt = N_t - μΔt * N_t = N_t(1 - μΔt)

def euler_integrate(mu, N0, T, M):
    """
    Integrate dN/dt = -μN using Euler method.
    
    Parameters:
    mu: decay constant
    N0: initial population
    T: total time
    M: number of steps
    
    Returns:
    times: array of time points
    N_values: array of population values at each time
    """
    dt = T / M
    times = np.linspace(0, T, M + 1)
    N_values = np.zeros(M + 1)
    N_values[0] = N0
    
    for i in range(M):
        N_values[i + 1] = N_values[i] * (1 - mu * dt)
    
    return times, N_values

# Part (d): Integrate with specific parameters
print("Part (d):")
mu = 0.8
N0 = 0.1
T = 4
M = 25
dt = T / M

times_d, N_values_d = euler_integrate(mu, N0, T, M)
mu_calculated_d = calculate_mu_from_results(N0, N_values_d[-1], T)

print(f"Input μ: {mu}")
print(f"Calculated μ from Euler integration (M={M}): {mu_calculated_d:.6f}")
print(f"Difference: {abs(mu - mu_calculated_d):.6f}")
print(f"Final value N(T): {N_values_d[-1]:.8f}")
print()

# Part (e): Analytical expression for NT

# From Euler method: N_{t+Δt} = N_t(1 - μΔt)
# Starting from N_0:
# N_Δt = N0(1 - μΔt)
# N_{2Δt} = N0(1 - μΔt)^2
# N_{MΔt} = N0(1 - μΔt)^M

print("Part (e):")
print("Analytical form from difference equation:")
print("N_M = N0 * (1 - μΔt)^M")
print(f"With our values: N_M = {N0} * (1 - {mu}*{dt})^{M}")
print(f"            = {N0} * ({1 - mu*dt})^{M}")
N_M_formula = N0 * (1 - mu * dt) ** M
print(f"            = {N_M_formula:.8f}")
print(f"From numerical integration: {N_values_d[-1]:.8f}")
print()

# Part (f): Convergence to true solution

print("Part (f):")
print("The Euler method gives: N_M = N0(1 - μΔt)^M = N0(1 - μT/M)^M")
print(f"As M → ∞, (1 - μT/M)^M → e^(-μT)")
print(f"This follows the definition e^x = lim_n→∞ (1 + x/n)^n")
print(f"With x = -μT, we get: e^(-μT) = lim_M→∞ (1 - μT/M)^M")
print()

# Part (g): Integration with M=250

print("Part (g):")
M_g = 250
times_g, N_values_g = euler_integrate(mu, N0, T, M_g)
mu_calculated_g = calculate_mu_from_results(N0, N_values_g[-1], T)

print(f"Input μ: {mu}")
print(f"Calculated μ from Euler integration (M={M_g}): {mu_calculated_g:.6f}")
print(f"Difference: {abs(mu - mu_calculated_g):.6f}")
print(f"Previous difference (M={M}): {abs(mu - mu_calculated_d):.6f}")
print(f"Improvement: The error reduced from {abs(mu - mu_calculated_d):.6f} to {abs(mu - mu_calculated_g):.6f}")
print()

# Part (h): Absolute error vs stepsize

print("Part (h):")

# True solution at T=4
N_true = N0 * np.exp(-mu * T)

# Calculate errors for different M values
M_values = kb_logspace(25, 25000, alpha=1.3)
absolute_errors = []

for M_val in M_values:
    _, N_vals = euler_integrate(mu, N0, T, M_val)
    abs_error = np.abs(N_vals[-1] - N_true)
    absolute_errors.append(abs_error)

absolute_errors = np.array(absolute_errors)

# Create double logarithmic plot
plt.figure(figsize=(10, 7))
plt.loglog(M_values, absolute_errors, 'bo-', linewidth=2, markersize=6)
plt.xlabel('Number of steps (M)', fontsize=12)
plt.ylabel('Absolute error in N(T)', fontsize=12)
plt.title('Absolute error in Euler method integration', fontsize=14)
plt.grid(True, which='both', alpha=0.3)

# Add theoretical line for first-order error (error ~ 1/M)
theoretical_errors = absolute_errors[0] * (M_values[0] / M_values)
plt.loglog(M_values, theoretical_errors, 'r--', linewidth=2, label='O(1/M) scaling')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('absolute_error_plot.png', dpi=150)
print(f"Plotted absolute error vs M for M = {M_values[0]} to {M_values[-1]}")
print(f"Number of points: {len(M_values)}")
print(f"Error at M=25: {absolute_errors[0]:.8f}")
print(f"Error at M=25000: {absolute_errors[-1]:.8f}")
print("The plot shows O(1/M) scaling, confirming first-order accuracy of Euler method.")
print()

# Part (i): Second-order Runge-Kutta (RK2)

def rk2_integrate(mu, N0, T, M):
    """
    Integrate dN/dt = -μN using second-order Runge-Kutta method.
    
    Parameters:
    mu: decay constant
    N0: initial population
    T: total time
    M: number of steps
    
    Returns:
    times: array of time points
    N_values: array of population values at each time
    """
    dt = T / M
    times = np.linspace(0, T, M + 1)
    N_values = np.zeros(M + 1)
    N_values[0] = N0
    
    for i in range(M):
        k1 = -mu * N_values[i]
        k2 = -mu * (N_values[i] + 0.5 * dt * k1)
        N_values[i + 1] = N_values[i] + dt * k2
    
    return times, N_values

print("Part (i):")
print("RK2 method implemented using midpoint method:")
print("k1 = -μ * N_i")
print("k2 = -μ * (N_i + 0.5 * Δt * k1)")
print("N_{i+1} = N_i + Δt * k2")
print()

# Part (j): Compare Euler and RK2

print("Part (j):")
print("=" * 50)

dt_j = 0.25
M_j = int(T / dt_j)

N_true_j = N0 * np.exp(-mu * T)

_, N_euler_j = euler_integrate(mu, N0, T, M_j)
_, N_rk2_j = rk2_integrate(mu, N0, T, M_j)

percent_error_euler_j = 100 * np.abs(N_true_j - N_euler_j[-1]) / np.abs(N_true_j)
percent_error_rk2_j = 100 * np.abs(N_true_j - N_rk2_j[-1]) / np.abs(N_true_j)

print(f"Stepsize Δt = {dt_j}")
print(f"Number of steps M = {M_j}")
print(f"True N(T) = {N_true_j:.8f}")
print()
print(f"Euler method:")
print(f"  N(T) = {N_euler_j[-1]:.8f}")
print(f"  Percent error = {percent_error_euler_j:.6f}%")
print()
print(f"RK2 method:")
print(f"  N(T) = {N_rk2_j[-1]:.8f}")
print(f"  Percent error = {percent_error_rk2_j:.6f}%")
print()
print(f"RK2 is {percent_error_euler_j / percent_error_rk2_j:.1f}x more accurate than Euler")
print()

# Part (k): Implicit Euler method

def implicit_euler_integrate(mu, N0, T, M):
    """
    Integrate dN/dt = -μN using implicit (backward) Euler method.
    For this linear ODE, the implicit solution is:
    N_{i+1} = N_i / (1 + μ*Δt)
    
    Parameters:
    mu: decay constant
    N0: initial population
    T: total time
    M: number of steps
    
    Returns:
    times: array of time points
    N_values: array of population values at each time
    """
    dt = T / M
    times = np.linspace(0, T, M + 1)
    N_values = np.zeros(M + 1)
    N_values[0] = N0
    
    for i in range(M):
        N_values[i + 1] = N_values[i] / (1 + mu * dt)
    
    return times, N_values

print("Part (k):")
print("Implicit Euler method:")
print("For dN/dt = -μN, the implicit Euler formula becomes:")
print("N_{i+1} = N_i / (1 + μ*Δt)")
print()

dt_k = 0.25
M_k = int(T / dt_k)

_, N_impl_euler_k = implicit_euler_integrate(mu, N0, T, M_k)
percent_error_impl_euler_k = 100 * np.abs(N_true_j - N_impl_euler_k[-1]) / np.abs(N_true_j)

print(f"Stepsize Δt = {dt_k}")
print(f"Number of steps M = {M_k}")
print(f"True N(T) = {N_true_j:.8f}")
print()
print(f"Implicit Euler method:")
print(f"  N(T) = {N_impl_euler_k[-1]:.8f}")
print(f"  Percent error = {percent_error_impl_euler_k:.6f}%")
print()

# Part (l): Large stepsize comparison

print("Part (l):")

dt_l = 4.0
M_l = int(T / dt_l)

N_true_l = N0 * np.exp(-mu * T)

_, N_euler_l = euler_integrate(mu, N0, T, M_l)
_, N_impl_euler_l = implicit_euler_integrate(mu, N0, T, M_l)

percent_error_euler_l = 100 * np.abs(N_true_l - N_euler_l[-1]) / np.abs(N_true_l)
percent_error_impl_euler_l = 100 * np.abs(N_true_l - N_impl_euler_l[-1]) / np.abs(N_true_l)

print(f"Large stepsize Δt = {dt_l} (single step from t=0 to t=T)")
print(f"Number of steps M = {M_l}")
print(f"True N(T) = {N_true_l:.8f}")
print()
print(f"Explicit Euler method:")
print(f"  N(T) = {N_euler_l[-1]:.8f}")
print(f"  Percent error = {percent_error_euler_l:.6f}%")
print()
print(f"Implicit Euler method:")
print(f"  N(T) = {N_impl_euler_l[-1]:.8f}")
print(f"  Percent error = {percent_error_impl_euler_l:.6f}%")
print()
print(f"With such a large stepsize, implicit Euler is much more stable.")
print(f"Explicit Euler gives {percent_error_euler_l:.2f}% error while implicit Euler gives {percent_error_impl_euler_l:.2f}%")
print()
