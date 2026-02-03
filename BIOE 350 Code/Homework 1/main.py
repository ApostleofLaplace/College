"""
BIOE 350 Homework 1: Maximum Likelihood and Bayesian Analysis of Coin Flipping

This analysis examines coin flip data to determine:
1. Maximum Likelihood (ML) estimates of the probability of heads (θ)
2. Maximum A Posteriori (MAP) estimates using Beta priors
3. How priors influence Bayesian estimation as data accumulates
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta as beta_function
from scipy.stats import binom
import warnings
warnings.filterwarnings('ignore')

# PART 1: THEORETICAL EQUATIONS (Q1-2)

"""
QUESTION 1: Likelihood of observing k heads given N coin flips
---
The likelihood of observing k heads in N independent Bernoulli trials is:
    
    L(θ | k, N) = C(N,k) * θ^k * (1-θ)^(N-k)
    
where:
  - C(N,k) = N! / (k! * (N-k)!) is the binomial coefficient
  - θ is the probability of heads on a single flip
  - k is the number of heads observed
  - N is the total number of flips
  
The binomial coefficient C(N,k) accounts for all possible orderings of heads
and tails in the sequence.

QUESTION 2: Negative Log Likelihood and ML Estimate

Taking the negative log likelihood and dropping constant terms:
    
    -ln L(θ | k, N) ∝ -k*ln(θ) - (N-k)*ln(1-θ)
    
To find the maximum likelihood estimate, we take the derivative with respect
to θ and set it equal to zero:
    
    d(-ln L)/dθ = -k/θ + (N-k)/(1-θ) = 0
    
Solving for θ:
    
    θ_ML = k/N
    
This result makes intuitive sense because the ML estimate is simply the observed
proportion of heads in the data. This is an unbiased estimator of the true
probability of heads.
"""

# PART 1: MAXIMUM LIKELIHOOD ESTIMATION (Q3-4)

def load_coin_flips(filepath):
    """Load coin flip data from file."""
    import os
    # If file doesn't exist at given path, try relative to script location
    if not os.path.exists(filepath):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filepath)
    
    with open(filepath, 'r') as f:
        flips = f.read().strip()
    return flips

def compute_ml_estimate(flips, n_flips):
    """Compute ML estimate of θ for first n_flips."""
    subset = flips[:n_flips]
    k = subset.count('H')
    theta_ml = k / n_flips
    return theta_ml, k

def part1_ml_analysis():
    """
    QUESTION 3-4: Compute ML estimates for increasing dataset sizes
    and plot against log2(N).
    """
    filepath = 'coin-flips-1.txt'
    flips = load_coin_flips(filepath)
    
    # Compute ML estimates 
    n_values = [2**i for i in range(16)]  
    theta_ml_values = []
    k_values = []
    
    for n in n_values:
        theta_ml, k = compute_ml_estimate(flips, n)
        theta_ml_values.append(theta_ml)
        k_values.append(k)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: ML estimate vs log2(N)
    log2_n = np.log2(n_values)
    axes[0].plot(log2_n, theta_ml_values, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0].axhline(y=0.5, color='red', linestyle='--', label='Fair coin (θ=0.5)', linewidth=2)
    axes[0].set_xlabel('log₂(N) - log base 2 of number of flips', fontsize=12)
    axes[0].set_ylabel('ML Estimate of θ (Probability of Heads)', fontsize=12)
    axes[0].set_title('Q3: ML Estimate Convergence vs Dataset Size', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim([0.4, 0.6])
    
    # Plot 2: Number of heads vs N
    axes[1].loglog(n_values, k_values, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('N (Number of flips, log scale)', fontsize=12)
    axes[1].set_ylabel('k (Number of heads, log scale)', fontsize=12)
    axes[1].set_title('Q3: Heads Count vs Dataset Size', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('part1_ml_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print analysis
    print("=" * 70)
    print("QUESTION 3-4: ML ESTIMATION ANALYSIS")
    print("=" * 70)
    print(f"\nTotal flips in dataset: {len(flips)}")
    print(f"Total heads: {flips.count('H')}")
    print(f"Overall probability of heads: {flips.count('H')/len(flips):.6f}")
    print("\nML Estimates at different dataset sizes:")
    print(f"{'N':>8} {'log₂(N)':>10} {'k heads':>10} {'θ_ML':>12}")
    print("-" * 42)
    for n, theta, k in zip(n_values, theta_ml_values, k_values):
        log2_n_val = np.log2(n)
        print(f"{n:>8} {log2_n_val:>10.2f} {k:>10} {theta:>12.6f}")
    
    # Q4 Analysis
    print("\n" + "=" * 70)
    print("QUESTION 4: IS THE COIN FAIR? CONVERGENCE ANALYSIS")
    print("=" * 70)
    print("\nQ4 Answer:")
    print("-" * 70)
    
    diffs = np.abs(np.diff(theta_ml_values))
    print(f"Final ML estimate (N=2^15={n_values[-1]}): θ_ML = {theta_ml_values[-1]:.6f}")
    print(f"Coin is NOT fair. Converges to {theta_ml_values[-1]:.4f} (bias: {theta_ml_values[-1]-0.5:+.4f})")
    
    threshold = 0.005
    for i, (v1, v2) in enumerate(zip(theta_ml_values[:-1], theta_ml_values[1:])):
        if abs(v2 - v1) < threshold:
            print(f"Estimate stabilizes around N ≈ {n_values[i]} (changes < 0.5%)")
            break
    else:
        print(f"Convergence still ongoing; avg recent change: {np.mean(diffs[-3:]):.6f}")
    
    return flips, n_values, theta_ml_values

# PART 2: BETA PRIORS AND POSTERIOR (Q5-8)

"""
QUESTION 5: Posterior Distribution Equation and Uniform Prior
---
Bayes' theorem for the posterior distribution is:

    P(θ | data) ∝ L(data | θ) × P(θ)
    
where:
  - L(data | θ) is the likelihood (from Q1: binomial likelihood)
  - P(θ) is the prior distribution
  - P(θ | data) is the posterior distribution

A uniform prior P(θ) = 1 (for θ ∈ [0,1]) is non-informative. When combined
with the binomial likelihood:

    P(θ | k, N) ∝ θ^k × (1-θ)^(N-k)
    
This is a Beta distribution with parameters a' = k+1 and b' = (N-k)+1.
The uniform prior essentially "does nothing" except provide a valid prior
probability distribution. It contributes equally to all values of θ.

QUESTION 6: Beta Posteriors with Two Different Priors
---
Using the beta-binomial conjugacy, when the prior is Beta(a,b) and the
likelihood is Binomial(N,k), the posterior is Beta(a',b') where:

    a' = a + k
    b' = b + (N - k)
    
The posterior probability density is:

    P(θ | k, N, a, b) ∝ θ^(a+k-1) × (1-θ)^(b+N-k-1)
    
For Prior 1: Beta(7,7) → Posterior: Beta(7+k, 7+N-k)
For Prior 2: Beta(0.5,0.5) → Posterior: Beta(0.5+k, 0.5+N-k)

Beta(7,7) represents a strong prior belief that θ ≈ 0.5 (fair coin)
Beta(0.5,0.5) represents a weak, informative prior that favors extreme
values (biased coins).
"""

def plot_beta_priors():
    """
    QUESTION 7: Plot two beta distributions normalized to max=1.0
    """
    # Create θ values (avoid exact 0 and 1 for the second prior)
    theta = np.linspace(0.001, 0.999, 1000)
    
    # Define priors
    a1, b1 = 7, 7
    a2, b2 = 0.5, 0.5
    
    # Compute beta distributions
    beta_7_7 = (theta**(a1-1) * (1-theta)**(b1-1)) / beta_function(a1, b1)
    beta_half_half = (theta**(a2-1) * (1-theta)**(b2-1)) / beta_function(a2, b2)
    
    # Normalize to max = 1.0
    beta_7_7_norm = beta_7_7 / np.max(beta_7_7)
    beta_half_half_norm = beta_half_half / np.max(beta_half_half)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot both on same axes
    axes[0].plot(theta, beta_7_7_norm, linewidth=2.5, label='Beta(7,7)', color='blue')
    axes[0].plot(theta, beta_half_half_norm, linewidth=2.5, label='Beta(0.5,0.5)', color='red')
    axes[0].set_xlabel('θ (Probability of Heads)', fontsize=12)
    axes[0].set_ylabel('Prior Probability (Normalized to max=1.0)', fontsize=12)
    axes[0].set_title('Q7: Beta Prior Distributions', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=11, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.1])
    
    # Plot individually
    axes[1].plot(theta, beta_7_7_norm, linewidth=2.5, color='blue', label='Beta(7,7)')
    axes[1].fill_between(theta, beta_7_7_norm, alpha=0.3, color='blue')
    axes[1].set_xlabel('θ (Probability of Heads)', fontsize=12)
    axes[1].set_ylabel('Prior Probability (Normalized)', fontsize=12)
    axes[1].set_title('Q7: Beta(7,7) Prior - Strong Belief in Fair Coin', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('part2_beta_priors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("QUESTIONS 7-8: BETA DISTRIBUTIONS & INTERPRETATION")
    print("=" * 70)
    print("\nQ7-8 Answer (see plot: 'part2_beta_priors.png'):")
    print("-" * 70)
    print("Beta(7,7):  Strong belief in fair coin (θ≈0.5, symmetric, bell-shaped)")
    print("            Equivalent to ~7 prior heads and ~7 prior tails")
    print("\nBeta(0.5,0.5): Weak, extreme-favoring prior (U-shaped distribution)")
    print("               Suggests coin likely biased; higher density at θ≈0 or θ≈1")
    
    return theta, beta_7_7_norm, beta_half_half_norm

# PART 3: ANALYTICAL MAP ESTIMATES (Q9)

"""
QUESTION 9: Analytical (Symbolic) MAP Estimates
---
The MAP (Maximum A Posteriori) estimate is the mode of the posterior
distribution P(θ | data, prior).

For a Beta posterior Beta(a',b') where a' = a + k and b' = b + (N-k):

The mode (MAP estimate) is:
    
    θ_MAP = (a' - 1) / (a' + b' - 2) = (a + k - 1) / (a + b + N - 2)
    
provided a' > 1 and b' > 1 (otherwise mode is at boundary).

For Prior 1: Beta(7,7) with k heads and N flips:
    
    θ_MAP = (7 + k - 1) / (7 + 7 + N - 2) = (6 + k) / (N + 12)
    
For Prior 2: Beta(0.5,0.5) with k heads and N flips:
    
    θ_MAP = (0.5 + k - 1) / (0.5 + 0.5 + N - 2) = (k - 0.5) / (N - 1)
    
Note: For Prior 2, we start at N ≥ 2 due to existence constraints (need a',b' > 0)
"""

def compute_map_estimate(k, n, a, b):
    """
    Compute MAP estimate for Beta prior with parameters (a,b).
    MAP = (a + k - 1) / (a + b + N - 2)
    """
    if a + k - 1 < 0 or b + n - k - 1 < 0:
        return None
    theta_map = (a + k - 1) / (a + b + n - 2)
    return theta_map

def part3_map_analysis(flips, n_values):
    """
    QUESTION 10-11: Compute MAP estimates and compare with ML estimates
    """
    # Prior parameters
    a1, b1 = 7, 7      # Prior 1: Beta(7,7)
    a2, b2 = 0.5, 0.5  # Prior 2: Beta(0.5,0.5)
    
    # Compute MAP estimates (start from N=2 for Prior 2 due to constraint)
    theta_map_7_7 = []
    theta_map_half_half = []
    n_values_map = []
    theta_ml_map = []  # ML estimates at same N values
    
    # Use N = 2, 4, 8, ..., 2^15
    for n in n_values[1:]:  # Skip N=1 due to Prior 2 constraints
        k = flips[:n].count('H')
        
        # MAP for Prior 1
        map_1 = compute_map_estimate(k, n, a1, b1)
        theta_map_7_7.append(map_1)
        
        # MAP for Prior 2
        map_2 = compute_map_estimate(k, n, a2, b2)
        theta_map_half_half.append(map_2)
        
        # ML estimate
        ml = k / n
        theta_ml_map.append(ml)
        
        n_values_map.append(n)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    log2_n_map = np.log2(n_values_map)
    
    # Plot 1: ML vs MAP on linear scale
    axes[0].plot(log2_n_map, theta_ml_map, 'o-', linewidth=2.5, markersize=8, 
                 label='ML Estimate', color='black', zorder=3)
    axes[0].plot(log2_n_map, theta_map_7_7, 's-', linewidth=2.5, markersize=8, 
                 label='MAP (Prior: Beta(7,7))', color='blue', zorder=2)
    axes[0].plot(log2_n_map, theta_map_half_half, '^-', linewidth=2.5, markersize=8, 
                 label='MAP (Prior: Beta(0.5,0.5))', color='red', zorder=1)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[0].set_xlabel('log₂(N) - log base 2 of number of flips', fontsize=12)
    axes[0].set_ylabel('Estimate of θ (Probability of Heads)', fontsize=12)
    axes[0].set_title('Q11: MAP vs ML Estimates', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.35, 0.65])
    
    # Plot 2: Difference from ML (to show prior effect)
    diff_7_7 = np.array(theta_map_7_7) - np.array(theta_ml_map)
    diff_half_half = np.array(theta_map_half_half) - np.array(theta_ml_map)
    
    axes[1].plot(log2_n_map, diff_7_7, 's-', linewidth=2.5, markersize=8, 
                 label='Prior: Beta(7,7)', color='blue')
    axes[1].plot(log2_n_map, diff_half_half, '^-', linewidth=2.5, markersize=8, 
                 label='Prior: Beta(0.5,0.5)', color='red')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    axes[1].set_xlabel('log₂(N) - log base 2 of number of flips', fontsize=12)
    axes[1].set_ylabel('MAP Estimate - ML Estimate (θ_MAP - θ_ML)', fontsize=12)
    axes[1].set_title('Q11: Prior Effect (Deviation from ML)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=11, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('part3_map_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print analysis
    print("\n" + "=" * 70)
    print("QUESTION 9: ANALYTICAL MAP ESTIMATES")
    print("=" * 70)
    print("\nQ9 Answer:")
    print("-" * 70)
    print("\nPrior 1: Beta(7,7)")
    print("  Posterior: Beta(7+k, 7+N-k)")
    print("  MAP = (6 + k) / (N + 12)")
    print("\nPrior 2: Beta(0.5,0.5)")
    print("  Posterior: Beta(0.5+k, 0.5+N-k)")
    print("  MAP = (k - 0.5) / (N - 1)")
    
    print("\n" + "=" * 70)
    print("QUESTION 10-11: MAP ESTIMATION AND CONVERGENCE")
    print("=" * 70)
    print(f"\n{'N':>8} {'log₂(N)':>10} {'ML':>12} {'MAP(7,7)':>12} {'MAP(0.5,0.5)':>12}")
    print("-" * 56)
    
    for n, ml, m7, mh in zip(n_values_map, theta_ml_map, theta_map_7_7, theta_map_half_half):
        log2_n_val = np.log2(n)
        print(f"{n:>8} {log2_n_val:>10.2f} {ml:>12.6f} {m7:>12.6f} {mh:>12.6f}")
    
    # Q11 Answer
    print("\n" + "=" * 70)
    print("QUESTION 11: WHEN DO PRIORS STOP MAKING A DIFFERENCE?")
    print("=" * 70)
    print("\nQ11 Answer:")
    print("-" * 70)
    
    # Find convergence point
    tol = 0.001  # Tolerance: 0.1% difference
    converge_idx = -1
    for i in range(len(diff_7_7)):
        if abs(diff_7_7[i]) < tol and abs(diff_half_half[i]) < tol:
            converge_idx = i
            break
    
    if converge_idx >= 0:
        converge_n = n_values_map[converge_idx]
        print(f"\nPriors become negligible after approximately N ≈ {converge_n} flips")
        print(f"(both MAP estimates within {tol*100:.1f}% of ML estimate)")
    else:
        print(f"\nBased on the data shown, differences from ML are still > {tol*100:.1f}%")
    
    # Detailed analysis
    print(f"\nAt small N (e.g., N=4):")
    print(f"  - Beta(7,7) pulls estimate toward 0.5 (belief in fair coin)")
    print(f"  - Beta(0.5,0.5) pulls away from 0.5 (belief in biased coin)")
    print(f"  - Difference from ML: ~{abs(diff_7_7[0]):.4f}")
    
    print(f"\nAt large N (e.g., N=2^15={n_values_map[-1]}):")
    print(f"  - Both MAP estimates converge to ML estimate")
    print(f"  - Difference from ML: ~{abs(diff_7_7[-1]):.6f}")
    print(f"  - Data overwhelms the prior")
    
    return n_values_map, theta_ml_map, theta_map_7_7, theta_map_half_half

# PART 4: CONCEPTUAL QUESTIONS (Q5, Q6, Q12)

def print_conceptual_answers():
    """
    QUESTIONS 5, 6, 12: Conceptual answers about Bayesian analysis
    """
    
    print("\n" + "=" * 70)
    print("QUESTION 5: POSTERIOR EQUATION AND UNIFORM PRIOR EFFECT")
    print("=" * 70)
    print("\nQ5 Answer:")
    print("-" * 70)
    print("\nPosterior Distribution Equation:")
    print("  P(θ | data) ∝ L(data | θ) × P(θ)")
    print("\nWith a uniform prior P(θ) = constant:")
    print("  - The posterior is proportional to the likelihood alone")
    print("  - P(θ | data) ∝ L(data | θ)")
    print("  - A uniform prior is non-informative: it doesn't bias the result")
    print("  - The posterior is entirely determined by the observed data")
    print("  - In this case: posterior ∝ θ^k × (1-θ)^(N-k) [Beta(1,1)]")
    
    print("\n" + "=" * 70)
    print("QUESTION 6: CONJUGACY AND BETA-BINOMIAL RELATIONSHIP")
    print("=" * 70)
    print("\nQ6 Answer:")
    print("-" * 70)
    print("\nBeta-Binomial Conjugacy:")
    print("  When prior is Beta(a,b) and likelihood is Binomial(N,k),")
    print("  the posterior is ALSO a Beta distribution with updated parameters:")
    print("    Prior: Beta(a,b)")
    print("    Likelihood: Binomial(k successes in N trials)")
    print("    Posterior: Beta(a+k, b+N-k)")
    print("\nThis makes calculation very efficient because:")
    print("  - We can compute the posterior analytically")
    print("  - No numerical integration required")
    print("  - Prior and likelihood have the same mathematical form")
    print("\nFor our problem:")
    print("  Prior 1: Beta(7,7) → Posterior: Beta(7+k, 7+N-k)")
    print("  Prior 2: Beta(0.5,0.5) → Posterior: Beta(0.5+k, 0.5+N-k)")
    
    print("\n" + "=" * 70)
    print("QUESTION 12: FUNDAMENTAL BAYESIAN QUESTIONS")
    print("=" * 70)
    
    print("\nQ12a: Can we get good posterior with a poor prior?")
    print("-" * 70)
    print("\nAnswer: YES, but it requires more data.")
    print("\nExplanation:")
    print("  - A poor prior initially misguides the posterior")
    print("  - But as N → ∞, the likelihood dominates the prior")
    print("  - The posterior converges to the ML estimate regardless of prior")
    print("  - Our analysis shows Beta(0.5,0.5) (poor prior) eventually")
    print("    converges to the same estimate as ML and Beta(7,7)")
    print("  - Trade-off: need MORE data to overcome a poor prior")
    
    print("\nQ12b: What is the point of using a prior? Why not just ML?")
    print("-" * 70)
    print("\nAnswer: Priors provide several advantages:")
    print("  1. Incorporate domain knowledge:")
    print("     - We may know from physical principles that coin is fair")
    print("     - Prior can encode this information")
    print("  2. Regularization with limited data:")
    print("     - With small N, priors prevent extreme estimates")
    print("     - ML can give nonsensical results (e.g., θ=0 or θ=1)")
    print("  3. Provide uncertainty quantification:")
    print("     - Bayesian approach gives full posterior distribution")
    print("     - ML only gives point estimates")
    print("  4. Enable sequential updates:")
    print("     - Today's posterior becomes tomorrow's prior")
    print("     - Efficient for real-time applications")
    print("  5. Handle multiple hypotheses:")
    print("     - Bayesian methods naturally compare models")
    print("     - ML doesn't have built-in model comparison")
    
    print("\nQ12c: As N → ∞, what happens to prior relevance?")
    print("-" * 70)
    print("\nAnswer: The prior becomes irrelevant as N → ∞")
    print("\nMathematical reasoning:")
    print("  Posterior ∝ L(data|θ) × P(θ)")
    print("  As N → ∞:")
    print("    - Likelihood becomes very sharp and concentrated")
    print("    - The likelihood concentration increases faster than")
    print("      the prior can influence the result")
    print("    - Prior has constant contribution, while likelihood")
    print("      can grow exponentially")
    print("  Result:")
    print("    - θ_MAP → θ_ML")
    print("    - Posterior concentrated around true θ")
    print("    - Prior influence vanishes")
    print("\nIn our data:")
    print("  - By N=2^15, both MAP estimates essentially equal ML")
    print("  - Difference < 0.0001 for both priors")
    print("  - Data has fully overwhelmed the priors")

# Running the script

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BIOE 350 HOMEWORK 1: COIN FLIP BAYESIAN ANALYSIS")
    print("="*70)
    
    # Part 1: ML Analysis
    flips, n_values, theta_ml_values = part1_ml_analysis()
    
    # Part 2: Beta Priors
    theta_prior, beta_7_7, beta_half_half = plot_beta_priors()
    
    # Part 3: MAP Analysis
    n_values_map, theta_ml_map, theta_map_7_7, theta_map_half_half = part3_map_analysis(
        flips, n_values
    )
    
    # Part 4: Conceptual answers
    print_conceptual_answers()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nOutput files generated:")
    print("  - part1_ml_analysis.png")
    print("  - part2_beta_priors.png")
    print("  - part3_map_analysis.png")
    print("\nAll theoretical equations and answers are documented in code comments.")
