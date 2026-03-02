import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, digamma
from scipy.optimize import minimize_scalar
import os

# PART (a): Read BglI-fragments.fasta and compute fragment lengths

def read_fasta_fragments(filename):
    """
    read dna frags from fasta and get lengths
    
    parameters:
    filename; path
    
    returns:
    fragment lengths
    """
    
    fragment_lengths = []
    
    with open(filename, 'r') as f:
        
        for line in f:
            # Remove whitespace and newlines
        
            sequence = line.strip()
        
            if sequence:  # Only add non-empty lines
                fragment_lengths.append(len(sequence))
    
    return fragment_lengths


# function: bin_data

def bin_data(counts, nbins=10):
    '''
    puts stuff into bins

    counts should be a list of integers
    '''
    
    # use np.histogram to do the counting and set up the bins
    y, x = np.histogram(counts, nbins, density=True)
    # use the bin edges to compute bin centers (last value is junk so drop it)
    bin_centers = (x + np.roll(x, -1)) / 2

    # Get differences between bin_centers
    bin_diffs = np.diff(bin_centers)

    bin_centers = bin_centers[:-1]

    # return bin centers as x and PDF as y
    return bin_centers, y, bin_diffs


# PART (c): Exponential distribution MLE

def negative_log_likelihood_exponential(mu, data):
    """
    negative log likelihood for exponential distribution P(l|mu) = (1/mu)*exp(-l/mu)
    
    parameters:
    mu: parameter of exponential distribution
    data: frag length data
    
    returns:
    float: negative log likelihood
    """
    
    if mu <= 0:
        return np.inf
    nll = -np.sum(np.log(1/mu) - data/mu)
    
    return nll


def fit_exponential(data):
    """
    fit exponential distribution to data using maximum likelihood estimation.
    
    for exponential distribution P(l|mu) = (1/mu)*exp(-l/mu), 
    the MLE is simply mu = mean(data)
    
    parameters:
    data: frag length data
    
    returns:
    max likelihood estimate of mu
    """
    mu_mle = np.mean(data)
    return mu_mle


# PART (e) & (f): Gamma distribution NLL and gradient

def negative_log_likelihood_gamma(alpha, beta, data):
    """
    negative log likelihood for gamma distribution P(l|alpha,beta) = (beta^alpha/Gamma(alpha)) * l^(alpha-1) * exp(-beta*l)
    
    parameters:
    alpha: shape parameter
    beta: rate parameter
    data: frag length data
    
    returns:
    negative log likelihood
    """
    if alpha <= 0 or beta <= 0:
        return np.inf
    
    nll = -np.sum(alpha * np.log(beta) - np.log(gamma(alpha)) + 
                  (alpha - 1) * np.log(data) - beta * data)
    return nll


def gradient_gamma(alpha, beta, data):
    """
    get gradient of negative log likelihood with respect to alpha and beta.
    
    grad components:
    d(NLL)/d(alpha) = -log(beta) + digamma(alpha) - log(data).mean()
    d(NLL)/d(beta) = -alpha/beta + data.mean()
    
    parameters:
    alpha: shape parameter
    beta: rate parameter
    data: frag length data
    
    returns:
    tuple: (grad_alpha, grad_beta)
    """
    n = len(data)
    grad_alpha = -np.log(beta) + digamma(alpha) - np.mean(np.log(data))
    grad_beta = -alpha / beta + np.mean(data)
    
    return grad_alpha, grad_beta


# PART (g): Reduce to 1D optimization problem

def get_beta_from_alpha(alpha, data):
    """
    solve d(NLL)/d(beta) = 0 for beta.
    from d(NLL)/d(beta) = -alpha/beta + mean(data) = 0,
    we get beta = alpha / mean(data)
    
    parameters:
    alpha: shape parameter
    data: frag length data
    
    returns:
    optimal beta given alpha
    """
    return alpha / np.mean(data)


def reduced_nll_gamma(alpha, data):
    """
    negative log likelihood in 1D form where beta is optimized for given alpha.
    
    parameters:
    alpha: shape parameter
    data: frag length data
    
    returns:
    negative log likelihood with beta optimized
    """
    beta = get_beta_from_alpha(alpha, data)
    return negative_log_likelihood_gamma(alpha, beta, data)


def reduced_gradient_gamma(alpha, data):
    """
    grad of reduced NLL with respect to alpha only.
    
    after solving for beta and substituting: 
    grad_alpha = -log(alpha / mean(data)) + digamma(alpha) - mean(log(data))
    
    parameters:
    alpha: shape parameter
    data: frag length data
    
    returns:
    gradient with respect to alpha
    """
    beta = get_beta_from_alpha(alpha, data)
    return -np.log(beta) + digamma(alpha) - np.mean(np.log(data))


# PART (h): Steepest descent algorithm

def steepest_descent_exponential(data, learning_rate=0.1, tolerance=1e-6, max_iterations=10000):
    """
    Steepest descent algorithm to fit exponential parameter.
    Used to validate the steepest descent implementation.
    
    For exponential P(l|mu) = (1/mu)*exp(-l/mu), the gradient is:
    d(NLL)/d(mu) = -n/mu + sum(data)/mu^2
    Setting to zero gives: mu = mean(data) 
    
    parameters:
    data: frag length data
    learning_rate: learning rate for gradient descent
    tolerance: convergence tolerance
    max_iterations: max number of iterations
    
    returns:
    dict: Contains final mu estimate and iteration history
    """
    # Initialize with a reasonable value
    mu = np.max(data) / 4
    best_mu = mu
    best_nll = negative_log_likelihood_exponential(mu, data)
    
    iteration_history = {'mu': [mu], 'nll': [best_nll], 'grad': []}
    
    n = len(data)
    sum_data = np.sum(data)
    
    for iteration in range(max_iterations):
        # Analytical gradient for exponential: d(NLL)/d(mu) = -n/mu + sum(data)/mu^2
        grad = -n / mu + sum_data / (mu ** 2)
        iteration_history['grad'].append(grad)
        
        # Update parameter with standard gradient descent
        mu_new = mu - learning_rate * grad
        
        if mu_new <= 0:  # Ensure mu stays positive
            mu_new = abs(mu_new) + 0.1
        
        # Compute NLL at new point
        nll_new = negative_log_likelihood_exponential(mu_new, data)
        
        # Keep track of best parameters
        if nll_new < best_nll:
            best_nll = nll_new
            best_mu = mu_new
        
        iteration_history['mu'].append(mu_new)
        iteration_history['nll'].append(nll_new)
        
        # Check convergence
        if abs(mu_new - mu) < tolerance or abs(grad) < tolerance:
            print(f"Exponential fit converged at iteration {iteration}")
            break
        
        mu = mu_new
    
    return {'mu': best_mu, 'history': iteration_history}


def steepest_descent_gamma(data, learning_rate=0.001, tolerance=1e-6, max_iterations=10000):
    """
    steepest descent algorithm to fit gamma parameters using 1D optimization on alpha.
    
    parameters:
    data: frag length data
    learning_rate: learning rate for gradient descent
    tolerance: convergence tolerance
    max_iterations: max number of iterations
    
    returns:
    dict: Contains final alpha, beta estimates and iteration history
    """
    # Initialize alpha with reasonable value
    alpha = 2.0
    best_alpha = alpha
    best_beta = get_beta_from_alpha(alpha, data)
    best_nll = reduced_nll_gamma(alpha, data)
    
    iteration_history = {'alpha': [alpha], 'beta': [best_beta], 'nll': [best_nll]}
    
    for iteration in range(max_iterations):
        # Compute gradient with respect to alpha
        grad = reduced_gradient_gamma(alpha, data)
        
        # Update alpha
        alpha_new = alpha - learning_rate * grad
        
        if alpha_new <= 0:  # Ensure alpha stays positive
            alpha_new = abs(alpha_new) + 0.1
        
        # Compute beta and NLL at new point
        beta_new = get_beta_from_alpha(alpha_new, data)
        nll = reduced_nll_gamma(alpha_new, data)
        
        # Keep track of best parameters
        if nll < best_nll:
            best_nll = nll
            best_alpha = alpha_new
            best_beta = beta_new
        
        iteration_history['alpha'].append(alpha_new)
        iteration_history['beta'].append(beta_new)
        iteration_history['nll'].append(nll)
        
        # Check convergence
        if abs(alpha_new - alpha) < tolerance or abs(grad) < tolerance:
            print(f"Gamma fit converged at iteration {iteration}")
            break
        
        alpha = alpha_new
    
    return {'alpha': best_alpha, 'beta': best_beta, 'history': iteration_history}


# main execution

if __name__ == "__main__":
    
    # PART (a): Read FASTA file
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fasta_file = os.path.join(script_dir, 'BgIl-fragments.fasta')

    print("PART (a): Reading BgIl-fragments.fasta")
    
    fragment_lengths = read_fasta_fragments(fasta_file)
    fragment_lengths = np.array(fragment_lengths)
    
    print(f"Number of fragments: {len(fragment_lengths)}")
    print(f"Min fragment length: {np.min(fragment_lengths)}")
    print(f"Max fragment length: {np.max(fragment_lengths)}")
    print(f"Mean fragment length: {np.mean(fragment_lengths):.2f}")
    print()
    
    # PART (b): Plot empirical distribution
    print("PART (b): Plotting empirical distribution")
    
    # Create figure for part (b)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bin the data with a reasonable number of bins
    nbins = 50
    bin_centers, bin_counts, bin_diffs = bin_data(fragment_lengths, nbins=nbins)
    
    # Plot as dot plot
    ax1.plot(bin_centers, bin_counts, 'bo', markersize=6, label='Empirical Distribution')
    ax1.set_xlabel('Fragment Length (bp)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Empirical Fragment Length Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig('part_b_empirical_distribution.png', dpi=150)
    print("Saved: part_b_empirical_distribution.png")
    print()
    
    # PART (c): Fit exponential distribution
    print("PART (c): Exponential MLE")
    
    # Calculate MLE for exponential
    mu_mle = fit_exponential(fragment_lengths)
    print(f"Maximum Likelihood Estimate for mu (exponential): {mu_mle:.2f}")
    
    # Verify with steepest descent
    print("\nVerifying with steepest descent algorithm...")
    exp_fit = steepest_descent_exponential(fragment_lengths, learning_rate=0.01)
    print(f"Steepest descent estimate for mu: {exp_fit['mu']:.2f}")
    print()
    
    # PART (d): Plot exponential fit
    print("PART (d): Plotting exponential fit")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Plot empirical data
    ax2.plot(bin_centers, bin_counts, 'bo', markersize=6, label='Empirical Distribution', alpha=0.7)
    
    
    # Plot fitted exponential distribution
    x_smooth = np.linspace(0, np.max(fragment_lengths), 1000)
    y_exponential = (1 / mu_mle) * np.exp(-x_smooth / mu_mle)
    ax2.plot(x_smooth, y_exponential, 'r-', linewidth=2.5, label=f'Exponential Fit (mu={mu_mle:.2f})')
    
    ax2.set_xlabel('Fragment Length (bp)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Fragment Length Distribution with Exponential Fit', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('part_d_exponential_fit.png', dpi=150)
    print("Saved: part_d_exponential_fit.png")
    print()
    
    
    # PART (e), (f), (g): Gamma distribution setup
    print("PART (e), (f), (g): Gamma distribution setup")
    print("Negative log likelihood for gamma: NLL = SUM[-a*log(b) + log(Gamma(a)) - (a-1)*log(li) + b*li]")
    print("Gradient wrt alpha: dNLL/da = -log(b) + psi(a) - <log(l)>")
    print("Gradient wrt beta: dNLL/db = -a/b + <l>")
    print("After solving dNLL/db=0: b = a / <l>")
    print("Reduces problem to 1D optimization in alpha")
    print()
    
    
    # PART (h): Steepest descent for gamma
    print("PART (h): Steepest descent for gamma distribution")
    
    gamma_fit = steepest_descent_gamma(fragment_lengths, learning_rate=0.001, tolerance=1e-6)
    alpha_fit = gamma_fit['alpha']
    beta_fit = gamma_fit['beta']
    
    print(f"Fitted alpha (shape): {alpha_fit:.4f}")
    print(f"Fitted beta (rate): {beta_fit:.6f}")
    print(f"Final NLL: {reduced_nll_gamma(alpha_fit, fragment_lengths):.2f}")
    print()
    
    
    # Compare NLL values
    exp_nll = negative_log_likelihood_exponential(mu_mle, fragment_lengths)
    gamma_nll = negative_log_likelihood_gamma(alpha_fit, beta_fit, fragment_lengths)
    print(f"Exponential NLL: {exp_nll:.2f}")
    print(f"Gamma NLL: {gamma_nll:.2f}")
    if gamma_nll < exp_nll:
        print("→ Gamma distribution provides better fit")
    else:
        print("→ Exponential distribution provides better fit")
    print()
    
    
    # Plot both fits
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Plot empirical data
    ax3.plot(bin_centers, bin_counts, 'bo', markersize=6, label='Empirical Distribution', alpha=0.7)
    
    # Plot exponential fit
    x_smooth = np.linspace(0, np.max(fragment_lengths), 1000)
    y_exponential = (1 / mu_mle) * np.exp(-x_smooth / mu_mle)
    ax3.plot(x_smooth, y_exponential, 'r--', linewidth=2, label=f'Exponential (μ={mu_mle:.2f})', alpha=0.8)
    
    # Plot gamma fit
    y_gamma = (beta_fit**alpha_fit / gamma(alpha_fit)) * (x_smooth**(alpha_fit - 1)) * np.exp(-beta_fit * x_smooth)
    ax3.plot(x_smooth, y_gamma, 'g-', linewidth=2.5, label=f'Gamma (a={alpha_fit:.4f}, b={beta_fit:.6f})')
    
    ax3.set_xlabel('Fragment Length (bp)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Fragment Length Distribution: Exponential vs Gamma Fit', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('part_h_gamma_fit.png', dpi=150)
    print("Saved: part_h_gamma_fit.png")
    print()
    
    # PART (i): Compare steepest descent with different learning rates
    print("PART (i): Steepest descent with different learning rates")
    
    # run with larger learning rate
    print("running steepest descent with learning rate 0.001...")
    gamma_fit_large_lr = steepest_descent_gamma(fragment_lengths, learning_rate=0.001, tolerance=1e-8)
    alpha_large = gamma_fit_large_lr['alpha']
    beta_large = gamma_fit_large_lr['beta']
    nll_large = gamma_fit_large_lr['history']['nll']
    
    # run with smaller learning rate
    print("running steepest descent with learning rate 0.000001...")
    gamma_fit_small_lr = steepest_descent_gamma(fragment_lengths, learning_rate=0.000001, tolerance=1e-8)
    alpha_small = gamma_fit_small_lr['alpha']
    beta_small = gamma_fit_small_lr['beta']
    nll_small = gamma_fit_small_lr['history']['nll']
    
    print(f"\nlarge learning rate (0.001): alpha = {alpha_large:.6f}, beta = {beta_large:.8f}")
    print(f"small learning rate (0.000001): alpha = {alpha_small:.6f}, beta = {beta_small:.8f}")
    print(f"parameter difference: delta_alpha = {abs(alpha_large - alpha_small):.8f}, delta_beta = {abs(beta_large - beta_small):.10f}")
    print()
    
    # plot cost vs iteration for both learning rates
    fig_lr, ax_lr = plt.subplots(figsize=(10, 6))
    iterations_large = range(len(nll_large))
    iterations_small = range(len(nll_small))
    ax_lr.plot(iterations_large, nll_large, 'r.-', label='Learning rate = 0.001', markersize=4, linewidth=1.5)
    ax_lr.plot(iterations_small, nll_small, 'b.-', label='Learning rate = 0.000001', markersize=3, linewidth=1)
    ax_lr.set_xlabel('Iteration', fontsize=12)
    ax_lr.set_ylabel('Negative Log Likelihood', fontsize=12)
    ax_lr.set_title('Steepest Descent: Effect of Learning Rate on Convergence', fontsize=14)
    ax_lr.grid(True, alpha=0.3)
    ax_lr.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('part_i_learning_rate_comparison.png', dpi=150)
    print("saved: part_i_learning_rate_comparison.png")
    print()
    
    # PART (k): Word length distribution
    print("PART (k): Word length distribution from wordlist")
    
    # read wordlist
    wordlist_path = os.path.join(script_dir, 'wordlist.txt')
    word_lengths = []
    with open(wordlist_path, 'r') as f:
        for line in f:
            word = line.strip()
            if word:
                word_lengths.append(len(word))
    
    word_lengths = np.array(word_lengths)
    print(f"total words: {len(word_lengths)}")
    print(f"min word length: {np.min(word_lengths)}")
    print(f"max word length: {np.max(word_lengths)}")
    print(f"mean word length: {np.mean(word_lengths):.2f}")
    
    # count occurrences of each length
    unique_lengths = np.arange(1, np.max(word_lengths) + 1)
    counts = np.array([np.sum(word_lengths == length) for length in unique_lengths])
    probabilities = counts / len(word_lengths)
    
    # plot word length distribution
    fig_words, ax_words = plt.subplots(figsize=(10, 6))
    ax_words.bar(unique_lengths, probabilities, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_words.set_xlabel('Word Length (letters)', fontsize=12)
    ax_words.set_ylabel('Probability', fontsize=12)
    ax_words.set_title('English Word Length Distribution', fontsize=14)
    ax_words.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('part_k_word_length_distribution.png', dpi=150)
    print("saved: part_k_word_length_distribution.png")
    print()
    
    # PART (l): Fit exponential and gamma to word length data
    print("PART (l): Fitting distributions to word lengths")
    
    # fit exponential to word lengths
    mu_words = fit_exponential(word_lengths)
    nll_exp_words = negative_log_likelihood_exponential(mu_words, word_lengths)
    print(f"exponential fit: mu = {mu_words:.4f}, NLL = {nll_exp_words:.2f}")
    
    # fit gamma to word lengths
    gamma_words_fit = steepest_descent_gamma(word_lengths, learning_rate=0.01, tolerance=1e-8)
    alpha_words = gamma_words_fit['alpha']
    beta_words = gamma_words_fit['beta']
    nll_gamma_words = reduced_nll_gamma(alpha_words, word_lengths)
    print(f"gamma fit: alpha = {alpha_words:.4f}, beta = {beta_words:.6f}, NLL = {nll_gamma_words:.2f}")
    print(f"NLL difference (exponential - gamma): {nll_exp_words - nll_gamma_words:.2f}")
    
    if nll_gamma_words < nll_exp_words:
        print("gamma provides better fit")
    else:
        print("exponential provides better fit")
    print()
    
    # plot both fits on word length data
    fig_word_fits, ax_word_fits = plt.subplots(figsize=(12, 6))
    
    # plot empirical data
    ax_word_fits.bar(unique_lengths, probabilities, color='lightgray', alpha=0.6, 
                     edgecolor='black', linewidth=0.5, label='empirical distribution')
    
    # plot exponential fit
    x_word = np.linspace(0.5, np.max(unique_lengths) + 0.5, 500)
    y_exp_words = (1 / mu_words) * np.exp(-x_word / mu_words)
    ax_word_fits.plot(x_word, y_exp_words, 'r--', linewidth=2.5, label=f'exponential (mu={mu_words:.2f})')
    
    # plot gamma fit
    y_gamma_words = (beta_words**alpha_words / gamma(alpha_words)) * \
                    (x_word**(alpha_words - 1)) * np.exp(-beta_words * x_word)
    ax_word_fits.plot(x_word, y_gamma_words, 'g-', linewidth=2.5, 
                      label=f'gamma (alpha={alpha_words:.4f}, beta={beta_words:.6f})')
    
    ax_word_fits.set_xlabel('Word Length (letters)', fontsize=12)
    ax_word_fits.set_ylabel('Probability', fontsize=12)
    ax_word_fits.set_title('Word Length Distribution with Exponential and Gamma Fits', fontsize=14)
    ax_word_fits.set_xlim(0, np.max(unique_lengths) + 1)
    ax_word_fits.grid(True, alpha=0.3, axis='y')
    ax_word_fits.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('part_l_word_length_fits.png', dpi=150)
    print("saved: part_l_word_length_fits.png")
    print()
    
    print("all analysis complete")
    
    plt.show()
