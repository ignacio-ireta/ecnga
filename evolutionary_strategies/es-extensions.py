import numpy as np
import math
import statistics
import matplotlib.pyplot as plt

def ackley(x):
    """
    Compute the Ackley function value for a given vector x.
    
    f(x) = -20 * exp(-0.2 * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(2π*x_i))) + 20 + e
    
    Parameters:
        x (np.array or list): Vector of decision variables
    
    Returns:
        float: The computed Ackley function value
    """
    n = len(x)
    sum1 = np.sum(np.square(x))
    sum2 = np.sum(np.cos(2 * np.pi * np.array(x)))
    
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    return term1 + term2 + 20.0 + np.exp(1)

# Challenge Extension 1: Self-adaptation of sigma with exponential update
def initialize_individual_self_adaptive(n):
    """
    Initialize an individual with n decision variables and a single mutation step size
    for self-adaptive ES with exponential update.
    
    Parameters:
        n (int): Number of decision variables
    
    Returns:
        np.array: Individual representation [x_1, x_2, ..., x_n, sigma]
    """
    # Generate n decision variables uniformly in [-30, 30]
    x = np.random.uniform(-30, 30, n)
    
    # Generate a random sigma in (0, 1)
    sigma = np.random.uniform(0, 1)
    
    # Return the individual as [x_1, x_2, ..., x_n, sigma]
    return np.append(x, sigma)

def mutate_self_adaptive(individual):
    """
    Mutate an individual using self-adaptation with exponential update.
    
    Parameters:
        individual (np.array): Individual to mutate [x_1, x_2, ..., x_n, sigma]
    
    Returns:
        np.array: Mutated individual [x'_1, x'_2, ..., x'_n, sigma']
    """
    n = len(individual) - 1  # Number of decision variables
    tau = 1 / np.sqrt(n)     # Learning rate
    
    # Create a mutated copy of the individual
    mutated = individual.copy()
    
    # Mutate sigma first (eq. 4.3: σ' = σ * e^(τ * N(0,1)))
    mutated[n] = individual[n] * np.exp(tau * np.random.normal(0, 1))
    
    # Mutate each decision variable using x'_i = x_i + sigma' * N(0,1)
    for i in range(n):
        mutated[i] = individual[i] + mutated[n] * np.random.normal(0, 1)
    
    return mutated

# Challenge Extension 2: Individual sigma per dimension
def initialize_individual_multiple_sigmas(n):
    """
    Initialize an individual with n decision variables and n mutation step sizes.
    
    Parameters:
        n (int): Number of decision variables
    
    Returns:
        np.array: Individual representation [x_1, x_2, ..., x_n, sigma_1, sigma_2, ..., sigma_n]
    """
    # Generate n decision variables uniformly in [-30, 30]
    x = np.random.uniform(-30, 30, n)
    
    # Generate n random sigmas in (0, 1)
    sigmas = np.random.uniform(0, 1, n)
    
    # Return the individual as [x_1, x_2, ..., x_n, sigma_1, sigma_2, ..., sigma_n]
    return np.concatenate((x, sigmas))

def mutate_multiple_sigmas(individual):
    """
    Mutate an individual with individual step sizes per dimension.
    
    Parameters:
        individual (np.array): Individual to mutate [x_1, ..., x_n, sigma_1, ..., sigma_n]
    
    Returns:
        np.array: Mutated individual with updated step sizes
    """
    n = len(individual) // 2  # Number of decision variables and sigmas
    tau = 1 / np.sqrt(2 * n)       # Overall learning rate
    tau_prime = 1 / np.sqrt(2 * np.sqrt(n))  # Coordinate-wise learning rate
    
    # Create a mutated copy of the individual
    mutated = individual.copy()
    
    # Global random number for coordinated changes
    global_z = np.random.normal(0, 1)
    
    # Mutate each sigma first
    for i in range(n, 2*n):
        # Individual random number for each sigma
        local_z = np.random.normal(0, 1)
        mutated[i] = individual[i] * np.exp(tau_prime * global_z + tau * local_z)
    
    # Then mutate each decision variable
    for i in range(n):
        mutated[i] = individual[i] + mutated[i+n] * np.random.normal(0, 1)
    
    return mutated

def evolutionary_strategy_with_logging(n, G, k, initial_sigma, c):
    """
    Implement the (1+1)-ES algorithm with logging for the Ackley function.
    
    Parameters:
        n (int): Dimension of the problem (number of decision variables)
        G (int): Maximum number of generations
        k (int): Adaptation interval for sigma
        initial_sigma (float): Initial value for sigma
        c (float): Adaptation constant for sigma
    
    Returns:
        tuple: (best_individual, best_fitness, log)
    """
    # Initialization
    t = 0
    m_successful = 0
    
    # Generate and evaluate initial individual
    parent = initialize_individual(n)
    parent[n] = initial_sigma  # Set the initial sigma value
    parent_fitness = ackley(parent[:n])
    
    best_individual = parent.copy()
    best_fitness = parent_fitness
    
    # Logging
    log = {
        "generation": [0],
        "sigma": [parent[n]],
        "fitness": [parent_fitness],
        "best_fitness": [best_fitness]
    }
    
    # Main loop
    while t < G:
        # Offspring Generation
        child = mutate(parent)
        child_fitness = ackley(child[:n])
        
        # Selection (1+1): Keep the better individual
        if child_fitness < parent_fitness:
            parent = child.copy()
            parent_fitness = child_fitness
            m_successful += 1
            
            # Update best solution if needed
            if child_fitness < best_fitness:
                best_individual = child.copy()
                best_fitness = child_fitness
        
        # Increment iteration counter
        t += 1
        
        # Adaptation of sigma every k iterations
        if t % k == 0:
            success_rate = m_successful / k
            
            # Apply 1/5 success rule
            if success_rate > 0.2:
                parent[n] = parent[n] * c  # Decrease sigma for exploitation
            elif success_rate < 0.2:
                parent[n] = parent[n] / c  # Increase sigma for exploration
            # If success_rate == 0.2, keep sigma unchanged
            
            # Reset successful mutation counter
            m_successful = 0
        
        # Log data
        log["generation"].append(t)
        log["sigma"].append(parent[n])
        log["fitness"].append(parent_fitness)
        log["best_fitness"].append(best_fitness)
    
    return best_individual, best_fitness, log

def evolutionary_strategy_self_adaptive(n, G):
    """
    Implement the (1+1)-ES algorithm with self-adaptation for the Ackley function.
    
    Parameters:
        n (int): Dimension of the problem (number of decision variables)
        G (int): Maximum number of generations
    
    Returns:
        tuple: (best_individual, best_fitness, log)
    """
    # Initialization
    t = 0
    
    # Generate and evaluate initial individual
    parent = initialize_individual_self_adaptive(n)
    parent_fitness = ackley(parent[:n])
    
    best_individual = parent.copy()
    best_fitness = parent_fitness
    
    # Logging
    log = {
        "generation": [0],
        "sigma": [parent[n]],
        "fitness": [parent_fitness],
        "best_fitness": [best_fitness]
    }
    
    # Main loop
    while t < G:
        # Offspring Generation with self-adaptation
        child = mutate_self_adaptive(parent)
        child_fitness = ackley(child[:n])
        
        # Selection (1+1): Keep the better individual
        if child_fitness < parent_fitness:
            parent = child.copy()
            parent_fitness = child_fitness
            
            # Update best solution if needed
            if child_fitness < best_fitness:
                best_individual = child.copy()
                best_fitness = child_fitness
        
        # Increment iteration counter
        t += 1
        
        # Log data
        log["generation"].append(t)
        log["sigma"].append(parent[n])
        log["fitness"].append(parent_fitness)
        log["best_fitness"].append(best_fitness)
    
    return best_individual, best_fitness, log

def evolutionary_strategy_multiple_sigmas(n, G):
    """
    Implement the (1+1)-ES algorithm with individual step sizes for the Ackley function.
    
    Parameters:
        n (int): Dimension of the problem (number of decision variables)
        G (int): Maximum number of generations
    
    Returns:
        tuple: (best_individual, best_fitness, log)
    """
    # Initialization
    t = 0
    
    # Generate and evaluate initial individual
    parent = initialize_individual_multiple_sigmas(n)
    parent_fitness = ackley(parent[:n])
    
    best_individual = parent.copy()
    best_fitness = parent_fitness
    
    # Calculate average sigma for logging
    avg_sigma = np.mean(parent[n:2*n])
    
    # Logging
    log = {
        "generation": [0],
        "avg_sigma": [avg_sigma],
        "fitness": [parent_fitness],
        "best_fitness": [best_fitness]
    }
    
    # Main loop
    while t < G:
        # Offspring Generation with multiple sigmas
        child = mutate_multiple_sigmas(parent)
        child_fitness = ackley(child[:n])
        
        # Selection (1+1): Keep the better individual
        if child_fitness < parent_fitness:
            parent = child.copy()
            parent_fitness = child_fitness
            
            # Update best solution if needed
            if child_fitness < best_fitness:
                best_individual = child.copy()
                best_fitness = child_fitness
        
        # Increment iteration counter
        t += 1
        
        # Calculate average sigma for logging
        avg_sigma = np.mean(parent[n:2*n])
        
        # Log data
        log["generation"].append(t)
        log["avg_sigma"].append(avg_sigma)
        log["fitness"].append(parent_fitness)
        log["best_fitness"].append(best_fitness)
    
    return best_individual, best_fitness, log

def plot_results(basic_log, self_adaptive_log, multiple_sigmas_log):
    """
    Plot comparison of the three ES variants.
    
    Parameters:
        basic_log (dict): Logging data from standard (1+1)-ES
        self_adaptive_log (dict): Logging data from self-adaptive ES
        multiple_sigmas_log (dict): Logging data from ES with multiple sigmas
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot best fitness evolution
    ax1.plot(basic_log["generation"], basic_log["best_fitness"], label="Basic (1+1)-ES")
    ax1.plot(self_adaptive_log["generation"], self_adaptive_log["best_fitness"], label="Self-Adaptive ES")
    ax1.plot(multiple_sigmas_log["generation"], multiple_sigmas_log["best_fitness"], label="Multiple Sigmas ES")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness")
    ax1.set_title("Convergence Comparison")
    ax1.legend()
    ax1.grid(True)
    
    # Plot sigma evolution
    ax2.plot(basic_log["generation"], basic_log["sigma"], label="Basic (1+1)-ES")
    ax2.plot(self_adaptive_log["generation"], self_adaptive_log["sigma"], label="Self-Adaptive ES")
    ax2.plot(multiple_sigmas_log["generation"], multiple_sigmas_log["avg_sigma"], label="Multiple Sigmas ES (avg)")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Sigma")
    ax2.set_title("Step Size Adaptation")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("es_comparison.png")
    plt.show()

def compare_es_variants():
    # Set parameters
    n = 10          # Dimension of the problem
    G = 1000        # Maximum number of generations
    k = 10          # Adaptation interval
    initial_sigma = 0.5  # Initial mutation step size
    c = 0.85        # Adaptation constant
    
    # Run basic (1+1)-ES with logging
    print("Running standard (1+1)-ES...")
    _, basic_fitness, basic_log = evolutionary_strategy_with_logging(n, G, k, initial_sigma, c)
    
    # Run self-adaptive ES
    print("Running self-adaptive ES...")
    _, self_adaptive_fitness, self_adaptive_log = evolutionary_strategy_self_adaptive(n, G)
    
    # Run ES with multiple sigmas
    print("Running ES with multiple sigmas...")
    _, multiple_sigmas_fitness, multiple_sigmas_log = evolutionary_strategy_multiple_sigmas(n, G)
    
    # Print comparison of final results
    print("\n===== ES Variant Comparison =====")
    print(f"Standard (1+1)-ES final fitness: {basic_fitness}")
    print(f"Self-Adaptive ES final fitness: {self_adaptive_fitness}")
    print(f"Multiple Sigmas ES final fitness: {multiple_sigmas_fitness}")
    
    # Plot comparison
    plot_results(basic_log, self_adaptive_log, multiple_sigmas_log)

if __name__ == "__main__":
    compare_es_variants()