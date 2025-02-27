import numpy as np
import math
import statistics

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

def initialize_individual(n):
    """
    Initialize an individual with n decision variables and a mutation step size.
    
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

def mutate(individual):
    """
    Mutate an individual by adding Gaussian noise scaled by sigma.
    
    Parameters:
        individual (np.array): Individual to mutate [x_1, x_2, ..., x_n, sigma]
    
    Returns:
        np.array: Mutated individual [x'_1, x'_2, ..., x'_n, sigma]
    """
    n = len(individual) - 1  # Number of decision variables
    sigma = individual[n]    # Extract sigma
    
    # Create a mutated copy of the individual
    mutated = individual.copy()
    
    # Mutate each decision variable using x'_i = x_i + sigma * N(0,1)
    for i in range(n):
        mutated[i] = individual[i] + sigma * np.random.normal(0, 1)
    
    # Sigma remains unchanged during mutation
    return mutated

def evolutionary_strategy(n, G, k, initial_sigma, c):
    """
    Implement the (1+1)-ES algorithm to minimize the Ackley function.
    
    Parameters:
        n (int): Dimension of the problem (number of decision variables)
        G (int): Maximum number of generations
        k (int): Adaptation interval for sigma
        initial_sigma (float): Initial value for sigma
        c (float): Adaptation constant for sigma
    
    Returns:
        tuple: (best_individual, best_fitness)
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
    
    return best_individual, best_fitness

def run_multiple_executions(M, n, G, k, initial_sigma, c):
    """
    Run the (1+1)-ES algorithm M times and report statistics.
    
    Parameters:
        M (int): Number of independent runs
        n (int): Dimension of the problem
        G (int): Maximum number of generations
        k (int): Adaptation interval
        initial_sigma (float): Initial sigma value
        c (float): Adaptation constant
    
    Returns:
        dict: Statistical results
    """
    objective_values = []
    best_solutions = []
    
    # Perform M independent runs
    for i in range(M):
        print(f"Run {i+1}/{M}...")
        best_individual, best_fitness = evolutionary_strategy(n, G, k, initial_sigma, c)
        objective_values.append(best_fitness)
        best_solutions.append(best_individual[:n])  # Store only the decision variables
    
    # Calculate statistics
    stats = {
        "best_value": min(objective_values),
        "worst_value": max(objective_values),
        "median_value": statistics.median(objective_values),
        "mean_value": statistics.mean(objective_values),
        "std_dev": statistics.stdev(objective_values),
        "best_solution": best_solutions[np.argmin(objective_values)],
        "worst_solution": best_solutions[np.argmax(objective_values)]
    }
    
    return stats

def main():
    # Set parameters
    n = 10          # Dimension of the problem
    G = 1000        # Maximum number of generations
    k = 10          # Adaptation interval
    initial_sigma = 0.5  # Initial mutation step size
    c = 0.85        # Adaptation constant (should be in [0.817, 1])
    
    # Single run test
    print("Testing a single run of (1+1)-ES...")
    best_individual, best_fitness = evolutionary_strategy(n, G, k, initial_sigma, c)
    print(f"Best fitness: {best_fitness}")
    print(f"Best individual: {best_individual[:n]}")
    
    # Number of independent runs
    M = int(input("\nEnter the number of independent runs (M): "))
    
    # Run multiple executions and get statistics
    stats = run_multiple_executions(M, n, G, k, initial_sigma, c)
    
    # Report results
    print("\n===== (1+1)-ES for Ackley Function =====")
    print(f"Parameters: n={n}, G={G}, k={k}, initial_sigma={initial_sigma}, c={c}")
    print("\n===== Statistical Results =====")
    print(f"Best objective value: {stats['best_value']}")
    print(f"Worst objective value: {stats['worst_value']}")
    print(f"Median objective value: {stats['median_value']}")
    print(f"Mean objective value: {stats['mean_value']}")
    print(f"Standard deviation: {stats['std_dev']}")
    
    print("\nBest solution vector:")
    print(stats['best_solution'])
    
    print("\nWorst solution vector:")
    print(stats['worst_solution'])

if __name__ == "__main__":
    main()