import numpy as np
import math
import sys

def ackley_function(x):
    """
    Implements the Ackley function for optimization
    f(x) = -20 * exp(-0.2 * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(2π*x_i))) + 20 + e
    """
    n = len(x)
    sum1 = np.sum(np.square(x))
    sum2 = np.sum(np.cos(2 * np.pi * x))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    return term1 + term2 + 20 + np.exp(1)

def run_ep_optimization(n, pop_size, max_gen, alpha, sigma0):
    """
    Runs a single execution of the EP algorithm for optimizing the Ackley function
    
    Parameters:
    - n: number of decision variables
    - pop_size: population size
    - max_gen: maximum number of generations
    - alpha: parameter for sigma adaptation
    - sigma0: initial sigma value
    
    Returns:
    - best_solution: the solution vector with the lowest function value
    - best_fitness: the corresponding function value
    """
    # Initialize the population
    population = []
    for _ in range(pop_size):
        solution = np.random.uniform(-30, 30, n)
        sigma = sigma0
        population.append((solution, sigma))
    
    # Run the generation loop
    for generation in range(max_gen):
        # Generate offspring through mutation
        offspring = []
        for solution, sigma in population:
            # Mutate the solution vector using the current sigma
            new_solution = np.zeros_like(solution)
            for i in range(n):
                new_solution[i] = solution[i] + np.random.normal(0, sigma)
                
                # Enforce domain constraints
                if new_solution[i] < -30:
                    new_solution[i] = -30
                elif new_solution[i] > 30:
                    new_solution[i] = 30
            
            # Adapt sigma for the next generation
            new_sigma = sigma * np.exp(alpha * np.random.normal(0, 1))
            
            offspring.append((new_solution, new_sigma))
        
        # Evaluate fitness and select the best individuals
        combined = population + offspring
        with_fitness = [(sol, sig, ackley_function(sol)) for sol, sig in combined]
        with_fitness.sort(key=lambda x: x[2])  # Sort by fitness (lower is better)
        
        # Select the best for the next generation
        population = [(sol, sig) for sol, sig, _ in with_fitness[:pop_size]]
    
    # Return the best solution
    best_solution, _ = population[0]
    best_fitness = ackley_function(best_solution)
    
    return best_solution, best_fitness

def format_solution(x):
    """Format the solution array to match the expected output"""
    return '[' + ', '.join(f'{val:.8e}' for val in x) + ']'

def main():
    # Parse input
    try:
        n = int(input())
        pop_size, max_gen = map(int, input().split())
        alpha, sigma0 = map(float, input().split())
        
        # Check if multiple runs are specified
        try:
            M = int(input())
            multiple_runs = True
        except:
            multiple_runs = False
            M = 1
    except EOFError:
        # If direct input fails, try reading from stdin
        lines = sys.stdin.readlines()
        
        n = int(lines[0].strip())
        pop_size, max_gen = map(int, lines[1].strip().split())
        alpha, sigma0 = map(float, lines[2].strip().split())
        
        multiple_runs = False
        M = 1
        
        if len(lines) > 3:
            try:
                M = int(lines[3].strip())
                multiple_runs = True
            except:
                multiple_runs = False
    
    if not multiple_runs:
        # Single run (main exercise)
        best_solution, best_fitness = run_ep_optimization(n, pop_size, max_gen, alpha, sigma0)
        
        # Format and print the result
        print(format_solution(best_solution))
        print(f"f(x) = {best_fitness:.3f}")
    else:
        # Multiple runs (bonus extension)
        best_solutions = []
        best_fitness_values = []
        
        for _ in range(M):
            solution, fitness = run_ep_optimization(n, pop_size, max_gen, alpha, sigma0)
            best_solutions.append(solution)
            best_fitness_values.append(fitness)
        
        # Compute statistics
        best_idx = np.argmin(best_fitness_values)
        worst_idx = np.argmax(best_fitness_values)
        
        # Find median solution
        sorted_indices = np.argsort(best_fitness_values)
        median_idx = sorted_indices[M // 2]
        
        # Calculate mean and std deviation
        mean_fitness = np.mean(best_fitness_values)
        std_dev = np.std(best_fitness_values)
        
        # Output statistical report
        print(f"Best solution: {format_solution(best_solutions[best_idx])}")
        print(f"Best f(x): {best_fitness_values[best_idx]:.3f}")
        
        print(f"Worst solution: {format_solution(best_solutions[worst_idx])}")
        print(f"Worst f(x): {best_fitness_values[worst_idx]:.3f}")
        
        print(f"Median solution: {format_solution(best_solutions[median_idx])}")
        print(f"Median f(x): {best_fitness_values[median_idx]:.3f}")
        
        print(f"Mean fitness: {mean_fitness:.3f}")
        print(f"Standard deviation: {std_dev:.3f}")

if __name__ == "__main__":
    main()