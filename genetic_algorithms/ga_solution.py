import numpy as np
import random
import math
from statistics import median, mean, stdev
from collections import namedtuple

# Define named tuples for storing results
Solution = namedtuple('Solution', ['chromosome', 'x_values', 'objective'])

# Global parameters
POPULATION_SIZE = 100
P_C = 0.9  # Crossover probability 
P_M = 0.001  # Mutation probability for binary GA
P_M_REAL = 0.01  # Mutation probability for real GA
NUM_GENERATIONS = 100
NUM_RUNS = 30  # Number of runs (M)

# Domain boundaries for Beale function
X_MIN = -4.5
X_MAX = 4.5

# =============================================================
# Helper Functions
# =============================================================

def random_binary_string(length):
    """Generate a random binary string of the specified length."""
    return ''.join(random.choice('01') for _ in range(length))

def decode_binary_chromosome(chromosome, x_min, x_max):
    """
    Decode a binary chromosome to real values.
    For Beale function: 28-bit chromosome (14 bits for each variable)
    """
    # Split the chromosome into two parts
    x1_bits = chromosome[:14]
    x2_bits = chromosome[14:]
    
    # Convert binary to integer
    x1_int = int(x1_bits, 2)
    x2_int = int(x2_bits, 2)
    
    # Map to real values in the domain [x_min, x_max]
    max_int = 2**14 - 1
    x1 = x_min + (x_max - x_min) * x1_int / max_int
    x2 = x_min + (x_max - x_min) * x2_int / max_int
    
    return [x1, x2]

def beale_function(x):
    """
    Calculate the Beale function value.
    f(x1,x2) = (1.5 - x1 + x1*x2)^2 + (2.25 - x1 + x1*x2^2)^2 + (2.625 - x1 + x1*x2^3)^2
    """
    x1, x2 = x
    term1 = (1.5 - x1 + x1*x2)**2
    term2 = (2.25 - x1 + x1*(x2**2))**2
    term3 = (2.625 - x1 + x1*(x2**3))**2
    return term1 + term2 + term3

def ackley_function(x, a=20, b=0.2, c=2*np.pi):
    """
    Calculate the Ackley function value.
    f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c*x_i))) + a + exp(1)
    """
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(c * xi) for xi in x)
    
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    return term1 + term2 + a + np.exp(1)

def fitness_beale(x):
    """Calculate fitness for the Beale function (minimization problem)."""
    f = beale_function(x)
    return 1 / (f + 1)  # Transform to maximization problem

def fitness_ackley(x):
    """Calculate fitness for the Ackley function (minimization problem)."""
    f = ackley_function(x)
    return 1 / (f + 1)  # Transform to maximization problem

def stochastic_universal_sampling(population, fitnesses, num_parents):
    """
    Select parents using Stochastic Universal Sampling (SUS).
    Returns indices of selected parents.
    """
    # Calculate total fitness
    total_fitness = sum(fitnesses)
    
    # Calculate distance between pointers
    distance = total_fitness / num_parents
    
    # Random start point
    start = random.uniform(0, distance)
    
    # Generate pointers
    pointers = [start + i * distance for i in range(num_parents)]
    
    # Select parents
    selected = []
    for pointer in pointers:
        i = 0
        fitness_sum = fitnesses[0]
        while fitness_sum < pointer:
            i += 1
            if i >= len(fitnesses):
                i = len(fitnesses) - 1
                break
            fitness_sum += fitnesses[i]
        selected.append(i)
    
    return selected

def two_point_crossover(parent1, parent2):
    """
    Perform two-point crossover between two binary chromosomes.
    """
    length = len(parent1)
    
    # Choose two crossover points
    points = sorted(random.sample(range(1, length), 2))
    
    # Create offspring
    offspring1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
    offspring2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
    
    return offspring1, offspring2

def intermediate_crossover(parent1, parent2, alpha=0.5):
    """
    Perform intermediate crossover between two real-valued chromosomes.
    """
    offspring1 = []
    offspring2 = []
    
    for i in range(len(parent1)):
        # Generate random weights
        w1 = random.uniform(0, 1+alpha)
        w2 = random.uniform(0, 1+alpha)
        
        # Create offspring genes
        gene1 = w1 * parent1[i] + (1 - w1) * parent2[i]
        gene2 = w2 * parent2[i] + (1 - w2) * parent1[i]
        
        # Ensure genes are within domain bounds
        gene1 = max(X_MIN, min(X_MAX, gene1))
        gene2 = max(X_MIN, min(X_MAX, gene2))
        
        offspring1.append(gene1)
        offspring2.append(gene2)
    
    return offspring1, offspring2

def bitwise_mutation(chromosome, p_m):
    """
    Perform bitwise mutation on a binary chromosome.
    """
    mutated = ""
    for bit in chromosome:
        if random.random() < p_m:
            mutated += '1' if bit == '0' else '0'
        else:
            mutated += bit
    return mutated

def uniform_mutation(chromosome, p_m, x_min, x_max):
    """
    Perform uniform mutation on a real-valued chromosome.
    """
    mutated = chromosome.copy()
    for i in range(len(chromosome)):
        if random.random() < p_m:
            # Replace with random value in the valid range
            mutated[i] = random.uniform(x_min, x_max)
    return mutated

def get_statistics(results):
    """
    Calculate statistics from a list of Solution objects.
    Returns best, worst, median solutions, plus mean and std_dev of objective values.
    """
    # Sort solutions by objective value (lower is better for minimization)
    sorted_results = sorted(results, key=lambda x: x.objective)
    
    # Best, worst solutions
    best_solution = sorted_results[0]
    worst_solution = sorted_results[-1]
    
    # Median solution
    n = len(sorted_results)
    if n % 2 == 0:
        median_solution = sorted_results[n // 2 - 1]
    else:
        median_solution = sorted_results[n // 2]
    
    # Mean and standard deviation of objective values
    objective_values = [sol.objective for sol in results]
    mean_obj = mean(objective_values)
    std_dev_obj = stdev(objective_values) if len(objective_values) > 1 else 0
    
    return {
        'best': best_solution,
        'worst': worst_solution,
        'median': median_solution,
        'mean': mean_obj,
        'std_dev': std_dev_obj
    }

# =============================================================
# Exercise 1: GA for Beale Function (Binary Representation)
# =============================================================

def binary_ga_beale(pop_size=POPULATION_SIZE, p_c=P_C, p_m=P_M, num_generations=NUM_GENERATIONS):
    """
    Run a binary-encoded GA for the Beale function.
    """
    # Initialize population
    population = [random_binary_string(28) for _ in range(pop_size)]
    
    best_solution = None
    
    for generation in range(num_generations):
        # Evaluate fitness
        fitnesses = []
        solutions = []
        
        for chromosome in population:
            x_values = decode_binary_chromosome(chromosome, X_MIN, X_MAX)
            obj_value = beale_function(x_values)
            fitness = fitness_beale(x_values)
            
            fitnesses.append(fitness)
            solutions.append(Solution(chromosome, x_values, obj_value))
        
        # Update best solution
        generation_best = min(solutions, key=lambda s: s.objective)
        if best_solution is None or generation_best.objective < best_solution.objective:
            best_solution = generation_best
        
        # Select parents
        selected_indices = stochastic_universal_sampling(population, fitnesses, pop_size)
        parents = [population[i] for i in selected_indices]
        
        # Create offspring through crossover
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:  # Ensure we have a pair
                if random.random() < p_c:
                    child1, child2 = two_point_crossover(parents[i], parents[i+1])
                else:
                    child1, child2 = parents[i], parents[i+1]
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                # In case of odd population size
                offspring.append(parents[i])
        
        # Apply mutation
        for i in range(len(offspring)):
            offspring[i] = bitwise_mutation(offspring[i], p_m)
        
        # Elitism: Keep the best individual
        offspring_decoded = [decode_binary_chromosome(chrom, X_MIN, X_MAX) for chrom in offspring]
        offspring_fitness = [fitness_beale(x) for x in offspring_decoded]
        offspring_obj_values = [beale_function(x) for x in offspring_decoded]
        
        # Find worst offspring
        worst_idx = offspring_obj_values.index(max(offspring_obj_values))
        
        # If the best solution is better than the worst offspring, replace it
        if best_solution is not None and best_solution.objective < offspring_obj_values[worst_idx]:
            offspring[worst_idx] = best_solution.chromosome
        
        # Replace population with offspring
        population = offspring
    
    return best_solution

# =============================================================
# Exercise 2: GA for Beale Function (Real Representation)
# =============================================================

def real_ga_beale(pop_size=POPULATION_SIZE, p_c=P_C, p_m=P_M_REAL, num_generations=NUM_GENERATIONS):
    """
    Run a real-encoded GA for the Beale function.
    """
    # Initialize population
    population = [[random.uniform(X_MIN, X_MAX), random.uniform(X_MIN, X_MAX)] for _ in range(pop_size)]
    
    best_solution = None
    
    for generation in range(num_generations):
        # Evaluate fitness
        fitnesses = []
        solutions = []
        
        for chromosome in population:
            obj_value = beale_function(chromosome)
            fitness = fitness_beale(chromosome)
            
            fitnesses.append(fitness)
            solutions.append(Solution(chromosome.copy(), chromosome.copy(), obj_value))
        
        # Update best solution
        generation_best = min(solutions, key=lambda s: s.objective)
        if best_solution is None or generation_best.objective < best_solution.objective:
            best_solution = generation_best
        
        # Select parents
        selected_indices = stochastic_universal_sampling(population, fitnesses, pop_size)
        parents = [population[i].copy() for i in selected_indices]
        
        # Create offspring through crossover
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:  # Ensure we have a pair
                if random.random() < p_c:
                    child1, child2 = intermediate_crossover(parents[i], parents[i+1])
                else:
                    child1, child2 = parents[i].copy(), parents[i+1].copy()
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                # In case of odd population size
                offspring.append(parents[i].copy())
        
        # Apply mutation
        for i in range(len(offspring)):
            offspring[i] = uniform_mutation(offspring[i], p_m, X_MIN, X_MAX)
        
        # Elitism: Keep the best individual
        offspring_fitness = [fitness_beale(x) for x in offspring]
        offspring_obj_values = [beale_function(x) for x in offspring]
        
        # Find worst offspring
        worst_idx = offspring_obj_values.index(max(offspring_obj_values))
        
        # If the best solution is better than the worst offspring, replace it
        if best_solution is not None and best_solution.objective < offspring_obj_values[worst_idx]:
            offspring[worst_idx] = best_solution.chromosome.copy()
        
        # Replace population with offspring
        population = [chrom.copy() for chrom in offspring]
    
    return best_solution

# =============================================================
# Exercise 3: GAs for Ackley Function (Binary & Real)
# =============================================================

def binary_ga_ackley(pop_size=POPULATION_SIZE, p_c=P_C, p_m=P_M, num_generations=NUM_GENERATIONS):
    """
    Run a binary-encoded GA for the Ackley function.
    """
    # Initialize population
    population = [random_binary_string(28) for _ in range(pop_size)]
    
    best_solution = None
    
    for generation in range(num_generations):
        # Evaluate fitness
        fitnesses = []
        solutions = []
        
        for chromosome in population:
            x_values = decode_binary_chromosome(chromosome, X_MIN, X_MAX)
            obj_value = ackley_function(x_values)
            fitness = fitness_ackley(x_values)
            
            fitnesses.append(fitness)
            solutions.append(Solution(chromosome, x_values, obj_value))
        
        # Update best solution
        generation_best = min(solutions, key=lambda s: s.objective)
        if best_solution is None or generation_best.objective < best_solution.objective:
            best_solution = generation_best
        
        # Select parents
        selected_indices = stochastic_universal_sampling(population, fitnesses, pop_size)
        parents = [population[i] for i in selected_indices]
        
        # Create offspring through crossover
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:  # Ensure we have a pair
                if random.random() < p_c:
                    child1, child2 = two_point_crossover(parents[i], parents[i+1])
                else:
                    child1, child2 = parents[i], parents[i+1]
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                # In case of odd population size
                offspring.append(parents[i])
        
        # Apply mutation
        for i in range(len(offspring)):
            offspring[i] = bitwise_mutation(offspring[i], p_m)
        
        # Elitism: Keep the best individual
        offspring_decoded = [decode_binary_chromosome(chrom, X_MIN, X_MAX) for chrom in offspring]
        offspring_fitness = [fitness_ackley(x) for x in offspring_decoded]
        offspring_obj_values = [ackley_function(x) for x in offspring_decoded]
        
        # Find worst offspring
        worst_idx = offspring_obj_values.index(max(offspring_obj_values))
        
        # If the best solution is better than the worst offspring, replace it
        if best_solution is not None and best_solution.objective < offspring_obj_values[worst_idx]:
            offspring[worst_idx] = best_solution.chromosome
        
        # Replace population with offspring
        population = offspring
    
    return best_solution

def real_ga_ackley(pop_size=POPULATION_SIZE, p_c=P_C, p_m=P_M_REAL, num_generations=NUM_GENERATIONS, num_vars=2):
    """
    Run a real-encoded GA for the Ackley function.
    """
    # Initialize population
    population = [[random.uniform(X_MIN, X_MAX) for _ in range(num_vars)] for _ in range(pop_size)]
    
    best_solution = None
    
    for generation in range(num_generations):
        # Evaluate fitness
        fitnesses = []
        solutions = []
        
        for chromosome in population:
            obj_value = ackley_function(chromosome)
            fitness = fitness_ackley(chromosome)
            
            fitnesses.append(fitness)
            solutions.append(Solution(chromosome.copy(), chromosome.copy(), obj_value))
        
        # Update best solution
        generation_best = min(solutions, key=lambda s: s.objective)
        if best_solution is None or generation_best.objective < best_solution.objective:
            best_solution = generation_best
        
        # Select parents
        selected_indices = stochastic_universal_sampling(population, fitnesses, pop_size)
        parents = [population[i].copy() for i in selected_indices]
        
        # Create offspring through crossover
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:  # Ensure we have a pair
                if random.random() < p_c:
                    child1, child2 = intermediate_crossover(parents[i], parents[i+1])
                else:
                    child1, child2 = parents[i].copy(), parents[i+1].copy()
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                # In case of odd population size
                offspring.append(parents[i].copy())
        
        # Apply mutation
        for i in range(len(offspring)):
            offspring[i] = uniform_mutation(offspring[i], p_m, X_MIN, X_MAX)
        
        # Elitism: Keep the best individual
        offspring_fitness = [fitness_ackley(x) for x in offspring]
        offspring_obj_values = [ackley_function(x) for x in offspring]
        
        # Find worst offspring
        worst_idx = offspring_obj_values.index(max(offspring_obj_values))
        
        # If the best solution is better than the worst offspring, replace it
        if best_solution is not None and best_solution.objective < offspring_obj_values[worst_idx]:
            offspring[worst_idx] = best_solution.chromosome.copy()
        
        # Replace population with offspring
        population = [chrom.copy() for chrom in offspring]
    
    return best_solution

# =============================================================
# Exercise 4: Run both GA versions on both test problems
# =============================================================

def run_experiment(ga_function, num_runs=NUM_RUNS, **kwargs):
    """
    Run a GA experiment multiple times and return statistics.
    """
    results = []
    
    for run in range(num_runs):
        solution = ga_function(**kwargs)
        results.append(solution)
    
    return get_statistics(results)

def print_comparative_table(results):
    """
    Print a comparative table of results.
    """
    print("\n" + "="*80)
    print(f"{'COMPARATIVE RESULTS':^80}")
    print("="*80)
    
    headers = ["Algorithm", "Problem", "Best Obj", "Best Solution", 
               "Worst Obj", "Median Obj", "Mean Obj", "Std Dev"]
    
    # Print headers
    print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<25} "
          f"{headers[4]:<10} {headers[5]:<10} {headers[6]:<10} {headers[7]:<10}")
    print("-"*110)
    
    # Print results
    for algorithm, problems in results.items():
        for problem, stats in problems.items():
            best_sol_str = f"x={[round(x, 4) for x in stats['best'].x_values]}"
            print(f"{algorithm:<15} {problem:<10} {stats['best'].objective:<10.6f} "
                  f"{best_sol_str:<25} {stats['worst'].objective:<10.6f} "
                  f"{stats['median'].objective:<10.6f} {stats['mean']:<10.6f} "
                  f"{stats['std_dev']:<10.6f}")
    
    print("="*110)

def run_all_experiments():
    """
    Run all GA experiments and print comparative results.
    """
    results = {
        "Binary GA": {
            "Beale": run_experiment(binary_ga_beale),
            "Ackley": run_experiment(binary_ga_ackley)
        },
        "Real GA": {
            "Beale": run_experiment(real_ga_beale),
            "Ackley-2D": run_experiment(real_ga_ackley, num_vars=2),
            "Ackley-5D": run_experiment(real_ga_ackley, num_vars=5),
            "Ackley-10D": run_experiment(real_ga_ackley, num_vars=10),
            "Ackley-20D": run_experiment(real_ga_ackley, num_vars=20)
        }
    }
    
    print_comparative_table(results)
    
    return results

# =============================================================
# Exercise 5: GA for Traveling Salesman Problem (TSP)
# =============================================================

def read_tsp_instance(file_path=None):
    """
    Read a TSP instance from a file or from standard input.
    """
    if file_path:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    else:
        # Read from stdin
        print("Enter TSP instance (first line: number of cities):")
        num_cities = int(input())
        print(f"Enter GA parameters (p_c, p_m, population_size):")
        params_line = input()
        
        lines = [str(num_cities), params_line]
        
        print(f"Enter cost matrix ({num_cities-1} lines):")
        for i in range(num_cities-1):
            lines.append(input())
    
    # Parse the input
    num_cities = int(lines[0].strip())
    
    params = list(map(float, lines[1].strip().split()))
    p_c, p_m, pop_size = params if len(params) == 3 else (0.9, 0.01, 100)
    
    # Parse cost matrix
    cost_matrix = [[0 for _ in range(num_cities)] for _ in range(num_cities)]
    
    line_index = 2
    for i in range(num_cities - 1):
        costs = list(map(int, lines[line_index].strip().split()))
        for j, cost in enumerate(costs):
            cost_matrix[i][i+1+j] = cost
            cost_matrix[i+1+j][i] = cost  # Symmetric matrix
        line_index += 1
    
    return num_cities, p_c, p_m, int(pop_size), cost_matrix

def initialize_tsp_population(pop_size, num_cities):
    """
    Initialize a population of permutations for TSP.
    Always start with city 0 and permute the rest.
    """
    population = []
    
    for _ in range(pop_size):
        # Create a random permutation of cities 1 to n-1
        perm = list(range(1, num_cities))
        random.shuffle(perm)
        # Add city 0 at the beginning
        perm = [0] + perm
        population.append(perm)
    
    return population

def calculate_route_cost(route, cost_matrix):
    """
    Calculate the total cost of a route.
    """
    total_cost = 0
    
    for i in range(len(route) - 1):
        total_cost += cost_matrix[route[i]][route[i+1]]
    
    # Add cost from last city back to the first (city 0)
    total_cost += cost_matrix[route[-1]][route[0]]
    
    return total_cost

def order_crossover(parent1, parent2):
    """
    Perform order crossover (OX) for TSP permutations.
    """
    size = len(parent1)
    
    # Select a substring
    start, end = sorted(random.sample(range(1, size), 2))  # Skip the first city (always 0)
    
    # Create children
    child1 = [-1] * size
    child2 = [-1] * size
    
    # Always keep city 0 at the beginning
    child1[0] = 0
    child2[0] = 0
    
    # Copy the selected segment
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]
    
    # Fill remaining positions
    fill_index1 = end if end < size else 1
    fill_index2 = end if end < size else 1
    
    for i in range(1, size):
        # Fill child1 with cities from parent2
        if child1[fill_index1] == -1:
            city = parent2[i]
            if city not in child1:
                child1[fill_index1] = city
                fill_index1 = (fill_index1 + 1) % size
                if fill_index1 == 0:  # Skip city 0
                    fill_index1 = 1
        
        # Fill child2 with cities from parent1
        if child2[fill_index2] == -1:
            city = parent1[i]
            if city not in child2:
                child2[fill_index2] = city
                fill_index2 = (fill_index2 + 1) % size
                if fill_index2 == 0:  # Skip city 0
                    fill_index2 = 1
    
    return child1, child2

def swap_mutation(route, p_m):
    """
    Perform swap mutation on a TSP route.
    """
    mutated = route.copy()
    
    if random.random() < p_m:
        # Select two positions to swap (excluding city 0)
        i, j = random.sample(range(1, len(route)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    
    return mutated

def tsp_ga(num_cities, cost_matrix, pop_size=100, p_c=0.9, p_m=0.01, num_generations=100):
    """
    Run a GA for the Traveling Salesman Problem.
    """
    # Initialize population
    population = initialize_tsp_population(pop_size, num_cities)
    
    best_solution = None
    best_cost = float('inf')
    
    for generation in range(num_generations):
        # Evaluate fitness (inverse of route cost)
        route_costs = [calculate_route_cost(route, cost_matrix) for route in population]
        fitnesses = [1 / cost for cost in route_costs]
        
        # Update best solution
        min_cost_idx = route_costs.index(min(route_costs))
        if route_costs[min_cost_idx] < best_cost:
            best_cost = route_costs[min_cost_idx]
            best_solution = population[min_cost_idx].copy()
        
        # Select parents
        selected_indices = stochastic_universal_sampling(population, fitnesses, pop_size)
        parents = [population[i].copy() for i in selected_indices]
        
        # Create offspring through crossover
        offspring = []
        for i in range(0, pop_size, 2):
            if i + 1 < pop_size:  # Ensure we have a pair
                if random.random() < p_c:
                    child1, child2 = order_crossover(parents[i], parents[i+1])
                else:
                    child1, child2 = parents[i].copy(), parents[i+1].copy()
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                # In case of odd population size
                offspring.append(parents[i].copy())
        
        # Apply mutation
        for i in range(len(offspring)):
            offspring[i] = swap_mutation(offspring[i], p_m)
        
        # Elitism: Keep the best solution
        if best_solution:
            # Calculate costs of offspring
            offspring_costs = [calculate_route_cost(route, cost_matrix) for route in offspring]
            
            # Find worst offspring
            worst_idx = offspring_costs.index(max(offspring_costs))
            
            # If the best solution is better than the worst offspring, replace it
            if best_cost < offspring_costs[worst_idx]:
                offspring[worst_idx] = best_solution.copy()
        
        # Replace population with offspring
        population = [route.copy() for route in offspring]
    
    return best_solution, best_cost

# =============================================================
# Exercise 6: Multiple Runs for the TSP GA with Statistical Reporting
# =============================================================

def run_tsp_experiment(num_cities, cost_matrix, num_runs=30, **kwargs):
    """
    Run the TSP GA multiple times and return statistics.
    """
    results = []
    
    for run in range(num_runs):
        route, cost = tsp_ga(num_cities, cost_matrix, **kwargs)
        results.append((route, cost))
    
    # Sort by cost
    sorted_results = sorted(results, key=lambda x: x[1])
    
    # Calculate statistics
    costs = [result[1] for result in results]
    mean_cost = mean(costs)
    std_dev = stdev(costs) if len(costs) > 1 else 0
    
    # Best, worst, median solutions
    best = sorted_results[0]
    worst = sorted_results[-1]
    
    n = len(sorted_results)
    median = sorted_results[n // 2] if n % 2 != 0 else sorted_results[n // 2 - 1]
    
    return {
        'best': best,
        'worst': worst,
        'median': median,
        'mean': mean_cost,
        'std_dev': std_dev
    }

def print_tsp_results(results):
    """
    Print results from TSP experiments.
    """
    print("\n" + "="*80)
    print(f"{'TSP RESULTS':^80}")
    print("="*80)
    
    print(f"Best route: {results['best'][0]}")
    print(f"Best cost: {results['best'][1]}")
    print(f"Worst cost: {results['worst'][1]}")
    print(f"Median cost: {results['median'][1]}")
    print(f"Mean cost: {results['mean']:.2f}")
    print(f"Standard deviation: {results['std_dev']:.2f}")
    print("="*80)

# =============================================================
# Main Execution
# =============================================================

def main():
    """
    Main function to run all exercises.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Genetic Algorithm Implementations")
    parser.add_argument('--function', type=str, choices=['beale', 'ackley', 'tsp', 'all'], 
                       default='all', help='Function to optimize')
    parser.add_argument('--tsp-file', type=str, help='TSP instance file')
    parser.add_argument('--runs', type=int, default=30, help='Number of runs')
    
    args = parser.parse_args()
    
    if args.function in ['beale', 'ackley', 'all']:
        # Run benchmark function experiments
        run_all_experiments()
    
    if args.function in ['tsp', 'all']:
        # Run TSP experiment
        if args.tsp_file:
            # Read from file
            num_cities, p_c, p_m, pop_size, cost_matrix = read_tsp_instance(args.tsp_file)
        else:
            # Read from stdin
            num_cities, p_c, p_m, pop_size, cost_matrix = read_tsp_instance()
        
        tsp_results = run_tsp_experiment(
            num_cities, cost_matrix, num_runs=args.runs,
            pop_size=pop_size, p_c=p_c, p_m=p_m
        )
        
        print_tsp_results(tsp_results)
        
        # Print just the best route and cost (as per the requirements)
        best_route = ''.join(map(str, tsp_results['best'][0]))
        best_cost = tsp_results['best'][1]
        print(f"\n{best_route}\n{best_cost}")

if __name__ == "__main__":
    main()