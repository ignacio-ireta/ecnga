import random
import math
import statistics
import sys
import argparse
from typing import List, Tuple, Dict, Set, Optional

def read_input():
    """
    Read the standard input for the TSP problem.
    Returns:
        N: Number of cities
        Imax: Maximum number of iterations
        cost_matrix: Matrix of costs between cities
    """
    N = int(input())
    Imax = int(input())
    
    # Initialize cost matrix with zeros
    cost_matrix = [[0 for _ in range(N)] for _ in range(N)]
    
    # Read the cost matrix - for each row i, we read costs to higher-indexed cities
    for i in range(N-1):  # There are N-1 rows to read
        values = list(map(int, input().split()))
        
        # The i-th row contains costs from city i to cities i+1, i+2, ..., N-1
        for j in range(len(values)):
            cost_matrix[i][i+j+1] = values[j]
            cost_matrix[i+j+1][i] = values[j]  # Make it symmetric
    
    return N, Imax, cost_matrix

def read_instance_from_file(filename: str):
    """
    Read a TSP instance from a file.
    Args:
        filename: Path to the file
    Returns:
        N: Number of cities
        Imax: Maximum number of iterations
        cost_matrix: Matrix of costs between cities
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        N = int(lines[0].strip())
        Imax = int(lines[1].strip())
        
        # Initialize cost matrix with zeros
        cost_matrix = [[0 for _ in range(N)] for _ in range(N)]
        
        # Read the cost matrix - for each row i, we read costs to higher-indexed cities
        for i in range(N-1):  # There are N-1 rows to read
            values = list(map(int, lines[i+2].strip().split()))
            
            # The i-th row contains costs from city i to cities i+1, i+2, ..., N-1
            for j in range(len(values)):
                cost_matrix[i][i+j+1] = values[j]
                cost_matrix[i+j+1][i] = values[j]  # Make it symmetric
    
    return N, Imax, cost_matrix

def greedy_initial_solution(N: int, cost_matrix: List[List[int]]) -> List[int]:
    """
    Generate an initial solution using a greedy algorithm starting from city 0.
    Args:
        N: Number of cities
        cost_matrix: Matrix of costs between cities
    Returns:
        route: A list representing the tour
    """
    # Start from city 0
    route = [0]
    unvisited = list(range(1, N))
    
    # Construct the route greedily
    while unvisited:
        current_city = route[-1]
        next_city = min(unvisited, key=lambda city: cost_matrix[current_city][city])
        route.append(next_city)
        unvisited.remove(next_city)
    
    return route

def calculate_cost(route: List[int], cost_matrix: List[List[int]]) -> int:
    """
    Calculate the total cost of a route.
    Args:
        route: A list representing the tour
        cost_matrix: Matrix of costs between cities
    Returns:
        total_cost: The total cost of the route
    """
    total_cost = 0
    N = len(route)
    
    for i in range(N - 1):
        total_cost += cost_matrix[route[i]][route[i+1]]
    
    # Add cost to return to the starting city
    total_cost += cost_matrix[route[N-1]][route[0]]
    
    return total_cost

def tabu_search(N: int, Imax: int, cost_matrix: List[List[int]]) -> Tuple[List[int], int]:
    """
    Tabu Search algorithm for TSP.
    Args:
        N: Number of cities
        Imax: Maximum number of iterations
        cost_matrix: Matrix of costs between cities
    Returns:
        best_solution: The best route found
        best_cost: The cost of the best route
    """
    # Generate initial solution
    current_solution = greedy_initial_solution(N, cost_matrix)
    best_solution = current_solution.copy()
    current_cost = calculate_cost(current_solution, cost_matrix)
    best_cost = current_cost
    
    # Initialize tabu list
    tabu_tenure = math.ceil(N / 2)
    tabu_list = {}  # Dictionary to track tabu moves
    
    # Main loop
    for iteration in range(Imax):
        # Randomly select a city to move (except the starting city 0)
        move_pos = random.randrange(1, N)
        move_city = current_solution[move_pos]
        
        # Generate neighborhood by moving the selected city to each position
        best_neighbor = None
        best_neighbor_cost = float('inf')
        
        for new_pos in range(1, N):  # Keep city 0 fixed at the start
            if new_pos == move_pos:
                continue
            
            # Create a neighbor by moving the city
            neighbor = current_solution.copy()
            neighbor.pop(move_pos)
            neighbor.insert(new_pos, move_city)
            
            # Calculate the cost
            neighbor_cost = calculate_cost(neighbor, cost_matrix)
            
            # Check if the move is tabu
            move_key = (move_city, new_pos)
            is_tabu = move_key in tabu_list and tabu_list[move_key] > 0
            
            # Apply aspiration criterion - accept if better than the best known
            if is_tabu and neighbor_cost >= best_cost:
                continue
            
            # Update the best neighbor if better
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
        
        # Update the current solution to the best neighbor
        if best_neighbor:
            current_solution = best_neighbor
            current_cost = best_neighbor_cost
            
            # Update the best solution if improved
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
            
            # Add the move to the tabu list
            move_city_idx = best_neighbor.index(move_city)
            tabu_list[(move_city, move_city_idx)] = tabu_tenure  # Prohibit moving this city to this position
            tabu_list[(move_city, move_pos)] = tabu_tenure  # Prohibit moving this city back to original position
        
        # Decrease tabu tenures and remove expired ones
        expired_moves = []
        for move in tabu_list:
            tabu_list[move] -= 1
            if tabu_list[move] <= 0:
                expired_moves.append(move)
        
        for move in expired_moves:
            del tabu_list[move]
    
    return best_solution, best_cost

def multi_run_tabu_search(M: int, N: int, Imax: int, cost_matrix: List[List[int]]) -> tuple:
    """
    Execute the Tabu Search algorithm multiple times and collect statistics.
    Args:
        M: Number of executions
        N: Number of cities
        Imax: Maximum number of iterations
        cost_matrix: Matrix of costs between cities
    Returns:
        best_solution: The best solution found across all runs
        worst_solution: The worst solution found across all runs
        median_solution: The median solution based on cost
        mean_cost: The mean cost of all solutions
        std_dev: The standard deviation of the costs
    """
    results = []
    
    for run in range(M):
        print(f"Running iteration {run+1}/{M}...")
        best_route, best_cost = tabu_search(N, Imax, cost_matrix)
        results.append((best_route, best_cost))
    
    # Sort by cost
    results.sort(key=lambda x: x[1])
    
    # Calculate statistics
    costs = [cost for _, cost in results]
    mean_cost = statistics.mean(costs)
    std_dev = statistics.stdev(costs) if len(costs) > 1 else 0
    
    # Get best, worst, and median
    best_solution = results[0]
    worst_solution = results[-1]
    
    # For median, handle even number of runs
    if M % 2 == 0:
        median_idx = M // 2 - 1  # 0-indexed, so subtract 1
    else:
        median_idx = M // 2
    
    median_solution = results[median_idx]
    
    return best_solution, worst_solution, median_solution, mean_cost, std_dev

def tabu_search_enhanced(N: int, Imax: int, cost_matrix: List[List[int]], 
                        dynamic_tenure: bool = False, 
                        diversification: bool = False, 
                        swap_moves: bool = False) -> Tuple[List[int], int]:
    """
    Enhanced Tabu Search algorithm for TSP with additional features.
    Args:
        N: Number of cities
        Imax: Maximum number of iterations
        cost_matrix: Matrix of costs between cities
        dynamic_tenure: Whether to use dynamic tabu tenure
        diversification: Whether to use frequency-based diversification
        swap_moves: Whether to use swap moves in addition to insert moves
    Returns:
        best_solution: The best route found
        best_cost: The cost of the best route
    """
    # Generate initial solution
    current_solution = greedy_initial_solution(N, cost_matrix)
    best_solution = current_solution.copy()
    current_cost = calculate_cost(current_solution, cost_matrix)
    best_cost = current_cost
    
    # Initialize tabu list
    base_tabu_tenure = math.ceil(N / 2)
    tabu_list = {}  # Dictionary to track tabu moves
    
    # For diversification
    frequency = {(i, j): 0 for i in range(N) for j in range(N) if i != j} if diversification else None
    
    # No improvement counter for dynamic tenure
    no_improvement = 0
    
    # Main loop
    for iteration in range(Imax):
        # Choose between insert and swap moves
        use_swap = swap_moves and random.random() < 0.5
        
        # Generate neighborhood
        best_neighbor = None
        best_neighbor_cost = float('inf')
        
        if use_swap:
            # Randomly select two positions to swap (except the starting city 0)
            pos1 = random.randrange(1, N)
            pos2 = random.randrange(1, N)
            while pos1 == pos2:
                pos2 = random.randrange(1, N)
            
            # Swap cities at the selected positions
            for _ in range(1):  # Just one swap combination
                neighbor = current_solution.copy()
                neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
                
                # Calculate the cost
                neighbor_cost = calculate_cost(neighbor, cost_matrix)
                
                # Apply diversification if enabled
                if diversification:
                    # Penalize frequently used edges
                    for i in range(N-1):
                        from_city = neighbor[i]
                        to_city = neighbor[i+1]
                        neighbor_cost += 0.01 * frequency[(from_city, to_city)]
                
                # Check if the move is tabu
                city1, city2 = current_solution[pos1], current_solution[pos2]
                swap_key1 = (city1, pos2)
                swap_key2 = (city2, pos1)
                is_tabu = (swap_key1 in tabu_list and tabu_list[swap_key1] > 0) or \
                          (swap_key2 in tabu_list and tabu_list[swap_key2] > 0)
                
                # Apply aspiration criterion
                if is_tabu and neighbor_cost >= best_cost:
                    continue
                
                # Update the best neighbor if better
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
        else:
            # Insert move (original approach)
            move_pos = random.randrange(1, N)
            move_city = current_solution[move_pos]
            
            for new_pos in range(1, N):
                if new_pos == move_pos:
                    continue
                
                # Create a neighbor by moving the city
                neighbor = current_solution.copy()
                neighbor.pop(move_pos)
                neighbor.insert(new_pos, move_city)
                
                # Calculate the cost
                neighbor_cost = calculate_cost(neighbor, cost_matrix)
                
                # Apply diversification if enabled
                if diversification:
                    for i in range(N-1):
                        from_city = neighbor[i]
                        to_city = neighbor[i+1]
                        neighbor_cost += 0.01 * frequency[(from_city, to_city)]
                
                # Check if the move is tabu
                move_key = (move_city, new_pos)
                is_tabu = move_key in tabu_list and tabu_list[move_key] > 0
                
                # Apply aspiration criterion
                if is_tabu and neighbor_cost >= best_cost:
                    continue
                
                # Update the best neighbor if better
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
        
        # Update the current solution to the best neighbor
        if best_neighbor:
            # Update frequencies if using diversification
            if diversification:
                for i in range(N-1):
                    from_city = best_neighbor[i]
                    to_city = best_neighbor[i+1]
                    frequency[(from_city, to_city)] += 1
                # Also update for the loop back
                from_city = best_neighbor[-1]
                to_city = best_neighbor[0]
                frequency[(from_city, to_city)] += 1
            
            current_solution = best_neighbor
            current_cost = best_neighbor_cost
            
            # Update the best solution if improved
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
                no_improvement = 0
            else:
                no_improvement += 1
            
            # Calculate tenure based on dynamic setting
            if dynamic_tenure:
                # Increase tenure when stuck in local optima
                tenure = base_tabu_tenure + min(no_improvement // 5, base_tabu_tenure)
            else:
                tenure = base_tabu_tenure
            
            # Update tabu list based on the move type
            if use_swap:
                pos1, pos2 = -1, -1
                for i in range(N):
                    if current_solution[i] != best_neighbor[i]:
                        if pos1 == -1:
                            pos1 = i
                        else:
                            pos2 = i
                            break
                
                if pos1 != -1 and pos2 != -1:
                    city1, city2 = best_neighbor[pos1], best_neighbor[pos2]
                    tabu_list[(city1, pos2)] = tenure
                    tabu_list[(city2, pos1)] = tenure
            else:
                # Find the city that was moved
                for i in range(N):
                    if current_solution[i] != best_neighbor[i]:
                        city = current_solution[i] if current_solution[i] not in best_neighbor[:i+1] else best_neighbor[i]
                        old_pos = current_solution.index(city)
                        new_pos = best_neighbor.index(city)
                        tabu_list[(city, old_pos)] = tenure
                        break
        
        # Decrease tabu tenures and remove expired ones
        expired_moves = []
        for move in tabu_list:
            tabu_list[move] -= 1
            if tabu_list[move] <= 0:
                expired_moves.append(move)
        
        for move in expired_moves:
            del tabu_list[move]
    
    return best_solution, best_cost

def main():
    """Main function to handle different execution modes."""
    parser = argparse.ArgumentParser(description='Tabu Search for TSP')
    parser.add_argument('--mode', type=str, choices=['basic', 'extension', 'challenge'], 
                        default='basic', help='Mode to run')
    parser.add_argument('--file', type=str, help='Instance file for extension/challenge modes')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs for extension/challenge modes')
    
    args = parser.parse_args()
    
    if args.mode == 'basic':
        # Basic mode: Read from stdin and run single tabu search
        N, Imax, cost_matrix = read_input()
        best_route, best_cost = tabu_search(N, Imax, cost_matrix)
        
        # Print the result
        route_str = ' '.join(map(str, best_route))
        print(f"{route_str} {best_cost}")
    
    elif args.mode == 'extension':
        # Extension mode: Run multiple times and report statistics
        if args.file:
            filename = args.file
        else:
            filename = input("Enter the filename of the instance: ")
        
        M = args.runs if args.runs else int(input("Enter the number of executions: "))
        
        N, Imax, cost_matrix = read_instance_from_file(filename)
        best, worst, median, mean, std_dev = multi_run_tabu_search(M, N, Imax, cost_matrix)
        
        # Print results
        print("\nResults:")
        print("Best solution:", ' '.join(map(str, best[0])), best[1])
        print("Worst solution:", ' '.join(map(str, worst[0])), worst[1])
        print("Median solution:", ' '.join(map(str, median[0])), median[1])
        print(f"Mean cost: {mean:.2f}")
        print(f"Standard deviation: {std_dev:.2f}")
    
    elif args.mode == 'challenge':
        # Challenge mode: Compare different enhancements
        if args.file:
            filename = args.file
        else:
            filename = input("Enter the filename of the instance: ")
        
        M = args.runs if args.runs else int(input("Enter the number of executions: "))
        
        N, Imax, cost_matrix = read_instance_from_file(filename)
        
        # Run with different configurations
        strategies = [
            ("Basic Tabu Search", False, False, False),
            ("With Dynamic Tenure", True, False, False),
            ("With Diversification", False, True, False),
            ("With Swap Moves", False, False, True),
            ("Full Enhancement", True, True, True)
        ]
        
        best_results = []
        
        for name, dynamic, diversification, swap in strategies:
            print(f"\nRunning {name}...")
            
            results = []
            for run in range(M):
                print(f"  Run {run+1}/{M}...")
                best_route, best_cost = tabu_search_enhanced(
                    N, Imax, cost_matrix, dynamic, diversification, swap
                )
                results.append((best_route, best_cost))
            
            # Calculate statistics
            costs = [cost for _, cost in results]
            mean_cost = statistics.mean(costs)
            std_dev = statistics.stdev(costs) if len(costs) > 1 else 0
            
            # Get best result
            best_result = min(results, key=lambda x: x[1])
            best_results.append((name, best_result, mean_cost, std_dev))
            
            print(f"  Best cost: {best_result[1]}")
            print(f"  Mean cost: {mean_cost:.2f}")
            print(f"  Std Dev: {std_dev:.2f}")
        
        # Compare all strategies
        print("\nComparison of Strategies:")
        for name, (route, cost), mean, std_dev in best_results:
            print(f"{name}: Best={cost}, Mean={mean:.2f}, StdDev={std_dev:.2f}")

if __name__ == "__main__":
    main()