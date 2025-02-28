# Metaheuristic Algorithms for Optimization Problems

This repository contains implementations of various metaheuristic algorithms to solve classic optimization problems. The project aims to demonstrate and compare different approaches to solving complex computational problems that are typically NP-hard.

## 🎯 Problems

### Currently Implemented
- **Traveling Salesman Problem (TSP)**
  - Implementations using:
    - Tabu Search
    - Genetic Algorithm
  - Features include:
    - Dynamic tabu tenure
    - Order crossover (OX)
    - Swap mutation
    - Multi-run statistical analysis

- **Knapsack Problem**
  - Implementation using Simulated Annealing
  - Features include:
    - Basic and improved variants
    - Adaptive cooling schedules
    - Enhanced neighborhood structures
    - Greedy initialization

- **Beale Function Optimization**
  - Implementations using:
    - Binary Genetic Algorithm
    - Real-coded Genetic Algorithm
  - Features include:
    - Two-point crossover
    - Intermediate recombination
    - Adaptive mutation rates
    - Statistical comparison tools

- **Ackley Function Optimization**
  - Implementations using:
    - Binary/Real Genetic Algorithms
    - Evolutionary Strategies
  - Features include:
    - Self-adaptive mutation
    - (μ, λ) and (μ + λ) selection
    - Multi-dimensional optimization (2D-20D)
    - CMA-ES implementation

## 🧮 Algorithms

### Currently Implemented
- **Tabu Search**
  - Memory-based metaheuristic
  - Short-term and long-term memory structures
  - Aspiration criteria
  - Diversification strategies

- **Simulated Annealing**
  - Temperature-based acceptance probability
  - Adaptive cooling schedules
  - Dynamic neighborhood structures
  - Reheating mechanisms

- **Genetic Algorithms**
  - Binary and real-coded representations
  - Stochastic Universal Sampling
  - Multiple crossover operators
  - Elitism and adaptive parameters
  - Specialized TSP operators

- **Evolutionary Strategies**
  - Self-adaptive mutation parameters
  - Derandomized adaptation
  - Covariance Matrix Adaptation (CMA-ES)
  - Multi-parent recombination

### Planned Implementations
- **Particle Swarm Optimization**
- **Differential Evolution**

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Algorithms

#### Tabu Search for TSP
```bash
python "tabu search\tabu-search_tsp.py" --mode [basic|extension|challenge] --file [instance_file] --runs [number_of_runs]
```

#### Simulated Annealing for Knapsack
```bash
python "simulated annealing\simulated-annealing_knapsack.py"
```

#### Genetic Algorithms
```bash
python "genetic_algorithms\ga_solution.py" --function [beale|ackley|tsp|all] --tsp-file [instance_file] --runs [number_of_runs]
```

#### Evolutionary Strategies
```bash
python "evolutionary_strategies\es_solution.py" --function [ackley] --dim [dimensions] --runs [number_of_runs]
```

## 🔧 Project Structure

```
ecnga\
├── tabu search\
│   └── tabu-search_tsp.py
├── simulated annealing\
│   ├── simulated-annealing_knapsack.py
│   └── input.txt
├── genetic_algorithms\
│   └── ga_solution.py
├── evolutionary_strategies\
│   └── es_solution.py
└── utils\
    ├── visualization.py
    └── statistics.py
```

## 📚 References

- Glover, F. (1989). Tabu Search—Part I. ORSA Journal on Computing
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization & Machine Learning
- Hansen, N. (2006). The CMA Evolution Strategy: A Comparing Review
- Beyer, H.-G., & Schwefel, H.-P. (2002). Evolution Strategies: A Comprehensive Introduction

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Happy Optimizing! 🎉