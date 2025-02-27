# Metaheuristic Algorithms for Optimization Problems

This repository contains implementations of various metaheuristic algorithms to solve classic optimization problems. The project aims to demonstrate and compare different approaches to solving complex computational problems that are typically NP-hard.

## 🎯 Problems

### Currently Implemented
- **Traveling Salesman Problem (TSP)**
  - Implementation using Tabu Search
  - Features include:
    - Dynamic tabu tenure
    - Frequency-based diversification
    - Hybrid move operations (insert and swap)
    - Multi-run statistical analysis

- **Knapsack Problem**
  - Implementation using Simulated Annealing
  - Features include:
    - Basic and improved variants
    - Adaptive cooling schedules
    - Enhanced neighborhood structures
    - Greedy initialization
    - Statistical analysis and comparison tools

- **Ackley's Function Optimization**
  - Implementation using Evolutionary Strategies
  - Features include:
    - (μ, λ) and (μ + λ) selection schemes
    - Self-adaptive mutation parameters
    - Covariance Matrix Adaptation (CMA-ES)
    - Multi-parent recombination
    - Statistical analysis tools

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
  - Multi-run statistical analysis

- **Evolutionary Strategies**
  - Population-based optimization
  - Self-adaptation mechanisms
  - CMA-ES implementation
  - Derandomized adaptation
  - Multi-parent recombination

### Planned Implementations
- **Genetic Algorithms**
- **Particle Swarm Optimization**

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
python "tabu search/tabu-search_tsp.py" --mode [basic|extension|challenge] --file [instance_file] --runs [number_of_runs]
```

#### Simulated Annealing for Knapsack
```bash
python "simulated annealing/simulated-annealing_knapsack.py"
```

#### Evolutionary Strategies for Ackley
```bash
python "evolutionary/evolutionary-strategies_ackley.py" --mode [basic|cmaes] --dim [dimensions] --runs [number_of_runs]
```

Interactive modes available for each algorithm with statistical analysis and parameter tuning options.

## 📊 Performance Analysis

Each implementation includes performance analysis tools to compare:
- Solution quality
- Convergence speed
- Statistical robustness
- Parameter sensitivity

## 🔧 Project Structure

```
ecnga/
├── tabu search/
│   └── tabu-search_tsp.py
├── simulated annealing/
│   ├── simulated-annealing_knapsack.py
│   └── input.txt
├── evolutionary/
│   ├── evolutionary-strategies_ackley.py
│   └── cmaes.py
├── instances/
│   ├── tsp/
│   ├── knapsack/
│   └── ackley/
└── utils/
    ├── visualization.py
    └── statistics.py
```

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📚 References

- Glover, F. (1989). Tabu Search—Part I. ORSA Journal on Computing
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing
- Hansen, N. (2006). The CMA Evolution Strategy: A Comparing Review
- Beyer, H.-G., & Schwefel, H.-P. (2002). Evolution Strategies: A Comprehensive Introduction

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Happy Optimizing! 🎉