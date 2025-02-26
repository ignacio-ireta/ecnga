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

### Coming Soon
- **Ackley's Function Optimization**

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

### Planned Implementations
- **Evolutionary Algorithms**
  - Genetic Algorithms
  - Evolution Strategies

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

Interactive modes available:
1. Single execution (with manual or file input)
2. Multiple executions with statistical analysis
3. Algorithm comparison (basic vs. improved variants)

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
├── evolutionary/           (coming soon)
├── instances/             (coming soon)
│   ├── tsp/
│   ├── knapsack/
│   └── ackley/
└── utils/                 (coming soon)
    ├── visualization.py
    └── statistics.py
```

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📚 References

- Glover, F. (1989). Tabu Search—Part I. ORSA Journal on Computing
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing
- Holland, J. H. (1992). Adaptation in Natural and Artificial Systems

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Happy Optimizing! 🎉