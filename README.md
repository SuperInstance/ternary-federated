# ternary-federated

Federated learning for ternary agents — multiple populations that share strategy insights without sharing raw data.

## Overview

This crate implements a federated learning framework where each **node** runs a local ternary population (agents with strategies drawn from `{+1, 0, -1}`). Nodes evolve independently, then periodically share aggregated strategy information in **federated rounds**. A central **aggregator** merges insights from all nodes using configurable strategies, while a **privacy budget** tracks cumulative information leakage.

## Key Concepts

### Ternary Strategies
Each agent's strategy is a vector of ternary values `(+1, 0, -1)`. This compact representation captures three-way decisions: commit, abstain, or oppose.

### Federated Rounds
1. Each node runs local evolution for some number of generations
2. Nodes submit strategy summaries (not raw data) to the aggregator
3. Aggregator merges summaries using one of several methods
4. Merged strategy is broadcast back to all nodes
5. Privacy budget is debited for each round

### Aggregation Methods
- **Majority Vote**: Each strategy position takes the value most common across nodes
- **Weighted Average**: Nodes contribute proportionally to their fitness scores
- **Best-Of**: Adopt the strategy from the highest-fitness node

### Privacy Budget
Inspired by differential privacy. Each federated round consumes some of the total privacy budget (ε). Once exhausted, nodes stop sharing to prevent reconstruction of individual data.

### Federated Experiment
Orchestrates the full workflow: N rounds across M nodes, tracking per-node fitness, global convergence, and cumulative privacy spend.

## Example

```rust
use ternary_federated::*;

// Create 4 nodes with populations of 50 agents, strategy length 10
let nodes: Vec<Node> = (0..4)
    .map(|_| Node::new(50, 10))
    .collect();

// Configure the experiment
let config = FederatedConfig {
    rounds: 20,
    local_generations: 10,
    aggregator: AggregationMethod::WeightedAverage,
    epsilon_per_round: 0.1,
    total_epsilon: 5.0,
};

// Run it
let result = FederatedExperiment::run(nodes, config);

println!("Global fitness: {:.3}", result.global_fitness());
println!("Privacy spent: {:.2}ε", result.privacy_spent());
```

## Federated Learning Methodology

### Why Federated?
Traditional centralized learning requires pooling all data in one place. Federated learning keeps data local — each node trains on its own data and only shares model updates. This is critical when:

- Data is sensitive (medical, financial, personal)
- Data is large (transmitting raw data is expensive)
- Regulations prevent data sharing (GDPR, HIPAA)
- Latency matters (local inference is faster)

### Ternary-Specific Design
Ternary strategies are naturally compact — a strategy of length L is just L trits (ternary digits). This makes aggregation efficient:

- **Bandwidth**: Sharing a strategy summary costs O(L) per node
- **Aggregation**: Majority vote on trits is O(N·L) total
- **Privacy**: Ternary noise is simpler to reason about than continuous noise

### Convergence
Under standard federated learning assumptions (bounded gradients, sufficient participation), the global strategy converges to a neighborhood of the optimal solution. The size of the neighborhood depends on:
- The privacy budget (more privacy → larger neighborhood)
- The number of nodes (more nodes → faster convergence)
- The aggregation method (weighted average converges fastest in practice)

## License

MIT

## See Also
- **ternary-distributed** — related
- **ternary-consensus** — related
- **ternary-mesh** — related
- **ternary-training** — related
- **ternary-transfer** — related

