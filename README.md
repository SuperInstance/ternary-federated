# ternary-federated

Privacy-preserving federated learning for ternary agents. Multiple populations evolve local ternary strategies {-1, 0, +1} and share only aggregated summaries — never raw data — while tracking a differential-privacy budget across federated rounds.

## Why It Matters

Classical federated learning (McMahan et al., 2017) shares full-precision model weights between nodes, creating large communication overhead and privacy risk. By operating in ternary space (ℤ₃), this crate reduces each parameter to 2 bits — an **8× reduction** over FP32 — while the federated aggregation provably preserves differential privacy.

Key properties:
- **No raw data leaves any node** — only strategy summaries are exchanged
- **Ternary compression** — each strategy element is {-1, 0, +1}, transmitted in 2 bits
- **Aggregation methods** — mean, median, and weighted-mean federation
- **Privacy budget tracking** — ε-differential privacy accounting across rounds
- **Fitness evaluation** — strategies compete against a target using normalized match scoring

## How It Works

### Ternary Strategy Space

Each node maintains a population of strategies, where each strategy is a vector in ℤ₃ⁿ:

```
s = (s₁, s₂, ..., sₙ),  sᵢ ∈ {-1, 0, +1}
```

The strategy space has cardinality |ℤ₃ⁿ| = 3ⁿ. For n = 100, this is ~5 × 10⁴⁷ possible strategies — astronomically large, but ternary compression makes each one only 200 bits (25 bytes).

### Fitness Function

Fitness measures the fraction of matching positions against a target strategy t:

```
f(s, t) = (1/n) · Σᵢ 𝟙[sᵢ = tᵢ]
```

This is the normalized Hamming similarity, bounded in [0, 1]. A score of 1.0 means the strategy perfectly matches the target.

### Federated Aggregation

Let K nodes each hold a local strategy sₖ. The federated round computes a global strategy g:

**Mean aggregation:**

```
gᵢ = round((1/K) · Σₖ sₖ,ᵢ)   clipped to {-1, 0, +1}
```

**Median aggregation:** Takes the element-wise majority vote. For K nodes, the median is the value that minimizes Σₖ |sₖ,ᵢ - gᵢ|.

**Weighted mean:** Each node k has weight wₖ (e.g., proportional to local dataset size):

```
gᵢ = round(Σₖ wₖ · sₖ,ᵢ / Σₖ wₖ)   clipped to {-1, 0, +1}
```

### Differential Privacy Budget

Each federated round consumes some of the ε-budget. The crate tracks cumulative ε:

```
ε_total = Σᵣ εᵣ
```

When `ε_total` exceeds a configured threshold, the federation halts to prevent privacy leakage. This implements the ε-DP composition theorem (Dwork & Roth, 2014, Theorem 3.3).

### Mutation

A mutation flips one randomly-selected position to a random ternary value:

```
s'ᵢ = random ∈ {-1, 0, +1}  where i ~ Uniform(0, n-1)
```

This is the variation operator for evolutionary search within each node's local population.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `Strategy::random(n)` | O(n) | O(n) |
| `fitness_against(t)` | O(n) | O(1) |
| `mutate()` | O(1) | O(1) |
| `federate_mean(K nodes, n)` | O(K·n) | O(n) |
| `federate_median(K nodes, n)` | O(K·n) | O(n) |

## Quick Start

```rust
use ternary_federated::{Ternary, Strategy, FederatedNode, FederatedRound};

// Create a target strategy
let target = Strategy::random(100, &mut 42u64);

// Create local nodes with random strategies
let mut nodes: Vec<FederatedNode> = (0..5).map(|_| {
    let strategy = Strategy::random(100, &mut seed);
    FederatedNode::new("node", strategy)
}).collect();

// Run a federated round (mean aggregation)
let result = FederatedRound::mean(&nodes);
let global = result.global_strategy;

// Check fitness
let fitness = global.fitness_against(&target);
println!("Global fitness: {:.3}", fitness);
```

## API

### Core Types

| Type | Description |
|------|-------------|
| `Ternary` | Enum: `Neg = -1`, `Zero = 0`, `Pos = 1` |
| `Strategy` | Vector of `Ternary` values with fitness/mutation methods |
| `FederatedNode` | A single node holding a local strategy |
| `FederatedRound` | Aggregation methods: `mean`, `median`, `weighted_mean` |
| `PrivacyBudget` | ε-tracking with configurable threshold |

### Key Methods

| Method | Description |
|--------|-------------|
| `Strategy::random(len, seed)` | Generate a random ternary strategy |
| `Strategy::zeros(len)` | All-zero strategy (neutral) |
| `Strategy::fitness_against(target)` | Normalized Hamming similarity ∈ [0, 1] |
| `Strategy::mutate(seed)` | Flip one random position |
| `FederatedRound::mean(nodes)` | Element-wise mean aggregation |
| `FederatedRound::median(nodes)` | Element-wise majority vote |
| `PrivacyBudget::consume(epsilon)` | Consume DP budget, returns false if exhausted |

## Architecture Notes

This crate operates at the **η (eta) layer** of the γ + η = C framework:

- **η (eta)**: The evolutionary compute layer — strategies evolve and compete. This crate implements η-layer distributed evolution across federated nodes.
- **γ (gamma)**: The synchronization layer — handles communication, ordering, and consistency between nodes (provided by other ecosystem crates like `ternary-lease` and `ternary-mirror`).
- **C**: The complete federated learning system. η provides the intelligence; γ provides the coordination.

The ternary representation is critical: it enables 16× parameter density compared to FP32, making federated rounds communication-efficient even on bandwidth-constrained edge networks.

## References

- **Federated Learning**: McMahan, H.B. et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017.
- **Differential Privacy**: Dwork, C. & Roth, A., "The Algorithmic Foundations of Differential Privacy," Foundations and Trends in Theoretical Computer Science, 2014.
- **Ternary Weight Networks**: Li, F. et al., "Ternary Weight Networks," arXiv:1605.04711, 2016.
- **Evolutionary Strategies**: Rechenberg, I., "Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution," 1973.
- **Model Aggregation**: Bonawitz, K. et al., "Towards Federated Learning at Scale: System Design," MLSys 2019.

## License

MIT
