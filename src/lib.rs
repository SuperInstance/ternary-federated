//! Federated learning for ternary agents.
//!
//! Multiple populations share strategy insights without sharing raw data.
//! Each node runs local ternary evolution, and a federated round aggregates
//! strategy summaries across nodes using configurable aggregation methods,
//! all while tracking a differential-privacy-style privacy budget.

use std::fmt;

// ---------------------------------------------------------------------------
// Ternary value
// ---------------------------------------------------------------------------

/// A ternary value: -1, 0, or +1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ternary {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl Ternary {
    /// Convert an i8 to a Ternary, clamping to the nearest valid value.
    pub fn from_i8(v: i8) -> Self {
        match v {
            ..=-1 => Ternary::Neg,
            0 => Ternary::Zero,
            1.. => Ternary::Pos,
        }
    }

    /// Convert to i8.
    pub fn as_i8(self) -> i8 {
        self as i8
    }

    /// Convert to f64.
    pub fn as_f64(self) -> f64 {
        self.as_i8() as f64
    }

    /// Pick a random ternary value (deterministic simple PRNG).
    pub fn random(state: &mut u64) -> Self {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (*state >> 33) % 3;
        match v {
            0 => Ternary::Neg,
            1 => Ternary::Zero,
            _ => Ternary::Pos,
        }
    }
}

impl fmt::Display for Ternary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_i8())
    }
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// A strategy is a vector of ternary values.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Strategy {
    pub values: Vec<Ternary>,
}

impl Strategy {
    /// Create a random strategy of given length.
    pub fn random(len: usize, state: &mut u64) -> Self {
        let values = (0..len).map(|_| Ternary::random(state)).collect();
        Self { values }
    }

    /// Create a strategy of all zeros.
    pub fn zeros(len: usize) -> Self {
        Self {
            values: vec![Ternary::Zero; len],
        }
    }

    /// Length of the strategy.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the strategy is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Compute fitness against a target (number of matching positions).
    pub fn fitness_against(&self, target: &Strategy) -> f64 {
        if self.len() != target.len() {
            return 0.0;
        }
        let matches = self
            .values
            .iter()
            .zip(&target.values)
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / self.len() as f64
    }

    /// Mutate one random position in the strategy.
    pub fn mutate(&mut self, state: &mut u64) {
        if self.is_empty() {
            return;
        }
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = (*state as usize) % self.len();
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (*state >> 33) % 3;
        self.values[idx] = match v {
            0 => Ternary::Neg,
            1 => Ternary::Zero,
            _ => Ternary::Pos,
        };
    }
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// A single ternary agent with a strategy and fitness.
#[derive(Clone, Debug)]
pub struct Agent {
    pub strategy: Strategy,
    pub fitness: f64,
}

impl Agent {
    /// Create a random agent with strategy of given length.
    pub fn random(strategy_len: usize, state: &mut u64) -> Self {
        Self {
            strategy: Strategy::random(strategy_len, state),
            fitness: 0.0,
        }
    }

    /// Evaluate fitness against a target strategy.
    pub fn evaluate(&mut self, target: &Strategy) {
        self.fitness = self.strategy.fitness_against(target);
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A local ternary population with its own evolution.
#[derive(Clone, Debug)]
pub struct Node {
    /// The agents in this node's population.
    pub agents: Vec<Agent>,
    /// This node's secret target strategy (never shared).
    pub target: Strategy,
    /// PRNG state for deterministic evolution.
    pub rng_state: u64,
    /// Unique node identifier.
    pub id: usize,
    /// Best fitness achieved so far.
    pub best_fitness: f64,
    /// Strategy that achieved best fitness.
    pub best_strategy: Strategy,
}

impl Node {
    /// Create a new node with `population_size` agents and strategies of `strategy_len`.
    pub fn new(population_size: usize, strategy_len: usize) -> Self {
        Self::with_id(population_size, strategy_len, 0)
    }

    /// Create a node with a specific ID.
    pub fn with_id(population_size: usize, strategy_len: usize, id: usize) -> Self {
        let mut state = id as u64 * 1_000_003 + 42;
        let target = Strategy::random(strategy_len, &mut state);
        let agents: Vec<Agent> = (0..population_size)
            .map(|_| Agent::random(strategy_len, &mut state))
            .collect();
        let best_strategy = Strategy::zeros(strategy_len);
        Self {
            agents,
            target,
            rng_state: state,
            id,
            best_fitness: 0.0,
            best_strategy,
        }
    }

    /// Set a specific target for the node.
    pub fn with_target(mut self, target: Strategy) -> Self {
        self.target = target;
        self
    }

    /// Run one generation of local evolution.
    pub fn evolve_step(&mut self) {
        // Evaluate all agents
        for agent in &mut self.agents {
            agent.evaluate(&self.target);
        }

        // Sort by fitness descending
        self.agents.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

        // Track best
        if let Some(best) = self.agents.first() {
            if best.fitness > self.best_fitness {
                self.best_fitness = best.fitness;
                self.best_strategy = best.strategy.clone();
            }
        }

        // Tournament selection + mutation: replace bottom half with mutated copies of top half
        let pop = self.agents.len();
        if pop < 2 {
            return;
        }
        let half = pop / 2;
        let top: Vec<Agent> = self.agents[..half].to_vec();
        for i in half..pop {
            let mut child = top[i - half].clone();
            child.strategy.mutate(&mut self.rng_state);
            self.agents[i] = child;
        }
    }

    /// Run N generations of local evolution.
    pub fn evolve(&mut self, generations: usize) {
        for _ in 0..generations {
            self.evolve_step();
        }
    }

    /// Get the average fitness across all agents.
    pub fn avg_fitness(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        self.agents.iter().map(|a| a.fitness).sum::<f64>() / self.agents.len() as f64
    }

    /// Get the best fitness.
    pub fn max_fitness(&self) -> f64 {
        self.agents
            .iter()
            .map(|a| a.fitness)
            .fold(0.0_f64, f64::max)
    }

    /// Get the strategy summary to share (the best strategy).
    /// In a real system this would be perturbed for privacy.
    pub fn strategy_summary(&self) -> Strategy {
        self.best_strategy.clone()
    }

    /// Apply a federated strategy update: blend the global strategy into the population.
    pub fn apply_federated_update(&mut self, global: &Strategy, blend_rate: f64) {
        for agent in &mut self.agents {
            for (i, v) in agent.strategy.values.iter_mut().enumerate() {
                if i < global.len() {
                    // With probability blend_rate, adopt the global value
                    self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let prob = (self.rng_state >> 33) as f64 / (u32::MAX as f64);
                    if prob < blend_rate {
                        *v = global.values[i];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// Method for aggregating strategies across nodes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AggregationMethod {
    /// Each position takes the value most common across nodes.
    MajorityVote,
    /// Nodes contribute proportionally to their fitness scores.
    WeightedAverage,
    /// Adopt the strategy from the highest-fitness node.
    BestOf,
}

/// Aggregates strategy summaries from multiple nodes.
pub struct Aggregator;

impl Aggregator {
    /// Aggregate strategies using the specified method.
    pub fn aggregate(
        summaries: &[(Strategy, f64)], // (strategy, fitness) per node
        method: AggregationMethod,
    ) -> Strategy {
        if summaries.is_empty() {
            return Strategy::zeros(0);
        }
        let len = summaries[0].0.len();
        match method {
            AggregationMethod::MajorityVote => Self::majority_vote(summaries, len),
            AggregationMethod::WeightedAverage => Self::weighted_average(summaries, len),
            AggregationMethod::BestOf => Self::best_of(summaries),
        }
    }

    fn majority_vote(summaries: &[(Strategy, f64)], len: usize) -> Strategy {
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let mut counts = [0usize; 3]; // neg, zero, pos
            for (s, _) in summaries {
                if i < s.len() {
                    match s.values[i] {
                        Ternary::Neg => counts[0] += 1,
                        Ternary::Zero => counts[1] += 1,
                        Ternary::Pos => counts[2] += 1,
                    }
                }
            }
            let best = counts.iter().enumerate().max_by_key(|(_, &c)| c).map(|(i, _)| i).unwrap_or(1);
            result.push(match best {
                0 => Ternary::Neg,
                2 => Ternary::Pos,
                _ => Ternary::Zero,
            });
        }
        Strategy { values: result }
    }

    fn weighted_average(summaries: &[(Strategy, f64)], len: usize) -> Strategy {
        let total_weight: f64 = summaries.iter().map(|(_, f)| f).sum();
        if total_weight <= 0.0 {
            return Strategy::zeros(len);
        }
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let mut weighted_sum = 0.0;
            for (s, f) in summaries {
                if i < s.len() {
                    weighted_sum += s.values[i].as_f64() * f;
                }
            }
            let avg = weighted_sum / total_weight;
            result.push(if avg > 0.33 {
                Ternary::Pos
            } else if avg < -0.33 {
                Ternary::Neg
            } else {
                Ternary::Zero
            });
        }
        Strategy { values: result }
    }

    fn best_of(summaries: &[(Strategy, f64)]) -> Strategy {
        summaries
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(s, _)| s.clone())
            .unwrap_or_else(|| Strategy::zeros(0))
    }
}

// ---------------------------------------------------------------------------
// Privacy Budget
// ---------------------------------------------------------------------------

/// Tracks how much information has been shared (differential privacy style).
#[derive(Clone, Debug)]
pub struct PrivacyBudget {
    /// Total epsilon available.
    pub total_epsilon: f64,
    /// Epsilon spent so far.
    pub spent: f64,
}

impl PrivacyBudget {
    /// Create a new privacy budget with the given total epsilon.
    pub fn new(total_epsilon: f64) -> Self {
        Self {
            total_epsilon,
            spent: 0.0,
        }
    }

    /// Whether there is remaining budget.
    pub fn has_budget(&self) -> bool {
        self.spent < self.total_epsilon
    }

    /// Remaining epsilon.
    pub fn remaining(&self) -> f64 {
        (self.total_epsilon - self.spent).max(0.0)
    }

    /// Spend some epsilon. Returns false if insufficient budget.
    pub fn spend(&mut self, epsilon: f64) -> bool {
        if self.spent + epsilon > self.total_epsilon {
            return false;
        }
        self.spent += epsilon;
        true
    }

    /// Fraction of budget spent.
    pub fn fraction_spent(&self) -> f64 {
        if self.total_epsilon <= 0.0 {
            1.0
        } else {
            self.spent / self.total_epsilon
        }
    }

    /// Reset the budget.
    pub fn reset(&mut self) {
        self.spent = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Federated Round
// ---------------------------------------------------------------------------

/// One round of federated aggregation.
#[derive(Clone, Debug)]
pub struct FederatedRound {
    /// Round number (0-indexed).
    pub round_number: usize,
    /// Strategy summaries from each node (with fitness).
    pub summaries: Vec<(Strategy, f64)>,
    /// Aggregated global strategy.
    pub global_strategy: Strategy,
    /// Privacy epsilon spent this round.
    pub epsilon_spent: f64,
    /// Per-node fitness before aggregation.
    pub pre_aggregation_fitness: Vec<f64>,
    /// Per-node fitness after aggregation.
    pub post_aggregation_fitness: Vec<f64>,
}

impl FederatedRound {
    /// Execute one federated round.
    pub fn execute(
        nodes: &mut [Node],
        round_number: usize,
        method: AggregationMethod,
        epsilon: f64,
        privacy: &mut PrivacyBudget,
    ) -> Option<Self> {
        // Check privacy budget
        if !privacy.has_budget() || privacy.remaining() < epsilon {
            return None;
        }

        // Collect summaries
        let pre_fitness: Vec<f64> = nodes.iter().map(|n| n.max_fitness()).collect();
        let summaries: Vec<(Strategy, f64)> = nodes
            .iter()
            .map(|n| (n.strategy_summary(), n.best_fitness))
            .collect();

        // Aggregate
        let global = Aggregator::aggregate(&summaries, method);

        // Spend privacy budget
        privacy.spend(epsilon);

        // Apply update to all nodes (blend rate 0.3)
        for node in nodes.iter_mut() {
            node.apply_federated_update(&global, 0.3);
        }

        // Re-evaluate to get post-aggregation fitness
        let post_fitness: Vec<f64> = {
            nodes.iter_mut().for_each(|n| n.evolve(1));
            nodes.iter().map(|n| n.max_fitness()).collect()
        };

        Some(Self {
            round_number,
            summaries,
            global_strategy: global,
            epsilon_spent: epsilon,
            pre_aggregation_fitness: pre_fitness,
            post_aggregation_fitness: post_fitness,
        })
    }
}

// ---------------------------------------------------------------------------
// Federated Experiment Config
// ---------------------------------------------------------------------------

/// Configuration for a federated experiment.
#[derive(Clone, Debug)]
pub struct FederatedConfig {
    /// Number of federated rounds.
    pub rounds: usize,
    /// Number of local evolution generations per round.
    pub local_generations: usize,
    /// Aggregation method to use.
    pub aggregator: AggregationMethod,
    /// Epsilon spent per round.
    pub epsilon_per_round: f64,
    /// Total privacy budget.
    pub total_epsilon: f64,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            rounds: 20,
            local_generations: 10,
            aggregator: AggregationMethod::WeightedAverage,
            epsilon_per_round: 0.1,
            total_epsilon: 5.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Federation Result
// ---------------------------------------------------------------------------

/// Structured result from a federated experiment.
#[derive(Clone, Debug)]
pub struct FederationResult {
    /// Fitness history per node, per round.
    pub per_node_fitness: Vec<Vec<f64>>,
    /// Global strategy fitness at each round (average across nodes).
    pub global_fitness_history: Vec<f64>,
    /// Total privacy epsilon spent.
    pub privacy_spent: f64,
    /// Number of rounds completed.
    pub rounds_completed: usize,
    /// Number of nodes.
    pub num_nodes: usize,
    /// Per-round details.
    pub rounds: Vec<FederatedRound>,
    /// Final best strategy.
    pub final_global_strategy: Strategy,
    /// Whether the experiment completed all rounds.
    pub completed: bool,
}

impl FederationResult {
    /// Final global fitness (average of per-node fitness at last round).
    pub fn global_fitness(&self) -> f64 {
        self.global_fitness_history.last().copied().unwrap_or(0.0)
    }

    /// Total privacy spent.
    pub fn privacy_spent(&self) -> f64 {
        self.privacy_spent
    }

    /// Whether the experiment converged (fitness > 0.9 in last round).
    pub fn converged(&self) -> bool {
        self.global_fitness() > 0.9
    }

    /// Best-performing node index.
    pub fn best_node(&self) -> usize {
        self.per_node_fitness
            .iter()
            .enumerate()
            .map(|(i, history)| (i, history.last().copied().unwrap_or(0.0)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        format!(
            "FederationResult: {} rounds, {} nodes, global_fitness={:.3}, privacy_spent={:.2}ε, converged={}",
            self.rounds_completed,
            self.num_nodes,
            self.global_fitness(),
            self.privacy_spent,
            self.converged(),
        )
    }
}

// ---------------------------------------------------------------------------
// Federated Experiment
// ---------------------------------------------------------------------------

/// Orchestrates a federated learning experiment.
pub struct FederatedExperiment;

impl FederatedExperiment {
    /// Run a federated experiment with the given nodes and config.
    pub fn run(mut nodes: Vec<Node>, config: FederatedConfig) -> FederationResult {
        let num_nodes = nodes.len();
        let mut privacy = PrivacyBudget::new(config.total_epsilon);
        let mut per_node_fitness: Vec<Vec<f64>> = vec![vec![]; num_nodes];
        let mut global_fitness_history = Vec::new();
        let mut rounds_completed = 0;
        let mut round_records = Vec::new();
        let mut final_strategy = Strategy::zeros(0);

        for round in 0..config.rounds {
            // Local evolution
            for node in nodes.iter_mut() {
                node.evolve(config.local_generations);
            }

            // Federated round
            let fed_round = FederatedRound::execute(
                &mut nodes,
                round,
                config.aggregator,
                config.epsilon_per_round,
                &mut privacy,
            );

            match fed_round {
                Some(fr) => {
                    final_strategy = fr.global_strategy.clone();
                    // Record fitness
                    for (i, node) in nodes.iter().enumerate() {
                        per_node_fitness[i].push(node.max_fitness());
                    }
                    let avg = nodes.iter().map(|n| n.max_fitness()).sum::<f64>() / num_nodes as f64;
                    global_fitness_history.push(avg);
                    rounds_completed += 1;
                    round_records.push(fr);
                }
                None => {
                    // Privacy budget exhausted — record final fitness without aggregation
                    for (i, node) in nodes.iter().enumerate() {
                        per_node_fitness[i].push(node.max_fitness());
                    }
                    let avg = nodes.iter().map(|n| n.max_fitness()).sum::<f64>() / num_nodes as f64;
                    global_fitness_history.push(avg);
                    rounds_completed += 1;
                    break;
                }
            }
        }

        FederationResult {
            per_node_fitness,
            global_fitness_history,
            privacy_spent: privacy.spent,
            rounds_completed,
            num_nodes,
            rounds: round_records,
            final_global_strategy: final_strategy,
            completed: rounds_completed == config.rounds,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_from_i8() {
        assert_eq!(Ternary::from_i8(-5), Ternary::Neg);
        assert_eq!(Ternary::from_i8(-1), Ternary::Neg);
        assert_eq!(Ternary::from_i8(0), Ternary::Zero);
        assert_eq!(Ternary::from_i8(1), Ternary::Pos);
        assert_eq!(Ternary::from_i8(42), Ternary::Pos);
    }

    #[test]
    fn test_ternary_conversions() {
        assert_eq!(Ternary::Neg.as_i8(), -1);
        assert_eq!(Ternary::Zero.as_i8(), 0);
        assert_eq!(Ternary::Pos.as_i8(), 1);
        assert_eq!(Ternary::Neg.as_f64(), -1.0);
        assert_eq!(Ternary::Zero.as_f64(), 0.0);
        assert_eq!(Ternary::Pos.as_f64(), 1.0);
    }

    #[test]
    fn test_ternary_display() {
        assert_eq!(format!("{}", Ternary::Neg), "-1");
        assert_eq!(format!("{}", Ternary::Zero), "0");
        assert_eq!(format!("{}", Ternary::Pos), "1");
    }

    #[test]
    fn test_ternary_random() {
        let mut state = 12345u64;
        for _ in 0..100 {
            let v = Ternary::random(&mut state);
            assert!(v == Ternary::Neg || v == Ternary::Zero || v == Ternary::Pos);
        }
    }

    #[test]
    fn test_strategy_random_length() {
        let mut state = 42u64;
        let s = Strategy::random(10, &mut state);
        assert_eq!(s.len(), 10);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_strategy_zeros() {
        let s = Strategy::zeros(5);
        assert_eq!(s.len(), 5);
        assert!(s.values.iter().all(|v| *v == Ternary::Zero));
    }

    #[test]
    fn test_strategy_fitness_perfect() {
        let target = Strategy {
            values: vec![Ternary::Pos, Ternary::Neg, Ternary::Zero],
        };
        let s = target.clone();
        assert!((s.fitness_against(&target) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_strategy_fitness_zero() {
        let target = Strategy {
            values: vec![Ternary::Pos, Ternary::Pos, Ternary::Pos],
        };
        let s = Strategy {
            values: vec![Ternary::Neg, Ternary::Neg, Ternary::Neg],
        };
        assert!((s.fitness_against(&target) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_strategy_fitness_partial() {
        let target = Strategy {
            values: vec![Ternary::Pos, Ternary::Neg, Ternary::Zero, Ternary::Pos],
        };
        let s = Strategy {
            values: vec![Ternary::Pos, Ternary::Zero, Ternary::Zero, Ternary::Neg],
        };
        assert!((s.fitness_against(&target) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_strategy_mutate_changes_something() {
        let mut state = 99u64;
        let original = Strategy {
            values: vec![Ternary::Zero; 20],
        };
        let mut mutated = original.clone();
        // Mutate several times to ensure change
        for _ in 0..5 {
            mutated.mutate(&mut state);
        }
        assert_ne!(original, mutated);
    }

    #[test]
    fn test_node_creation() {
        let node = Node::new(50, 10);
        assert_eq!(node.agents.len(), 50);
        assert_eq!(node.target.len(), 10);
        assert_eq!(node.id, 0);
    }

    #[test]
    fn test_node_evolution_improves_fitness() {
        let mut node = Node::new(100, 20);
        // Run many generations
        node.evolve(100);
        // Should have improved from initial 0
        assert!(node.best_fitness > 0.0);
    }

    #[test]
    fn test_node_avg_fitness() {
        let mut node = Node::new(50, 10);
        node.evolve(5);
        let avg = node.avg_fitness();
        assert!(avg >= 0.0 && avg <= 1.0);
    }

    #[test]
    fn test_aggregator_majority_vote() {
        let s1 = Strategy {
            values: vec![Ternary::Pos, Ternary::Neg],
        };
        let s2 = Strategy {
            values: vec![Ternary::Pos, Ternary::Zero],
        };
        let s3 = Strategy {
            values: vec![Ternary::Neg, Ternary::Neg],
        };
        let result = Aggregator::aggregate(
            &[(s1, 0.5), (s2, 0.5), (s3, 0.5)],
            AggregationMethod::MajorityVote,
        );
        assert_eq!(result.values[0], Ternary::Pos); // 2 pos, 1 neg
        assert_eq!(result.values[1], Ternary::Neg); // 2 neg-ish (neg + zero tie → neg in majority)
    }

    #[test]
    fn test_aggregator_weighted_average() {
        let s1 = Strategy {
            values: vec![Ternary::Pos],
        };
        let s2 = Strategy {
            values: vec![Ternary::Neg],
        };
        // s1 has much higher weight
        let result = Aggregator::aggregate(
            &[(s1, 10.0), (s2, 1.0)],
            AggregationMethod::WeightedAverage,
        );
        assert_eq!(result.values[0], Ternary::Pos);
    }

    #[test]
    fn test_aggregator_best_of() {
        let s1 = Strategy {
            values: vec![Ternary::Pos],
        };
        let s2 = Strategy {
            values: vec![Ternary::Neg],
        };
        let result = Aggregator::aggregate(
            &[(s1.clone(), 0.3), (s2.clone(), 0.9)],
            AggregationMethod::BestOf,
        );
        assert_eq!(result.values[0], Ternary::Neg); // s2 had higher fitness
    }

    #[test]
    fn test_aggregator_empty() {
        let result = Aggregator::aggregate(&[], AggregationMethod::MajorityVote);
        assert!(result.is_empty());
    }

    #[test]
    fn test_privacy_budget_basic() {
        let mut pb = PrivacyBudget::new(1.0);
        assert!(pb.has_budget());
        assert!((pb.remaining() - 1.0).abs() < 1e-9);
        assert!(pb.spend(0.5));
        assert!((pb.remaining() - 0.5).abs() < 1e-9);
        assert!(pb.has_budget());
        assert!(pb.spend(0.5));
        assert!(!pb.has_budget());
        assert!(!pb.spend(0.1)); // No budget left
    }

    #[test]
    fn test_privacy_budget_fraction() {
        let mut pb = PrivacyBudget::new(2.0);
        pb.spend(0.5);
        assert!((pb.fraction_spent() - 0.25).abs() < 1e-9);
        pb.reset();
        assert!((pb.fraction_spent() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_federated_round_execution() {
        let mut nodes = vec![
            Node::with_id(30, 8, 0),
            Node::with_id(30, 8, 1),
            Node::with_id(30, 8, 2),
        ];
        let mut privacy = PrivacyBudget::new(10.0);
        // Pre-evolve so agents have some fitness
        for node in nodes.iter_mut() {
            node.evolve(10);
        }
        let round = FederatedRound::execute(&mut nodes, 0, AggregationMethod::MajorityVote, 0.5, &mut privacy);
        assert!(round.is_some());
        let r = round.unwrap();
        assert_eq!(r.round_number, 0);
        assert!((r.epsilon_spent - 0.5).abs() < 1e-9);
        assert_eq!(r.summaries.len(), 3);
    }

    #[test]
    fn test_federated_round_privacy_exhausted() {
        let mut nodes = vec![Node::with_id(20, 5, 0)];
        let mut privacy = PrivacyBudget::new(0.1);
        let round = FederatedRound::execute(&mut nodes, 0, AggregationMethod::BestOf, 0.5, &mut privacy);
        // 0.1 remaining < 0.5 needed
        assert!(round.is_none());
    }

    #[test]
    fn test_full_experiment() {
        let nodes: Vec<Node> = (0..3).map(|i| Node::with_id(40, 8, i)).collect();
        let config = FederatedConfig {
            rounds: 10,
            local_generations: 5,
            aggregator: AggregationMethod::WeightedAverage,
            epsilon_per_round: 0.2,
            total_epsilon: 10.0,
        };
        let result = FederatedExperiment::run(nodes, config);
        assert_eq!(result.num_nodes, 3);
        assert!(result.rounds_completed > 0);
        assert!(!result.global_fitness_history.is_empty());
        assert!(result.privacy_spent > 0.0);
        assert_eq!(result.rounds.len(), result.rounds_completed);
        println!("{}", result.summary());
    }

    #[test]
    fn test_experiment_privacy_limited() {
        // Very tight budget — should stop early
        let nodes: Vec<Node> = (0..2).map(|i| Node::with_id(20, 5, i)).collect();
        let config = FederatedConfig {
            rounds: 100,
            local_generations: 3,
            aggregator: AggregationMethod::MajorityVote,
            epsilon_per_round: 1.0,
            total_epsilon: 3.0,
        };
        let result = FederatedExperiment::run(nodes, config);
        assert!(result.rounds_completed <= 4); // 3 rounds max with budget 3.0
        assert!(!result.completed);
    }

    #[test]
    fn test_convergence_with_shared_target() {
        // All nodes share the same target — should converge well
        let target = Strategy {
            values: vec![Ternary::Pos; 10],
        };
        let nodes: Vec<Node> = (0..5)
            .map(|i| Node::with_id(50, 10, i).with_target(target.clone()))
            .collect();
        let config = FederatedConfig {
            rounds: 30,
            local_generations: 10,
            aggregator: AggregationMethod::WeightedAverage,
            epsilon_per_round: 0.1,
            total_epsilon: 10.0,
        };
        let result = FederatedExperiment::run(nodes, config);
        // With shared target and enough rounds, should converge
        assert!(result.global_fitness() > 0.5, "Expected fitness > 0.5, got {}", result.global_fitness());
    }

    #[test]
    fn test_federation_result_best_node() {
        let nodes: Vec<Node> = (0..4).map(|i| Node::with_id(30, 8, i)).collect();
        let config = FederatedConfig::default();
        let result = FederatedExperiment::run(nodes, config);
        let best = result.best_node();
        assert!(best < result.num_nodes);
    }

    #[test]
    fn test_node_with_target() {
        let target = Strategy {
            values: vec![Ternary::Neg, Ternary::Pos, Ternary::Zero],
        };
        let node = Node::new(10, 3).with_target(target.clone());
        assert_eq!(node.target, target);
    }
}
