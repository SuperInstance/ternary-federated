# Future Integration: ternary-federated

## Current State
Provides federated learning for ternary agents: multiple populations run local ternary evolution, and federated rounds aggregate strategy summaries across nodes using configurable aggregation methods (mean, median, trimmed mean), with differential-privacy-style privacy budget tracking. Each node has `FederatedNode` with local ternary population and `FederatedRound` for aggregation.

## Integration Opportunities

### With ternary-cell (Distributed Grid Training)
A cell grid distributed across multiple rooms (Codespaces) trains via federated learning. Each room runs a local ternary-cell grid with its own data (sensor readings, user interactions). `FederatedRound` aggregates cell state distributions across rooms without sharing raw data. The aggregated strategy becomes the global cell behavior, while each room retains local adaptations. Privacy budgets track how much information leaks per round.

### With ternary-transfer (Federated Knowledge Sharing)
ternary-transfer moves knowledge between rooms. ternary-federated aggregates knowledge from multiple rooms simultaneously. Together: instead of one-to-one transfer, use federated aggregation to combine knowledge from N source rooms before transferring to the target. `TransferStrategy::WeightedBlend` uses federated weights — rooms that are more similar to the target get higher weight in the aggregation.

### With ternary-consensus (Federated Consensus)
ternary-consensus requires agreement among nodes. ternary-federated aggregates strategies across nodes. The aggregation IS a form of consensus — federated mean/median/trimmed-mean are consensus protocols. Trimmed mean is particularly interesting: it removes extreme values (potential Byzantine outliers) before averaging, providing Byzantine fault tolerance without the full 3f+1 overhead.

## Potential in Mature Systems
In room-as-codespace, rooms are distributed across Codespaces, Jetsons, ESP32s, and browsers. Each room has local data it cannot share (privacy, bandwidth, latency). ternary-federated enables rooms to learn from each other without sharing raw data: each room computes a strategy summary (ternary distribution, fitness statistics), sends the summary to PLATO, and PLATO aggregates. Privacy budgets ensure the summaries don't leak too much information over time. ESP32 nodes participate via lightweight summaries (just their ternary distribution histogram — 3 values).

## Cross-Pollination Ideas
- **ternary-adversarial**: Federated adversarial training — each room trains against local adversaries, then federated aggregation shares adversarial robustness across rooms.
- **ternary-pareto**: Federated multi-objective optimization — each room optimizes local objectives, federated rounds find globally Pareto-optimal solutions.
- **ternary-curriculum**: Federated curriculum — rooms share lesson plans (curriculum schedules) via federated aggregation, learning from each other's teaching experiences.

## Dependencies for Next Steps
- Define `RoomFederatedNode` wrapping `FederatedNode` with room-specific context
- Implement lightweight summary format for ESP32 participation (3-value histogram)
- Add federated aggregation to PLATO tile store synchronization
- Benchmark federated round latency across distributed Codespaces
