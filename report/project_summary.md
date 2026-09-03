
University of Sheffield
School of Electrical & Electronic Engineering
Evolution of Robotic Locomotion through Morphological Optimisation
MSc in Artificial Intelligence for Engineering
Final Project Presentation
Student: Dharani Saravanan
Supervisor: Shuhei Miyashita

---

Background
Automated Robot Design
• Automated Robot Design: Replaces manual engineering with AI + Physics Co-Design to evolve morphology and behavior concurrently.
• Bio-Inspired Evolutionary Robotics: Harnesses natural selection to optimize physical configurations without human bias.
• Intelligent Materials & Swarm Behaviour: Builds decentralized, collective intelligence directly from modular physical units.
• RL in Action: Industry benchmarks (e.g., Boston Dynamics / Isaac Sim + PPO) train complex robots across millions of parallelized physics simulations.
• Goal: Bring this evolutionary and deep RL playbook to modular, self-assembling systems.

Image sourced from [https://developer.nvidia.com/](https://developer.nvidia.com/)

---

Context
What are "Roblets"?
• Modular, 3D-printed units equipped with magnetic "jigsaw smart-glue" bonding sites.
• Self-Folding Mechanism: Heat-responsive PVC hinges transform 2D flat sheets into 3D robot modules.
• External Actuation: Driven entirely by an external oscillating magnetic field for locomotion.
• Communication: Light-based "virtual pheromones" for collective swarm behaviour.

(a) Mechanical agitation
(b) Self-assembly
(c) Hot water -> Self-folded 3D robot
(d) Magnetic actuation

Roblets: Robotic Tablets that Self-assemble and Self-fold into a Robot, Han et al. (2023)

---

Motivation
Physics-in-the-Loop Evolution of Self-Assembling Systems
Challenge:
• The Bottleneck: All existing Roblet designs (Han et al., IROS 2023) are hand-crafted by human intuition, severely capping functional variety.
• Combinatorial Explosion: A graph-structured search space for morphology design (6 design variables) makes brute-force or manual design mathematically intractable.

Solution:
• In-the-Loop Evolution: Replaces trial-and-error with an automated, physics-in-the-loop evolutionary search process.
• Embodied Evaluation: Tests candidate morphologies directly against real physics: evaluating 2D-to-3D self-folding, magnetic locomotion, and light-driven (pheromone) response inside the loop.

---

Methodology Justification
Methodology chosen: NSGA-III + Reinforcement-Learning-Guided Graph Transformer Evolution

Multi-Objective Optimization
• 4 objectives, >3-objective Pareto search
• Single-fitness GA bottleneck - premature convergence

Novel Combination - Reinforcement Learning for Structural Edits
• PPO (Reinforcement Learning) guides structural edits (mutation + crossover), not locomotion control

Reference 1 - Neural Graph Evolution (Wang et al., 2019)
• Reused: Graph-as-genotype formulation
• Improved: One continuously-trained policy replaces per-species policy sharing

Reference 2 - Evolving Embodied Intelligence (Wang et al., 2026)
• Reuse: Attention-over-structure for variable morphology
• Improved:

* Multi-head attention per edge (not GAT's single coefficient) - sharper node targeting
* Unified mutation-and-crossover policy (vs. their mutation-only) - skips parent-to-child weight mapping

---

Graph-Based Structural Representation
Genotype: Morphology as a Directed Graph

Why Graph Representations?
Modular robots are topological networks: Modules = Nodes, Connectors = Edges.
Graph format: Python NetworkX Directed graph

---

Physics simulation in MuJoCo
Assembly Robot body actuating in MuJoCo

indo_stats.json:
{
"success": 1,
"physics_ok": 1,
"is_stable": 1,
"B_intensity_T": 0.05,
"total_mass_mg": 7374.65,
"average_torque_Nm": 0.0009,
"total_simulated_steps": 2703,
"average_velocity_mmps": 13.4429,
"total_distance_mm": 94.3769,
"shape_entropy_2d": 0.979288,
"shape_entropy_3d": 1.0,
"pheromone_yaw_response_deg": 51.5425,
"pheromone_speed_response": 0.585938
}

Simulation Outputs for fitness quantification

---

Methodology: Pipeline Overview
Evolution Pipeline

Population Pool -> Grammar Masking / Mirror Symmetry / Collision Gate (legal action/node/port checks) -> MuJoCo Sim - N Parallel Processes (B-Sweep Gait, Light Response: Pheromone Mode [Attractive Mode / Repulsive Mode]) -> Score Objectives -> NSGA-III (pymoo) (Split Feasible / Infeasible, Non-Dominated Sort, Reference-Dir Niching) -> Next Gen -> Reinforcement Learning - PPO (Actor Picks Action -> Mutate + Crossover, Action Advantage / Reward Signal, Critic Scores Value)

---

Methodology: Design Variables, Objectives and Actions

Design Variables:
• Module Type: Non-foldable, Mountain fold, Valley fold
• Graph Adjacency: Port-to-port connectivity (Ports 1, 2, 3)
• Module Count: 2-40
• Hinge Angle: 0° - 45°
• Light-Sensitive Joint Selection: Boolean, per foldable module
• Light-Detection Hinge Angle: 0° - 45°

RL Mutation Actions:
• Add Node
• Delete Node
• Prune Subtree
• Mutate Fold Type
• Mutate Hinge Angle
• Reconnect Port
• Toggle Light Sensor
• Mutate Light Hinge Angle

RL Crossover Actions:
• Graft Subtree
• Swap Subtrees

Multi-Objective Fitness:
• Folded-Gait Velocity: ↑
• Folding Complexity: ↑
• Pheromone Yaw Response: Attractive ↑ | Repulsive ↓
• Pheromone Speed Response: Attractive ↑ | Repulsive ↓

---

Results: Convergence and Fitness trends
GA Convergence - Best vs. Mean Population Fitness
Scalarized fitness across Generation (0 to 56)

Fitness Trends Across Generations - Best Individual per Generation:
• Folded-Gait Velocity (m/s)
• Entropy - Folding-Complexity Gain (Entropy ΔH = H3D - H2D)
• Pheromone Yaw Response (Yaw Response - deg)
• Pheromone Speed Response (Speed Response - Δv/v)

---

Results: Pareto Front
Pareto Front - Final Generation 59:
Scatter plots showing:
• f1 vs f2 (Folded-Gait Velocity vs Entropy-Folding-Complexity Gain)
• f2 vs f3 (Entropy-Folding-Complexity Gain vs Pheromone Yaw Response)
• f3 vs f4 (Pheromone Yaw Response vs Pheromone Speed Response)
• f4 vs f1 (Pheromone Speed Response vs Folded-Gait Velocity)
Legend: Rank 1-4, Pareto front-rank 0

Pareto Front Parallel Coordinates Generation 59:
Normalized fitness coordinates across:
• Folded-Gait Velocity
• Entropy - Folding-Complexity Gain
• Pheromone Yaw Response
• Pheromone Speed Response

---

Results: Morphological Complexity vs Locomotion
Entropy - Folding-Complexity Gain vs. Folded-Gait Velocity
Plot comparing Entropy Δ = H3D - H2D vs Velocity - m/s across Generations (0 to 54)

---

Resulted Morphologies
Left: Pheromone attracted
Right: Pheromone repelled

---

Conclusion and Future work
Conclusions and discussions:
• Combinatorial design challenge tackled effectively using NSGA-III + RL-guided graph edits
• RL is learning - reward, policy/value loss, entropy all trending as expected
• Evolved morphologies re-confirmed under simulation
• Interesting, diverse morphologies emerged across generations
• ~1.2 min per generation, fully parallelized

Future Work:
• Scale to ≥300 generations, larger population
• Close the gap where random genetic operations outperform RL-guided ones
• Multi-seed trials for statistical robustness
• Test evolved morphologies on physical Roblet hardware

---

Thank You
University of Sheffield
