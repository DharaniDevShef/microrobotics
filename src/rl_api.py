"""
RL API - Graph Transformer (multi-head self-attention over graph
neighborhoods, via torch_geometric's TransformerConv) Actor-Critic policy
that selects and parameterizes roblet_grammar mutation operators, trained
with single-step (contextual-bandit) PPO: state = parent graph, action =
one mutation op (+ its parameters), reward = scalarized objective
improvement of the resulting MuJoCo-evaluated child vs. its parent
(computed in moo_api.py after mujoco_api/objectives_api run). Since each
mutation is evaluated and scored independently in one shot, the episode
length is always 1 - GAE/discounting reduce to advantage = reward - V(s),
which is what PPOTrainer.update() below computes.

Scope note: per the design doc, RL should drive BOTH mutation and
crossover. This pass covers mutation only (roblet_grammar.MUTATION_ACTIONS,
the six ADD_NODE/DELETE_NODE/PRUNE_SUBTREE/MUTATE_FOLD_TYPE/
MUTATE_HINGE_ANGLE/RECONNECT_PORT ops). Crossover (GRAFT_SUBTREE /
SWAP_SUBTREES) is still grammar-legal, but moo_api.py picks it via random
choice rather than a learned policy - a learned crossover policy needs a
two-graph cross-attention architecture, left as a follow-up.

Also note: PPO here is a small hand-written actor-critic trained directly
with torch + torch_geometric, not stable_baselines3/sb3-contrib. SB3's
Discrete/MultiDiscrete action spaces assume a fixed-size, gym.Env-shaped
problem; our action space is a variable-size, per-node, grammar-masked
hierarchical choice over graphs of changing size, which is naturally
expressed as a direct policy-gradient loop (this is also the standard
formulation in graph/NAS-controller RL literature) rather than forced
into a padded Box/Discrete gym.Env just to reuse SB3's PPO class.
"""

import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool

import roblet_grammar as rg

NODE_FEATURE_DIM = 7
HIDDEN_DIM = 32
NEG_INF = -1e9


# ---------------------------------------------------------------------
# Graph <-> tensor conversion
# ---------------------------------------------------------------------

def graph_to_pyg_data(G):
    """nx.DiGraph -> torch_geometric.data.Data, with node feature layout:
    [one-hot module_type (3), hinge_angle/90, depth/10 (clipped), free-port
    fraction, is_root]. Edges are added in both directions so GAT message
    passing isn't limited to the parent->child tree orientation."""
    node_ids = list(G.nodes)
    index_of = {n: i for i, n in enumerate(node_ids)}

    feats = torch.zeros((len(node_ids), NODE_FEATURE_DIM), dtype=torch.float32)
    for i, n in enumerate(node_ids):
        attrs = G.nodes[n]
        feats[i, attrs["type_id"]] = 1.0
        feats[i, 3] = attrs["hinge_angle"] / rg.MAX_HINGE_ANGLE
        feats[i, 4] = min(attrs["depth"], 10) / 10.0
        feats[i, 5] = len(rg.free_ports(G, n)) / 3.0
        feats[i, 6] = 1.0 if rg.is_root(G, n) else 0.0

    edges = []
    for u, v in G.edges:
        edges.append((index_of[u], index_of[v]))
        edges.append((index_of[v], index_of[u]))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges else torch.zeros((2, 0), dtype=torch.long)
    )

    data = Data(x=feats, edge_index=edge_index)
    data.node_ids = node_ids
    return data


def _action_masks(G):
    """(n_nodes, n_mutation_actions) bool tensor."""
    node_ids = list(G.nodes)
    mask = torch.zeros((len(node_ids), len(rg.MUTATION_ACTIONS)), dtype=torch.bool)
    for i, n in enumerate(node_ids):
        node_mask = rg.compute_node_action_mask(G, n)
        for a_idx, action in enumerate(rg.MUTATION_ACTIONS):
            mask[i, a_idx] = node_mask[action]
    return mask


def _port_mask(G, node_id):
    mask = torch.zeros(3, dtype=torch.bool)
    for p in rg.free_ports(G, node_id):
        mask[p - 1] = True
    return mask


def _masked_categorical(logits, mask):
    if not torch.any(mask):
        # Should not happen if callers check masks first; degrade to
        # uniform rather than crash on an all -inf distribution.
        return torch.distributions.Categorical(logits=torch.zeros_like(logits))
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, NEG_INF))


def _hinge_angle_dist(mu_raw, log_std_raw):
    std = F.softplus(log_std_raw) + 1e-3
    return torch.distributions.Normal(mu_raw, std)


# ---------------------------------------------------------------------
# Policy network - Graph Transformer encoder
# ---------------------------------------------------------------------

class GraphTransformerBlock(nn.Module):
    """One Graph Transformer block: multi-head self-attention over graph
    neighborhoods (Shi et al. 2021's `TransformerConv` - dot-product
    Q/K/V attention plus a learned gated residual, the standard "Graph
    Transformer" conv) followed by a position-wise feed-forward
    sublayer, each wrapped in a residual connection + LayerNorm - the
    graph analogue of a plain Transformer encoder block."""

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "hidden dim must be divisible by heads"
        self.attn = TransformerConv(dim, dim // heads, heads=heads, concat=True,
                                     beta=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, edge_index):
        x = self.norm1(x + self.attn(x, edge_index))
        x = self.norm2(x + self.ffn(x))
        return x


class ActorNet(nn.Module):
    """Graph Transformer POLICY network. Encodes the graph with its own
    stack of `GraphTransformerBlock`s (weights are NOT shared with
    CriticNet - two separate models, per the design doc's "Policy
    Controller" / critic split) and produces the hierarchical action
    distributions (action type -> target node -> action-specific params)."""

    def __init__(self, in_dim=NODE_FEATURE_DIM, hidden=HIDDEN_DIM, n_blocks=2, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([GraphTransformerBlock(hidden, heads=heads) for _ in range(n_blocks)])

        self.action_type_head = nn.Linear(hidden, len(rg.MUTATION_ACTIONS))
        self.node_head = nn.Linear(hidden, 1)
        self.port_head = nn.Linear(hidden, 3)
        self.fold_type_head = nn.Linear(hidden, len(rg.MODULE_TYPES))
        self.hinge_angle_head = nn.Linear(hidden, 2)  # (mu_raw, log_std_raw)

    def encode(self, data):
        h = self.input_proj(data.x)
        for block in self.blocks:
            h = block(h, data.edge_index)
        batch = torch.zeros(h.size(0), dtype=torch.long)
        pooled = global_mean_pool(h, batch)
        return h, pooled


class CriticNet(nn.Module):
    """Graph Transformer VALUE network. A second, independently
    parameterized Graph Transformer encoder (same block architecture as
    ActorNet, but its own separate weights - it never shares a forward
    pass or gradients with the actor) feeding a single scalar
    state-value head. See PPOTrainer for how the two are wired together
    during a training update."""

    def __init__(self, in_dim=NODE_FEATURE_DIM, hidden=HIDDEN_DIM, n_blocks=2, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([GraphTransformerBlock(hidden, heads=heads) for _ in range(n_blocks)])
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, data):
        h = self.input_proj(data.x)
        for block in self.blocks:
            h = block(h, data.edge_index)
        batch = torch.zeros(h.size(0), dtype=torch.long)
        pooled = global_mean_pool(h, batch)
        return self.value_head(pooled).squeeze(0).squeeze(-1)


@dataclass
class Decision:
    action: "rg.Action"
    node_id: str
    params: dict
    old_logprob: torch.Tensor
    old_value: torch.Tensor
    data: Data
    action_masks: torch.Tensor
    action_type_idx: int
    node_idx: int
    port_mask: torch.Tensor = None
    port_idx: int = None
    fold_type_idx: int = None
    hinge_raw_sample: float = None


def act(actor, critic, G, rng=None):
    """Samples one mutation Decision for graph G: `actor` picks the action
    (and its parameters), `critic` independently estimates the state
    value used later as the PPO baseline. Caller must ensure at least one
    roblet_grammar.MUTATION_ACTIONS is legal somewhere in G."""
    rng = rng or random
    data = graph_to_pyg_data(G)
    masks = _action_masks(G)
    if not masks.any():
        raise ValueError("No legal mutation action anywhere in this graph")

    with torch.no_grad():
        h, pooled = actor.encode(data)

        action_type_mask = masks.any(dim=0)
        at_dist = _masked_categorical(actor.action_type_head(pooled).squeeze(0), action_type_mask)
        action_type_idx = at_dist.sample()
        action = rg.MUTATION_ACTIONS[action_type_idx.item()]
        logprob = at_dist.log_prob(action_type_idx)

        node_mask = masks[:, action_type_idx.item()]
        node_dist = _masked_categorical(actor.node_head(h).squeeze(-1), node_mask)
        node_idx = node_dist.sample()
        node_id = data.node_ids[node_idx.item()]
        logprob = logprob + node_dist.log_prob(node_idx)

        h_v = h[node_idx].unsqueeze(0)
        params = {}
        port_mask = port_idx = fold_type_idx = hinge_raw_sample = None

        if action == rg.Action.ADD_NODE:
            port_mask = _port_mask(G, node_id)
            port_dist = _masked_categorical(actor.port_head(h_v).squeeze(0), port_mask)
            port_idx = port_dist.sample()
            logprob = logprob + port_dist.log_prob(port_idx)

            ft_dist = torch.distributions.Categorical(logits=actor.fold_type_head(h_v).squeeze(0))
            fold_type_idx = ft_dist.sample()
            logprob = logprob + ft_dist.log_prob(fold_type_idx)
            new_fold_type = rg.MODULE_TYPES[fold_type_idx.item()]

            mu_raw, log_std_raw = actor.hinge_angle_head(h_v).squeeze(0)
            hdist = _hinge_angle_dist(mu_raw, log_std_raw)
            hinge_raw_sample = hdist.sample()
            logprob = logprob + hdist.log_prob(hinge_raw_sample)
            angle = (rg.MAX_HINGE_ANGLE * torch.sigmoid(hinge_raw_sample)).item()

            params = dict(port=port_idx.item() + 1, module_type=new_fold_type,
                          hinge_angle=(0.0 if new_fold_type == "non-foldable" else angle))
            port_idx, fold_type_idx = port_idx.item(), fold_type_idx.item()
            hinge_raw_sample = hinge_raw_sample.item()

        elif action in (rg.Action.DELETE_NODE, rg.Action.PRUNE_SUBTREE):
            params = {}

        elif action == rg.Action.MUTATE_FOLD_TYPE:
            ft_dist = torch.distributions.Categorical(logits=actor.fold_type_head(h_v).squeeze(0))
            fold_type_idx = ft_dist.sample()
            logprob = logprob + ft_dist.log_prob(fold_type_idx)
            params = dict(new_fold_type=rg.MODULE_TYPES[fold_type_idx.item()])
            fold_type_idx = fold_type_idx.item()

        elif action == rg.Action.MUTATE_HINGE_ANGLE:
            mu_raw, log_std_raw = actor.hinge_angle_head(h_v).squeeze(0)
            hdist = _hinge_angle_dist(mu_raw, log_std_raw)
            hinge_raw_sample = hdist.sample()
            logprob = logprob + hdist.log_prob(hinge_raw_sample)
            angle = (rg.MAX_HINGE_ANGLE * torch.sigmoid(hinge_raw_sample)).item()
            params = dict(new_angle=angle)
            hinge_raw_sample = hinge_raw_sample.item()

        elif action == rg.Action.RECONNECT_PORT:
            port_mask = _port_mask(G, node_id)
            port_dist = _masked_categorical(actor.port_head(h_v).squeeze(0), port_mask)
            port_idx = port_dist.sample()
            logprob = logprob + port_dist.log_prob(port_idx)
            # old_port is an unlearned uniform pick among occupied ports
            # (usually only 1-2 candidates) - documented simplification,
            # not part of the PPO logprob.
            old_port = rng.choice(rg.occupied_ports(G, node_id))
            params = dict(old_port=old_port, new_port=port_idx.item() + 1)
            port_idx = port_idx.item()

        value = critic(data)

    return Decision(
        action=action, node_id=node_id, params=params,
        old_logprob=logprob.detach(), old_value=value.detach(),
        data=data, action_masks=masks,
        action_type_idx=action_type_idx.item(), node_idx=node_idx.item(),
        port_mask=port_mask, port_idx=port_idx, fold_type_idx=fold_type_idx,
        hinge_raw_sample=hinge_raw_sample,
    )


def apply_decision(G, decision):
    """Executes `decision` (from act()) against G via roblet_grammar."""
    a, n, p = decision.action, decision.node_id, decision.params
    if a == rg.Action.ADD_NODE:
        return rg.add_node(G, n, p["port"], p["module_type"], hinge_angle=p["hinge_angle"])
    if a == rg.Action.DELETE_NODE:
        return rg.delete_node(G, n)
    if a == rg.Action.PRUNE_SUBTREE:
        return rg.prune_subtree(G, n)
    if a == rg.Action.MUTATE_FOLD_TYPE:
        return rg.mutate_fold_type(G, n, p["new_fold_type"])
    if a == rg.Action.MUTATE_HINGE_ANGLE:
        return rg.mutate_hinge_angle(G, n, p["new_angle"])
    if a == rg.Action.RECONNECT_PORT:
        return rg.reconnect_port(G, n, p["old_port"], p["new_port"])
    raise ValueError(f"Unhandled action {a}")


def _recompute_actor(actor, decision):
    """Fresh forward pass through the (possibly-updated) actor, returning
    (new_logprob, entropy) for the exact sample recorded in `decision` -
    used by PPOTrainer.update() to form the clipped policy ratio."""
    data, masks = decision.data, decision.action_masks
    h, pooled = actor.encode(data)

    action_type_mask = masks.any(dim=0)
    at_dist = _masked_categorical(actor.action_type_head(pooled).squeeze(0), action_type_mask)
    at_idx = torch.tensor(decision.action_type_idx)
    logprob = at_dist.log_prob(at_idx)
    entropy = at_dist.entropy()

    node_mask = masks[:, decision.action_type_idx]
    node_dist = _masked_categorical(actor.node_head(h).squeeze(-1), node_mask)
    node_idx = torch.tensor(decision.node_idx)
    logprob = logprob + node_dist.log_prob(node_idx)
    entropy = entropy + node_dist.entropy()

    h_v = h[decision.node_idx].unsqueeze(0)
    action = decision.action

    if action == rg.Action.ADD_NODE:
        port_dist = _masked_categorical(actor.port_head(h_v).squeeze(0), decision.port_mask)
        logprob = logprob + port_dist.log_prob(torch.tensor(decision.port_idx))
        entropy = entropy + port_dist.entropy()

        ft_dist = torch.distributions.Categorical(logits=actor.fold_type_head(h_v).squeeze(0))
        logprob = logprob + ft_dist.log_prob(torch.tensor(decision.fold_type_idx))
        entropy = entropy + ft_dist.entropy()

        mu_raw, log_std_raw = actor.hinge_angle_head(h_v).squeeze(0)
        hdist = _hinge_angle_dist(mu_raw, log_std_raw)
        logprob = logprob + hdist.log_prob(torch.tensor(decision.hinge_raw_sample))
        entropy = entropy + hdist.entropy()

    elif action == rg.Action.MUTATE_FOLD_TYPE:
        ft_dist = torch.distributions.Categorical(logits=actor.fold_type_head(h_v).squeeze(0))
        logprob = logprob + ft_dist.log_prob(torch.tensor(decision.fold_type_idx))
        entropy = entropy + ft_dist.entropy()

    elif action == rg.Action.MUTATE_HINGE_ANGLE:
        mu_raw, log_std_raw = actor.hinge_angle_head(h_v).squeeze(0)
        hdist = _hinge_angle_dist(mu_raw, log_std_raw)
        logprob = logprob + hdist.log_prob(torch.tensor(decision.hinge_raw_sample))
        entropy = entropy + hdist.entropy()

    elif action == rg.Action.RECONNECT_PORT:
        port_dist = _masked_categorical(actor.port_head(h_v).squeeze(0), decision.port_mask)
        logprob = logprob + port_dist.log_prob(torch.tensor(decision.port_idx))
        entropy = entropy + port_dist.entropy()

    return logprob, entropy


def _recompute_critic(critic, decision):
    """Fresh forward pass through the (possibly-updated) critic, returning
    the new value estimate V(s) for the state stored in `decision`."""
    return critic(decision.data)


# ---------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------

class PPOTrainer:
    """Owns the two independent Graph Transformer models: `self.actor`
    (policy) and `self.critic` (state-value baseline). They are separate
    nn.Module instances with separate parameters and separate Adam
    optimizers - no weight sharing, so `self.actor.parameters()` and
    `self.critic.parameters()` are disjoint sets. Each is stepped with
    its own loss (actor: clipped PPO surrogate + entropy bonus; critic:
    MSE against the observed reward); the only place they interact is
    through the numbers (the critic's value estimate sets the advantage
    the actor's surrogate is scaled by), never through shared gradients.
    """

    def __init__(self, lr=3e-4, clip_eps=0.2, entropy_coef=0.01, value_coef=0.5,
                 epochs=4, seed=None):
        self.actor = ActorNet()
        self.critic = CriticNet()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs
        self.rng = random.Random(seed)
        self.buffer = []  # list[(Decision, reward)]
        self.history = {"policy_loss": [], "value_loss": [], "entropy": [], "reward": []}

    def select_mutation(self, G):
        return act(self.actor, self.critic, G, rng=self.rng)

    def record(self, decision, reward):
        self.buffer.append((decision, reward))
        self.history["reward"].append(reward)

    def update(self):
        """Runs `self.epochs` clipped-surrogate PPO passes over everything
        recorded since the last update, then clears the buffer. Transitions
        are looped one-by-one (not batched via torch_geometric.Batch)
        since population sizes here are small (tens per generation) -
        batching would be the natural speed-up for larger populations."""
        if not self.buffer:
            return

        rewards = torch.tensor([r for _, r in self.buffer], dtype=torch.float32)
        old_values = torch.stack([d.old_value for d, _ in self.buffer])
        old_logprobs = torch.stack([d.old_logprob for d, _ in self.buffer])

        advantages = rewards - old_values
        if advantages.numel() > 1 and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            policy_losses, value_losses, entropies = [], [], []
            for (decision, _), advantage, old_logprob, ret in zip(
                self.buffer, advantages, old_logprobs, rewards
            ):
                new_logprob, entropy = _recompute_actor(self.actor, decision)
                ratio = torch.exp(new_logprob - old_logprob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
                policy_losses.append(-torch.min(surr1, surr2))
                entropies.append(entropy)

                new_value = _recompute_critic(self.critic, decision)
                value_losses.append((new_value - ret) ** 2)

            policy_loss = torch.stack(policy_losses).mean()
            entropy_bonus = torch.stack(entropies).mean()
            value_loss = torch.stack(value_losses).mean()

            actor_loss = policy_loss - self.entropy_coef * entropy_bonus
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            critic_loss = self.value_coef * value_loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.history["policy_loss"].append(policy_loss.item())
            self.history["value_loss"].append(value_loss.item())
            self.history["entropy"].append(entropy_bonus.item())

        self.buffer.clear()
