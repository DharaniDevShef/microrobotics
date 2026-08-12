"""
RL API - Graph Transformer (multi-head self-attention over graph
neighborhoods, via torch_geometric's TransformerConv) Actor-Critic policy
that selects and parameterizes ALL roblet_grammar actions - both mutation
(roblet_grammar.MUTATION_ACTIONS, single-graph) AND crossover
(roblet_grammar.CROSSOVER_ACTIONS, two-graph) - from one unified,
grammar-masked action head. Trained with single-step (contextual-bandit)
PPO: state = (parent_a, parent_b), action = one grammar op (+ its
parameters), reward = scalarized objective improvement of the resulting
MuJoCo-evaluated child/children vs. their parent(s) (computed in
moo_api.py after evaluate_population()/objectives_api run). Since each action is
evaluated and scored independently in one shot, the episode length is
always 1 - GAE/discounting reduce to advantage = reward - V(s), which is
what PPOTrainer.update() below computes.

How crossover is handled: mutation only ever needs to look at one graph
(parent_a), but crossover needs to compare TWO graphs to pick a node in
each. So the actor encodes parent_a AND parent_b with its own Graph
Transformer (same weights, run twice), fuses the two pooled embeddings
via cross-attention for the top-level "which action" decision, and adds
one extra head (`partner_node_head`) that scores parent_b's nodes when
the sampled action is GRAFT_SUBTREE (donor root) or SWAP_SUBTREES
(swap partner). moo_api.py no longer picks crossover itself - it just
calls PPOTrainer.select_action(parent_a, parent_b) and applies whatever
the policy decided, exactly like it does for mutation.

Also note: PPO here is a small hand-written actor-critic trained directly
with torch + torch_geometric, not stable_baselines3/sb3-contrib. SB3's
Discrete/MultiDiscrete action spaces assume a fixed-size, gym.Env-shaped
problem; our action space is a variable-size, per-node, grammar-masked
hierarchical choice over a PAIR of graphs of changing size, which is
naturally expressed as a direct policy-gradient loop (this is also the
standard formulation in graph/NAS-controller RL literature) rather than
forced into a padded Box/Discrete gym.Env just to reuse SB3's PPO class.
"""

import random
from dataclasses import dataclass

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
    fraction, is_root]. Edges are added in both directions so message
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


def _mutation_masks(G):
    """(n_nodes, n_mutation_actions) bool tensor - which of
    roblet_grammar.MUTATION_ACTIONS is legal on each node of G."""
    node_ids = list(G.nodes)
    mask = torch.zeros((len(node_ids), len(rg.MUTATION_ACTIONS)), dtype=torch.bool)
    for i, n in enumerate(node_ids):
        node_mask = rg.compute_node_action_mask(G, n)
        for a_idx, action in enumerate(rg.MUTATION_ACTIONS):
            mask[i, a_idx] = node_mask[action]
    return mask


def _graft_host_mask(G):
    return torch.tensor([rg.graft_host_eligible(G, n) for n in G.nodes], dtype=torch.bool)


def _swap_eligible_mask(G):
    return torch.tensor([rg.swap_eligible(G, n) for n in G.nodes], dtype=torch.bool)


def _action_type_mask(mutation_masks, graft_host_mask, swap_a_mask, swap_b_mask, distinct_parents):
    """(len(roblet_grammar.ALL_ACTIONS),) bool tensor for the top-level
    action-type choice. Crossover entries are only ever True when
    parent_a and parent_b are actually two different graphs."""
    mask = torch.zeros(len(rg.ALL_ACTIONS), dtype=torch.bool)
    mask[: len(rg.MUTATION_ACTIONS)] = mutation_masks.any(dim=0)
    graft_idx = rg.ALL_ACTIONS.index(rg.Action.GRAFT_SUBTREE)
    swap_idx = rg.ALL_ACTIONS.index(rg.Action.SWAP_SUBTREES)
    mask[graft_idx] = bool(distinct_parents and graft_host_mask.any())
    mask[swap_idx] = bool(distinct_parents and swap_a_mask.any() and swap_b_mask.any())
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


def has_any_legal_action(G_a, G_b):
    """Whether act() has anything at all it could legally do for this pair
    - moo_api.py checks this before calling select_action() so it can fall
    back to a no-op copy in the (very rare) case nothing is legal."""
    distinct = G_a is not G_b
    if any(rg.any_node_allows(G_a, a) for a in rg.MUTATION_ACTIONS):
        return True
    if distinct and any(rg.graft_host_eligible(G_a, n) for n in G_a.nodes):
        return True
    if distinct and any(rg.swap_eligible(G_a, n) for n in G_a.nodes) and \
            any(rg.swap_eligible(G_b, n) for n in G_b.nodes):
        return True
    return False


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


class _GraphTransformerEncoder(nn.Module):
    """Shared building block for both ActorNet and CriticNet - projects
    node features to `hidden` dims, then runs them through a stack of
    GraphTransformerBlocks. ActorNet and CriticNet each own their OWN
    instance of this (separate weights, no sharing between actor/critic)."""

    def __init__(self, in_dim, hidden, n_blocks, heads):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([GraphTransformerBlock(hidden, heads=heads) for _ in range(n_blocks)])

    def forward(self, data):
        h = self.input_proj(data.x)
        for block in self.blocks:
            h = block(h, data.edge_index)
        batch = torch.zeros(h.size(0), dtype=torch.long)
        pooled = global_mean_pool(h, batch)
        return h, pooled


class ActorNet(nn.Module):
    """Graph Transformer POLICY network. Encodes parent_a AND parent_b
    with its own two-graph-capable encoder (weights are NOT shared with
    CriticNet - two separate models), fuses their pooled embeddings via
    cross-attention for the top-level action-type choice, and produces
    every action's parameters: node_head scores parent_a's nodes
    (mutation target / GRAFT host / SWAP's parent_a side), while
    partner_node_head scores parent_b's nodes (GRAFT donor root / SWAP's
    parent_b side) - this is what lets one policy drive mutation AND
    crossover, per the design doc's "RL for crossover and mutation"."""

    def __init__(self, in_dim=NODE_FEATURE_DIM, hidden=HIDDEN_DIM, n_blocks=2, heads=4):
        super().__init__()
        self.encoder = _GraphTransformerEncoder(in_dim, hidden, n_blocks, heads)
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads=heads, batch_first=True)

        self.action_type_head = nn.Linear(hidden, len(rg.ALL_ACTIONS))
        self.node_head = nn.Linear(hidden, 1)           # scores nodes of parent_a
        self.partner_node_head = nn.Linear(hidden, 1)   # scores nodes of parent_b (crossover only)
        self.port_head = nn.Linear(hidden, 3)
        self.fold_type_head = nn.Linear(hidden, len(rg.MODULE_TYPES))
        self.hinge_angle_head = nn.Linear(hidden, 2)     # (mu_raw, log_std_raw)

    def encode_pair(self, data_a, data_b):
        h_a, pooled_a = self.encoder(data_a)
        h_b, pooled_b = self.encoder(data_b)
        # parent_a's pooled summary attends over parent_b's per-node
        # embeddings, so the action-type decision can react to what's
        # actually available in the potential donor/partner graph
        # (e.g. "don't bother proposing a crossover parent_b has nothing
        # useful to offer").
        cross_out, _ = self.cross_attn(
            pooled_a.unsqueeze(1), h_b.unsqueeze(0), h_b.unsqueeze(0)
        )
        joint = pooled_a + cross_out.squeeze(1)
        return h_a, h_b, joint


class CriticNet(nn.Module):
    """Graph Transformer VALUE network. A second, independently
    parameterized two-graph encoder (mirrors ActorNet's architecture but
    its own separate weights) - pools parent_a and parent_b separately
    and concatenates them into a single state-value estimate, since the
    reward for a crossover decision depends on both parents."""

    def __init__(self, in_dim=NODE_FEATURE_DIM, hidden=HIDDEN_DIM, n_blocks=2, heads=4):
        super().__init__()
        self.encoder = _GraphTransformerEncoder(in_dim, hidden, n_blocks, heads)
        self.value_head = nn.Linear(hidden * 2, 1)

    def forward(self, data_a, data_b):
        _, pooled_a = self.encoder(data_a)
        _, pooled_b = self.encoder(data_b)
        combined = torch.cat([pooled_a, pooled_b], dim=-1)
        return self.value_head(combined).squeeze(0).squeeze(-1)


@dataclass
class Decision:
    action: "rg.Action"
    params: dict
    old_logprob: torch.Tensor
    old_value: torch.Tensor
    # bookkeeping needed to recompute the exact sample during a PPO update
    data_a: Data
    data_b: Data
    mutation_masks: torch.Tensor
    action_type_mask: torch.Tensor
    action_type_idx: int
    node_a_idx: int = None          # index into data_a.node_ids
    node_a_mask: torch.Tensor = None
    node_b_idx: int = None          # index into data_b.node_ids (crossover only)
    node_b_mask: torch.Tensor = None
    node_id_a: str = None
    node_id_b: str = None
    port_mask: torch.Tensor = None
    port_idx: int = None
    fold_type_idx: int = None
    hinge_raw_sample: float = None


def act(actor, critic, G_a, G_b, rng=None):
    """Samples one Decision for the (parent_a=G_a, parent_b=G_b) pair:
    `actor` picks the action (mutation on G_a, or a crossover between G_a
    and G_b) and its parameters; `critic` independently estimates the
    state value used later as the PPO baseline. Caller should check
    has_any_legal_action(G_a, G_b) first."""
    rng = rng or random
    data_a = graph_to_pyg_data(G_a)
    data_b = graph_to_pyg_data(G_b)
    distinct_parents = G_a is not G_b

    mutation_masks = _mutation_masks(G_a)
    graft_host_mask = _graft_host_mask(G_a) if distinct_parents else torch.zeros(G_a.number_of_nodes(), dtype=torch.bool)
    swap_a_mask = _swap_eligible_mask(G_a) if distinct_parents else torch.zeros(G_a.number_of_nodes(), dtype=torch.bool)
    swap_b_mask = _swap_eligible_mask(G_b) if distinct_parents else torch.zeros(G_b.number_of_nodes(), dtype=torch.bool)
    action_type_mask = _action_type_mask(mutation_masks, graft_host_mask, swap_a_mask, swap_b_mask, distinct_parents)
    if not action_type_mask.any():
        raise ValueError("No legal mutation or crossover action for this (parent_a, parent_b) pair")

    with torch.no_grad():
        h_a, h_b, joint = actor.encode_pair(data_a, data_b)

        at_dist = _masked_categorical(actor.action_type_head(joint).squeeze(0), action_type_mask)
        action_type_idx = at_dist.sample()
        action = rg.ALL_ACTIONS[action_type_idx.item()]
        logprob = at_dist.log_prob(action_type_idx)

        params = {}
        node_a_idx = node_b_idx = None
        node_a_mask = node_b_mask = None
        node_id_a = node_id_b = None
        port_mask = port_idx = fold_type_idx = hinge_raw_sample = None

        if action in rg.MUTATION_ACTIONS:
            mutation_idx = rg.MUTATION_ACTIONS.index(action)
            node_a_mask = mutation_masks[:, mutation_idx]
            node_dist = _masked_categorical(actor.node_head(h_a).squeeze(-1), node_a_mask)
            node_a_idx = node_dist.sample()
            node_id_a = data_a.node_ids[node_a_idx.item()]
            logprob = logprob + node_dist.log_prob(node_a_idx)

            h_v = h_a[node_a_idx].unsqueeze(0)

            if action == rg.Action.ADD_NODE:
                port_mask = _port_mask(G_a, node_id_a)
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
                port_mask = _port_mask(G_a, node_id_a)
                port_dist = _masked_categorical(actor.port_head(h_v).squeeze(0), port_mask)
                port_idx = port_dist.sample()
                logprob = logprob + port_dist.log_prob(port_idx)
                # old_port is an unlearned uniform pick among occupied ports
                # (usually only 1-2 candidates) - documented simplification,
                # not part of the PPO logprob.
                old_port = rng.choice(rg.occupied_ports(G_a, node_id_a))
                params = dict(old_port=old_port, new_port=port_idx.item() + 1)
                port_idx = port_idx.item()

            node_a_idx = node_a_idx.item()

        elif action == rg.Action.GRAFT_SUBTREE:
            node_a_mask = graft_host_mask
            host_dist = _masked_categorical(actor.node_head(h_a).squeeze(-1), node_a_mask)
            node_a_idx = host_dist.sample()
            node_id_a = data_a.node_ids[node_a_idx.item()]
            logprob = logprob + host_dist.log_prob(node_a_idx)

            port_mask = _port_mask(G_a, node_id_a)
            port_dist = _masked_categorical(actor.port_head(h_a[node_a_idx].unsqueeze(0)).squeeze(0), port_mask)
            port_idx = port_dist.sample()
            logprob = logprob + port_dist.log_prob(port_idx)

            node_b_mask = torch.ones(len(data_b.node_ids), dtype=torch.bool)
            donor_dist = _masked_categorical(actor.partner_node_head(h_b).squeeze(-1), node_b_mask)
            node_b_idx = donor_dist.sample()
            node_id_b = data_b.node_ids[node_b_idx.item()]
            logprob = logprob + donor_dist.log_prob(node_b_idx)

            params = dict(host_port=port_idx.item() + 1)
            node_a_idx, node_b_idx, port_idx = node_a_idx.item(), node_b_idx.item(), port_idx.item()

        elif action == rg.Action.SWAP_SUBTREES:
            node_a_mask = swap_a_mask
            a_dist = _masked_categorical(actor.node_head(h_a).squeeze(-1), node_a_mask)
            node_a_idx = a_dist.sample()
            node_id_a = data_a.node_ids[node_a_idx.item()]
            logprob = logprob + a_dist.log_prob(node_a_idx)

            node_b_mask = swap_b_mask
            b_dist = _masked_categorical(actor.partner_node_head(h_b).squeeze(-1), node_b_mask)
            node_b_idx = b_dist.sample()
            node_id_b = data_b.node_ids[node_b_idx.item()]
            logprob = logprob + b_dist.log_prob(node_b_idx)

            node_a_idx, node_b_idx = node_a_idx.item(), node_b_idx.item()

        value = critic(data_a, data_b)

    return Decision(
        action=action, params=params,
        old_logprob=logprob.detach(), old_value=value.detach(),
        data_a=data_a, data_b=data_b,
        mutation_masks=mutation_masks, action_type_mask=action_type_mask,
        action_type_idx=action_type_idx.item(),
        node_a_idx=node_a_idx, node_a_mask=node_a_mask,
        node_b_idx=node_b_idx, node_b_mask=node_b_mask,
        node_id_a=node_id_a, node_id_b=node_id_b,
        port_mask=port_mask, port_idx=port_idx, fold_type_idx=fold_type_idx,
        hinge_raw_sample=hinge_raw_sample,
    )


def apply_decision(G_a, G_b, decision):
    """Executes `decision` (from act()) against (G_a, G_b) via
    roblet_grammar, returning a LIST of resulting child graphs - length 1
    for mutation/GRAFT_SUBTREE, length 2 for SWAP_SUBTREES (it produces
    one recombined offspring for each parent)."""
    a, p = decision.action, decision.params
    n_a, n_b = decision.node_id_a, decision.node_id_b

    if a == rg.Action.ADD_NODE:
        return [rg.add_node(G_a, n_a, p["port"], p["module_type"], hinge_angle=p["hinge_angle"])]
    if a == rg.Action.DELETE_NODE:
        return [rg.delete_node(G_a, n_a)]
    if a == rg.Action.PRUNE_SUBTREE:
        return [rg.prune_subtree(G_a, n_a)]
    if a == rg.Action.MUTATE_FOLD_TYPE:
        return [rg.mutate_fold_type(G_a, n_a, p["new_fold_type"])]
    if a == rg.Action.MUTATE_HINGE_ANGLE:
        return [rg.mutate_hinge_angle(G_a, n_a, p["new_angle"])]
    if a == rg.Action.RECONNECT_PORT:
        return [rg.reconnect_port(G_a, n_a, p["old_port"], p["new_port"])]
    if a == rg.Action.GRAFT_SUBTREE:
        return [rg.graft_subtree(G_a, n_a, p["host_port"], G_b, n_b)]
    if a == rg.Action.SWAP_SUBTREES:
        new_a, new_b = rg.swap_subtrees(G_a, n_a, G_b, n_b)
        return [new_a, new_b]
    raise ValueError(f"Unhandled action {a}")


def _recompute_actor(actor, decision):
    """Fresh forward pass through the (possibly-updated) actor, returning
    (new_logprob, entropy) for the exact sample recorded in `decision` -
    used by PPOTrainer.update() to form the clipped policy ratio."""
    h_a, h_b, joint = actor.encode_pair(decision.data_a, decision.data_b)

    at_dist = _masked_categorical(actor.action_type_head(joint).squeeze(0), decision.action_type_mask)
    at_idx = torch.tensor(decision.action_type_idx)
    logprob = at_dist.log_prob(at_idx)
    entropy = at_dist.entropy()

    action = decision.action

    if action in rg.MUTATION_ACTIONS:
        node_a_idx_t = torch.tensor(decision.node_a_idx)
        node_dist = _masked_categorical(actor.node_head(h_a).squeeze(-1), decision.node_a_mask)
        logprob = logprob + node_dist.log_prob(node_a_idx_t)
        entropy = entropy + node_dist.entropy()

        h_v = h_a[decision.node_a_idx].unsqueeze(0)

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

    elif action == rg.Action.GRAFT_SUBTREE:
        node_a_idx_t = torch.tensor(decision.node_a_idx)
        host_dist = _masked_categorical(actor.node_head(h_a).squeeze(-1), decision.node_a_mask)
        logprob = logprob + host_dist.log_prob(node_a_idx_t)
        entropy = entropy + host_dist.entropy()

        port_dist = _masked_categorical(
            actor.port_head(h_a[decision.node_a_idx].unsqueeze(0)).squeeze(0), decision.port_mask)
        logprob = logprob + port_dist.log_prob(torch.tensor(decision.port_idx))
        entropy = entropy + port_dist.entropy()

        donor_dist = _masked_categorical(actor.partner_node_head(h_b).squeeze(-1), decision.node_b_mask)
        logprob = logprob + donor_dist.log_prob(torch.tensor(decision.node_b_idx))
        entropy = entropy + donor_dist.entropy()

    elif action == rg.Action.SWAP_SUBTREES:
        a_dist = _masked_categorical(actor.node_head(h_a).squeeze(-1), decision.node_a_mask)
        logprob = logprob + a_dist.log_prob(torch.tensor(decision.node_a_idx))
        entropy = entropy + a_dist.entropy()

        b_dist = _masked_categorical(actor.partner_node_head(h_b).squeeze(-1), decision.node_b_mask)
        logprob = logprob + b_dist.log_prob(torch.tensor(decision.node_b_idx))
        entropy = entropy + b_dist.entropy()

    return logprob, entropy


def _recompute_critic(critic, decision):
    """Fresh forward pass through the (possibly-updated) critic, returning
    the new value estimate V(s) for the (parent_a, parent_b) state stored
    in `decision`."""
    return critic(decision.data_a, decision.data_b)


# ---------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------

class PPOTrainer:
    """Owns the two independent Graph Transformer models: `self.actor`
    (policy, over BOTH mutation and crossover) and `self.critic`
    (state-value baseline). They are separate nn.Module instances with
    separate parameters and separate Adam optimizers - no weight sharing.
    Each is stepped with its own loss (actor: clipped PPO surrogate +
    entropy bonus; critic: MSE against the observed reward); the only
    place they interact is through the numbers (the critic's value
    estimate sets the advantage the actor's surrogate is scaled by),
    never through shared gradients.
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

    def select_action(self, G_a, G_b):
        """Picks one grammar action - mutation on G_a, or a crossover
        between G_a and G_b - via the shared actor/critic. Caller should
        check has_any_legal_action(G_a, G_b) first."""
        return act(self.actor, self.critic, G_a, G_b, rng=self.rng)

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
