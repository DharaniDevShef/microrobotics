"""
Random Baseline - the non-RL genetic-operator arm, used when
main.py's RL_ASSISTED_GENETIC_OPERATIONS = False. Mirrors rl_api.act()'s
decision process (action type -> target node -> params) over the same
grammar-legal action space, but every choice is drawn uniformly at random
instead of by a trained policy - the classic-GA comparison baseline.
"""

from dataclasses import dataclass

import roblet_grammar as rg


@dataclass
class Decision:
    """Minimal stand-in for rl_api.Decision (duck-typed by apply_decision());
    no PPO-bookkeeping fields since there's no policy to train here."""
    action: "rg.Action"
    params: dict
    node_id_a: str
    node_id_b: str = None


def act(G_a, G_b, rng):
    """Uniformly-random counterpart to rl_api.act(): picks a legal action type, target
    node, and params by uniform draw. Caller should check
    rl_api.has_any_legal_action(G_a, G_b) first, same as the RL path."""
    distinct_parents = G_a is not G_b

    legal_actions = [a for a in rg.MUTATION_ACTIONS if rg.any_node_allows(G_a, a)]
    if distinct_parents:
        if any(rg.graft_host_eligible(G_a, n) for n in G_a.nodes):
            legal_actions.append(rg.Action.GRAFT_SUBTREE)
        if (any(rg.swap_eligible(G_a, n) for n in G_a.nodes)
                and any(rg.swap_eligible(G_b, n) for n in G_b.nodes)):
            legal_actions.append(rg.Action.SWAP_SUBTREES)
    if not legal_actions:
        raise ValueError("No legal mutation or crossover action for this (parent_a, parent_b) pair")

    action = rng.choice(legal_actions)

    if action in rg.MUTATION_ACTIONS:
        candidates = [n for n in G_a.nodes if rg.compute_node_action_mask(G_a, n)[action]]
        node_id_a = rng.choice(candidates)
        params = _random_mutation_params(G_a, node_id_a, action, rng)
        return Decision(action=action, params=params, node_id_a=node_id_a)

    if action == rg.Action.GRAFT_SUBTREE:
        host_candidates = [n for n in G_a.nodes if rg.graft_host_eligible(G_a, n)]
        node_id_a = rng.choice(host_candidates)
        host_port = rng.choice(rg.growable_ports(G_a, node_id_a))
        node_id_b = rng.choice(list(G_b.nodes))  # any node of G_b is a legal donor root
        return Decision(action=action, params=dict(host_port=host_port),
                         node_id_a=node_id_a, node_id_b=node_id_b)

    if action == rg.Action.SWAP_SUBTREES:
        a_candidates = [n for n in G_a.nodes if rg.swap_eligible(G_a, n)]
        b_candidates = [n for n in G_b.nodes if rg.swap_eligible(G_b, n)]
        node_id_a = rng.choice(a_candidates)
        node_id_b = rng.choice(b_candidates)
        return Decision(action=action, params={}, node_id_a=node_id_a, node_id_b=node_id_b)

    raise ValueError(f"Unhandled action {action}")


def _random_mutation_params(G, node_id, action, rng):
    if action == rg.Action.ADD_NODE:
        port = rng.choice(rg.growable_ports(G, node_id))
        module_type = rng.choice(rg.MODULE_TYPES)
        hinge_angle = 0.0 if module_type == "non-foldable" else rng.uniform(rg.MIN_HINGE_ANGLE, rg.MAX_HINGE_ANGLE)
        return dict(port=port, module_type=module_type, hinge_angle=hinge_angle)
    if action in (rg.Action.DELETE_NODE, rg.Action.PRUNE_SUBTREE):
        return {}
    if action == rg.Action.MUTATE_FOLD_TYPE:
        return dict(new_fold_type=rng.choice(rg.MODULE_TYPES))
    if action == rg.Action.MUTATE_HINGE_ANGLE:
        return dict(new_angle=rng.uniform(rg.MIN_HINGE_ANGLE, rg.MAX_HINGE_ANGLE))
    if action == rg.Action.TOGGLE_LIGHT_SENSOR:
        return {}
    if action == rg.Action.MUTATE_LIGHT_HINGE_ANGLE:
        return dict(new_angle=rng.uniform(rg.MIN_HINGE_ANGLE, rg.MAX_HINGE_ANGLE))
    if action == rg.Action.RECONNECT_PORT:
        old_port = rng.choice(rg.reconnectable_ports(G, node_id))
        new_port = rng.choice(rg.growable_ports(G, node_id))
        return dict(old_port=old_port, new_port=new_port)
    raise ValueError(f"Unhandled mutation action {action}")
