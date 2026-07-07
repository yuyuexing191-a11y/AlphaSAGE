import torch

from src.alpha_gfn.env.core import GFNEnvCore
from alphagen.data.expression import Div, Feature, Inv, Log, Pow
from alphagen.data.tokens import FeatureToken, OperatorToken
from alphagen_qlib.stock_data import FeatureType


class DummyPool:
    pass


def make_env(enable_structure_mask=True):
    return GFNEnvCore(
        pool=DummyPool(),
        encoder=None,
        device=torch.device("cpu"),
        mask_dropout_prob=0.0,
        enable_structure_mask=enable_structure_mask,
    )


def action_id(env, token):
    for idx, candidate in env.id_to_token_map.items():
        if str(candidate) == str(token):
            return idx
    raise AssertionError(f"token not found: {token}")


def make_state(env, tokens):
    ids = [env.token_to_id_map[env.beg_token[0]]] + [action_id(env, token) for token in tokens]
    values = ids + [-1] * (env.state_shape[0] - len(ids))
    return torch.tensor(values, dtype=torch.long, device=env.device)


def test_structure_mask_blocks_more_risky_ops_after_dangerous_prefix():
    env = make_env()
    tokens = [
        FeatureToken(FeatureType.CLOSE),
        OperatorToken(Inv),
        OperatorToken(Log),
        FeatureToken(FeatureType.OPEN),
        OperatorToken(Pow),
    ]
    state = make_state(env, tokens)

    valid_actions = [True] * env.n_actions
    env._apply_structure_aware_action_mask(valid_actions, state)

    for op in (Div, Inv, Log, Pow):
        assert not valid_actions[action_id(env, OperatorToken(op))]
    assert valid_actions[action_id(env, FeatureToken(FeatureType.LOW))]


def test_structure_mask_does_not_block_risky_ops_for_simple_prefix():
    env = make_env()
    state = make_state(env, [FeatureToken(FeatureType.CLOSE)])

    valid_actions = [True] * env.n_actions
    env._apply_structure_aware_action_mask(valid_actions, state)

    for op in (Div, Inv, Log, Pow):
        assert valid_actions[action_id(env, OperatorToken(op))]
