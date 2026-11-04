"""Tests del inicio aleatorio desde checkpoints (P3.7)."""

import gymnasium as gym
import pytest

from mario_env import ENV_ID, MarioEnv
from mario_game.constants import MARIO_W, TILE
from mario_game.game import MarioGame
from mario_game.level import build_level_1, build_level_2
from mario_stub import StubMarioGame


def stub_env(**kwargs):
    return MarioEnv(game_factory=StubMarioGame, **kwargs)


def start_cols(env, n, seed=None):
    """Columnas de inicio de n resets seguidos (el primero con `seed`)."""
    cols = [env.reset(seed=seed)[1]["start_col"]]
    cols += [env.reset()[1]["start_col"] for _ in range(n - 1)]
    return cols


def test_el_inicio_cae_siempre_en_un_checkpoint():
    env = stub_env(random_start=True, random_start_prob=1.0)
    checkpoints = env.game.level.checkpoints
    env.reset(seed=0)
    for _ in range(50):
        _, info = env.reset()
        col = info["start_col"]
        assert col in checkpoints
        # Mario está de verdad en esa columna, como lo coloca el motor.
        assert env.game.mario.x == pytest.approx(col * TILE + (TILE - MARIO_W) / 2)
    assert len(set(start_cols(env, 50, seed=0))) > 1      # no siempre el mismo


def test_sin_random_start_empieza_en_start_col():
    env = stub_env()
    level = env.game.level
    assert set(start_cols(env, 20, seed=0)) == {level.start_col}


def test_con_probabilidad_intermedia_mezcla_inicio_y_checkpoints():
    env = stub_env(random_start=True, random_start_prob=0.5)
    level = env.game.level
    cols = start_cols(env, 100, seed=0)
    assert level.start_col in cols
    assert any(c != level.start_col for c in cols)
    assert set(cols) <= set(level.checkpoints) | {level.start_col}


def test_misma_semilla_mismo_inicio():
    a = stub_env(random_start=True, random_start_prob=1.0)
    b = stub_env(random_start=True, random_start_prob=1.0)
    assert start_cols(a, 10, seed=3) == start_cols(b, 10, seed=3)
    assert start_cols(a, 10, seed=3) != start_cols(a, 10, seed=4)


def test_options_start_col_tiene_prioridad():
    env = stub_env(random_start=True, random_start_prob=1.0)
    for _ in range(10):
        _, info = env.reset(options={"start_col": 50})
        assert info["start_col"] == 50
    _, info = env.reset(options={"start_col": None})     # None = como si no se pasara
    assert info["start_col"] in env.game.level.checkpoints


def test_sin_random_start_no_consume_numeros_aleatorios():
    # Si consumiera, cambiaría la secuencia de P3.9 (acciones pegajosas) al activarlo.
    env = stub_env()
    env.reset(seed=1)
    state = env.np_random.bit_generator.state
    env.reset()
    env.reset(options={"start_col": 50})
    assert env.np_random.bit_generator.state == state


@pytest.mark.parametrize("builder", [build_level_1, build_level_2])
def test_checkpoints_validos_con_el_motor_real(builder):
    env = MarioEnv(game_factory=lambda: MarioGame(builder()),
                   random_start=True, random_start_prob=1.0)
    level = env.game.level
    for col in level.checkpoints:            # todos: el motor lanza ValueError si uno es malo
        obs, info = env.reset(options={"start_col": col})
        assert info["start_col"] == col
        assert env.game.mario.on_ground and not info["dead"]
        assert env.observation_space.contains(obs)
    for col in start_cols(env, 30, seed=0):
        assert col in level.checkpoints


def test_random_start_prob_fuera_de_rango():
    with pytest.raises(ValueError):
        stub_env(random_start=True, random_start_prob=1.5)


def test_random_start_a_traves_de_gym_make():
    env = gym.make(ENV_ID, random_start=True, random_start_prob=1.0)
    _, info = env.reset(seed=0)
    assert info["start_col"] in env.unwrapped.game.level.checkpoints
    env.close()
