"""Tests del parámetro `level` del entorno (P3.10): niveles 1 y 2."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from helpers import make_level
from mario_env import ENV_ID, LEVEL_BUILDERS, MarioEnv
from mario_game.level import build_level_1, build_level_2
from mario_stub import StubMarioGame


def test_carga_el_nivel_1_y_el_2():
    env_1, env_2 = MarioEnv(level=1), MarioEnv(level=2)
    assert env_1.game.level.tiles == build_level_1().tiles
    assert env_2.game.level.tiles == build_level_2().tiles
    assert env_1.game.level.tiles != env_2.game.level.tiles
    assert MarioEnv().game.level.tiles == build_level_1().tiles     # por defecto, el 1
    assert sorted(LEVEL_BUILDERS) == [1, 2]


@pytest.mark.parametrize("level", [1, 2])
def test_check_env_con_cada_nivel(level):
    check_env(gym.make(ENV_ID, level=level).unwrapped)
    check_env(MarioEnv(level=level, obs_type="pixels"), skip_render_check=True)


def test_random_start_usa_los_checkpoints_del_nivel_elegido():
    env = MarioEnv(level=2, random_start=True, random_start_prob=1.0)
    checkpoints_2 = build_level_2().checkpoints
    env.reset(seed=0)
    starts = {env.reset()[1]["start_col"] for _ in range(60)}
    assert starts <= set(checkpoints_2)
    # El nivel 2 tiene checkpoints que el nivel 1 no tiene.
    assert starts - set(build_level_1().checkpoints)


def test_partidas_en_el_nivel_2():
    env = MarioEnv(level=2, random_start=True)
    rng = np.random.default_rng(0)
    for episode in range(10):
        obs, info = env.reset(seed=episode)
        assert info["level"] == 2
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(int(rng.integers(7)))
            assert env.observation_space.contains(obs)
            done = terminated or truncated


def test_level_acepta_una_funcion():
    env = MarioEnv(level=build_level_2)
    assert env.game.level.tiles == build_level_2().tiles
    assert env.reset()[1]["level"] == "build_level_2"

    corto = MarioEnv(level=lambda: make_level(["#" * 30] * 2, start_col=2, flag_col=20))
    corto.reset()
    assert corto.game.level.width == 30
    for _ in range(100):
        _, _, terminated, _, info = corto.step(3)
        if terminated:
            break
    assert info["flag_get"] is True


def test_cada_entorno_tiene_su_propio_level_tambien_en_el_2():
    a, b = MarioEnv(level=2), MarioEnv(level=2)
    assert a.game.level is not b.game.level


def test_game_factory_tiene_prioridad_sobre_level():
    env = MarioEnv(game_factory=StubMarioGame, level=2)
    assert isinstance(env.game, StubMarioGame)
    assert env.reset()[1]["level"] is None


@pytest.mark.parametrize("level", [0, 3, "2", 1.0, True, None])
def test_nivel_desconocido(level):
    with pytest.raises(ValueError):
        MarioEnv(level=level)


def test_info_trae_el_nivel():
    assert MarioEnv(level=1).reset()[1]["level"] == 1
    env = gym.make(ENV_ID, level=2)
    _, info = env.reset(seed=0)
    assert info["level"] == 2
    _, _, _, _, info = env.step(1)
    assert info["level"] == 2
    env.close()
