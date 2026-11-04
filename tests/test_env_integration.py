"""Integración del entorno con el motor real (P3.8).

La fábrica por defecto de MarioEnv ya es MarioGame(build_level_1()). Estos tests
comprueban que las muertes y la bandera del motor real llegan bien a la
recompensa, a terminated y a info.
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from helpers import make_level
from mario_env import (DEATH_PENALTY, ENV_ID, FLAG_REWARD, PROGRESS_REWARD_DIV,
                       STEP_PENALTY, MarioEnv)
from mario_game.game import MarioGame
from mario_game.level import build_level_1

RIGHT, RIGHT_RUN = 1, 3


def real_env(rows, **level_kwargs):
    return MarioEnv(game_factory=lambda: MarioGame(make_level(rows, **level_kwargs)))


def run_until_done(env, action, max_steps=500):
    """Repite `action` hasta que el episodio acabe. Devuelve el último paso y la suma."""
    total = 0.0
    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            return reward, total, terminated, truncated, info
    raise AssertionError(f"el episodio no acabó en {max_steps} pasos")


def test_por_defecto_usa_el_motor_real_con_el_nivel_1():
    env = MarioEnv()
    assert isinstance(env.game, MarioGame)
    level_1 = build_level_1()
    assert env.game.level.tiles == level_1.tiles
    assert env.game.level.flag_col == level_1.flag_col


def test_cada_entorno_tiene_su_propio_level():
    # "Un Level por MarioGame" (INTERFACES.md §4): el motor modifica los bloques ?.
    a, b = MarioEnv(), MarioEnv()
    assert a.game.level is not b.game.level
    assert a.game.level.tiles is not b.game.level.tiles


def test_check_env_con_el_motor_real():
    check_env(MarioEnv())
    check_env(MarioEnv(obs_type="pixels"), skip_render_check=True)
    check_env(gym.make(ENV_ID).unwrapped)


def test_muerte_en_un_foso():
    env = real_env(["#" * 10 + "..." + "#" * 27] * 2, start_col=2)
    env.reset()
    reward, _, terminated, truncated, info = run_until_done(env, RIGHT_RUN)
    assert (terminated, truncated) == (True, False)
    assert info["dead"] is True and info["flag_get"] is False
    assert info["death_reason"] == "pit"
    assert reward < -DEATH_PENALTY + 1       # el último paso incluye el −15


def test_muerte_contra_un_goomba():
    env = real_env(["#" * 40] * 2, start_col=2, goomba_cols=[12])
    env.reset()
    reward, _, terminated, _, info = run_until_done(env, RIGHT)
    assert terminated is True
    assert info["death_reason"] == "goomba"
    assert reward < -DEATH_PENALTY + 1


def test_llegar_a_la_bandera():
    env = real_env(["#" * 40] * 2, start_col=2, flag_col=30)
    _, start_info = env.reset()
    reward, total, terminated, truncated, info = run_until_done(env, RIGHT_RUN)
    assert (terminated, truncated) == (True, False)
    assert info["flag_get"] is True and info["dead"] is False
    assert info["progress"] == 1.0
    assert reward > FLAG_REWARD - 1          # el último paso incluye el +50
    # Recompensa total = distancia recorrida / 10 − 0.01 por paso + 50.
    distance = info["x_pos"] - start_info["x_pos"]
    assert total == pytest.approx(
        distance / PROGRESS_REWARD_DIV - STEP_PENALTY * info["steps"] + FLAG_REWARD)


def test_el_nivel_1_por_defecto_se_puede_jugar_entero():
    # Partidas aleatorias en el nivel por defecto: todas acaban (muerte,
    # bandera o truncado) y el info siempre es coherente con el juego.
    env = MarioEnv(random_start=True)
    rng = np.random.default_rng(0)
    for episode in range(20):
        env.reset(seed=episode)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(int(rng.integers(7)))
            done = terminated or truncated
        assert info["dead"] == env.game.dead and info["flag_get"] == env.game.won
        assert terminated == (info["dead"] or info["flag_get"])
