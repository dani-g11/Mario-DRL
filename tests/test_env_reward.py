"""Tests de la recompensa, la terminación y el truncado del entorno (P3.5).

Casi todos usan el stub: es determinista y avanza en línea recta, así que la
recompensa de cada paso se puede calcular exacta. Los dos últimos usan el motor
real para los casos que el stub no tiene (retroceder y chocar con una pared).
"""

import pytest

from helpers import make_level
from mario_env import (DEATH_PENALTY, FLAG_REWARD, MAX_STEPS,
                       PROGRESS_REWARD_DIV, STEP_PENALTY, STUCK_STEPS,
                       MarioEnv)
from mario_game.constants import RUN_MAX_SPEED, TILE, WALK_MAX_SPEED
from mario_game.game import MarioGame
from mario_stub import StubMarioGame

NOOP, RIGHT, RIGHT_RUN, LEFT = 0, 1, 3, 6
FLAT = ["#" * 40, "#" * 40]


def stub_env(**kwargs):
    stub_kwargs = {k: kwargs.pop(k) for k in ("time_limit", "level") if k in kwargs}
    return MarioEnv(game_factory=lambda: StubMarioGame(**stub_kwargs), **kwargs)


def test_valores_por_defecto_del_plan():
    assert (PROGRESS_REWARD_DIV, STEP_PENALTY) == (10, 0.01)
    assert (DEATH_PENALTY, FLAG_REWARD) == (15, 50)
    assert (STUCK_STEPS, MAX_STEPS) == (150, 3000)
    env = stub_env()
    assert (env.stuck_steps, env.max_steps) == (150, 3000)


def test_recompensa_por_avanzar():
    env = stub_env()
    env.reset()
    _, reward, terminated, truncated, _ = env.step(RIGHT)
    assert reward == pytest.approx(4 * WALK_MAX_SPEED / 10 - 0.01)     # 0.59
    assert isinstance(reward, float)
    assert (terminated, truncated) == (False, False)
    _, reward, *_ = env.step(RIGHT_RUN)
    assert reward == pytest.approx(4 * RUN_MAX_SPEED / 10 - 0.01)      # 0.99


def test_quedarse_quieto_cuesta_y_trunca_a_los_150_pasos():
    env = stub_env()
    env.reset()
    for i in range(1, STUCK_STEPS):
        _, reward, terminated, truncated, _ = env.step(NOOP)
        assert reward == pytest.approx(-STEP_PENALTY)
        assert (terminated, truncated) == (False, False), f"paso {i}"
    _, _, terminated, truncated, _ = env.step(NOOP)                 # paso 150
    assert (terminated, truncated) == (False, True)


def test_avanzar_reinicia_el_contador_sin_progreso():
    env = stub_env()
    env.reset()
    for _ in range(STUCK_STEPS - 1):
        env.step(NOOP)
    env.step(RIGHT)                          # nuevo máximo de x
    for _ in range(STUCK_STEPS - 1):
        _, _, _, truncated, _ = env.step(NOOP)
        assert truncated is False
    _, _, _, truncated, _ = env.step(NOOP)
    assert truncated is True


def test_muerte_por_tiempo_termina_y_resta_15():
    env = stub_env(time_limit=8)             # 2 pasos de 4 frames
    env.reset()
    _, reward, terminated, _, _ = env.step(NOOP)
    assert terminated is False
    _, reward, terminated, truncated, _ = env.step(NOOP)
    assert (terminated, truncated) == (True, False)
    assert env.game.death_reason == "time"
    assert reward == pytest.approx(-STEP_PENALTY - DEATH_PENALTY)


def test_bandera_termina_y_suma_50():
    level = make_level(["#" * 20, "#" * 20], start_col=2, flag_col=8)
    env = stub_env(level=level)
    env.reset()
    total, terminated = 0.0, False
    while not terminated:
        x0 = env.game.mario.x
        _, reward, terminated, truncated, _ = env.step(RIGHT_RUN)
        total += reward
        assert truncated is False
    assert env.game.won
    dx = env.game.mario.x - x0
    assert reward == pytest.approx(dx / 10 - STEP_PENALTY + FLAG_REWARD)
    assert total > FLAG_REWARD


def test_max_steps_trunca_aunque_avance():
    env = stub_env(max_steps=5)
    env.reset()
    for _ in range(4):
        _, _, _, truncated, _ = env.step(RIGHT)
        assert truncated is False
    _, _, terminated, truncated, _ = env.step(RIGHT)
    assert (terminated, truncated) == (False, True)


def test_reset_reinicia_los_contadores():
    env = stub_env(max_steps=5)
    env.reset()
    for _ in range(5):
        env.step(RIGHT)
    env.reset()
    _, _, _, truncated, _ = env.step(RIGHT)
    assert truncated is False


def test_retroceder_resta_recompensa():
    # Motor real: el stub no se mueve a la izquierda.
    env = MarioEnv(game_factory=lambda: MarioGame(make_level(FLAT, start_col=20)))
    env.reset()
    _, reward, *_ = env.step(LEFT)
    assert env.game.mario.vx < 0
    assert reward < -STEP_PENALTY


def test_atascado_contra_una_tuberia_trunca():
    # Motor real: caminando sin saltar contra una tubería de 3, Mario no avanza.
    rows = ["..........P" + "." * 29] * 3 + FLAT
    env = MarioEnv(game_factory=lambda: MarioGame(make_level(rows, start_col=3)))
    env.reset()
    for _ in range(STUCK_STEPS + 50):
        _, _, terminated, truncated, _ = env.step(RIGHT)
        if terminated or truncated:
            break
    assert (terminated, truncated) == (False, True)
    assert env.game.mario.right <= 10 * TILE    # sigue pegado a la tubería
