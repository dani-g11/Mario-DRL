"""Tests del entorno MarioEnv (P3.2 en adelante).

Los tests que dependen de números exactos pasan el stub explícitamente
(game_factory=StubMarioGame): en P3.8 la fábrica por defecto pasa al motor real.
"""

import numpy as np
from gymnasium.utils.env_checker import check_env

from mario_env import ACTION_NAMES, ACTIONS, MarioEnv
from mario_game.constants import WALK_MAX_SPEED
from mario_stub import StubMarioGame


def test_check_env_pasa_con_stub():
    check_env(MarioEnv(game_factory=StubMarioGame))


def test_siete_acciones_y_nombres():
    assert len(ACTIONS) == 7
    assert len(ACTION_NAMES) == 7
    # La traducción de cada acción coincide con la tabla de §7.
    esperado = [
        (False, False, False, False),
        (False, True, False, False),
        (False, True, True, False),
        (False, True, False, True),
        (False, True, True, True),
        (False, False, True, False),
        (True, False, False, False),
    ]
    assert ACTIONS == esperado


def test_step_avanza_frame_skip():
    env = MarioEnv(game_factory=StubMarioGame)
    env.reset()
    x0 = env.game.mario.x
    env.step(1)                      # acción → mantenida 4 frames
    assert env.game.mario.x - x0 == 4 * WALK_MAX_SPEED
    assert env.game.frame == 4


def test_frame_skip_se_corta_al_terminar():
    # El stub muere por tiempo en el frame 2: el paso no debe simular los otros 2.
    env = MarioEnv(game_factory=lambda: StubMarioGame(time_limit=2))
    env.reset()
    _, _, terminated, truncated, _ = env.step(1)
    assert terminated is True
    assert truncated is False
    assert env.game.frame == 2


def test_acepta_acciones_numpy():
    # Los agentes (Stable-Baselines3) pasan numpy.int64
    env = MarioEnv(game_factory=StubMarioGame)
    env.reset()
    x0 = env.game.mario.x
    env.step(np.int64(3))            # → + correr
    assert env.game.mario.x > x0
