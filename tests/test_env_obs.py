"""Tests de la observación `grid` del entorno (P3.3).

Usan el motor real (ya está en main) con niveles mínimos de tests/helpers.py,
para comprobar que la rejilla refleja lo que hay de verdad alrededor de Mario.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from helpers import make_level
from mario_env import (ENEMIES_OFFSET, ENEMY_FEATURES, GRID_CELLS,
                       GRID_COLS, GRID_COLS_BEHIND, GRID_OBS_DIM, STATE_OFFSET,
                       MarioEnv)
from mario_game.constants import ROWS, SCREEN_W
from mario_game.game import MarioGame
from mario_game.level import build_level_1
from mario_stub import StubMarioGame

# Índices del bloque de estado (ver el docstring de mario_env).
Y, VX, VY, ON_GROUND, FRAC_X, JUMP_HELD = range(STATE_OFFSET, STATE_OFFSET + 6)

FLAT = ["#" * 40, "#" * 40]


def env_for(rows, **level_kwargs):
    """Entorno grid con el motor real sobre un nivel ASCII mínimo."""
    return MarioEnv(game_factory=lambda: MarioGame(make_level(rows, **level_kwargs)))


def solid_channel(obs):
    return obs[:GRID_CELLS].reshape(ROWS, GRID_COLS)


def enemy_channel(obs):
    return obs[GRID_CELLS:STATE_OFFSET].reshape(ROWS, GRID_COLS)


def test_forma_tipo_y_rango_en_el_nivel_1():
    env = MarioEnv(game_factory=lambda: MarioGame(build_level_1()))
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (GRID_OBS_DIM,) == (498,)
    for _ in range(2000):        # varias partidas: carreras, saltos, muertes
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(7)))
        if terminated or truncated:
            obs, _ = env.reset()


def test_tuberia_delante_aparece_en_el_canal_de_solidos():
    # Mario empieza en la columna 3 y la tubería (2 de alto) está en la 8: 5 por delante.
    rows = ["........P" + "." * 31,
            "........P" + "." * 31] + FLAT
    env = env_for(rows, start_col=3)
    obs, _ = env.reset()
    solid = solid_channel(obs)
    col = GRID_COLS_BEHIND + 5
    assert solid[11, col] == 1 and solid[12, col] == 1      # la tubería
    assert solid[:11, col].sum() == 0                        # aire encima
    assert (solid[13:, :] == 1).all()                        # el suelo
    assert solid[11:13, col + 1].sum() == 0                  # la tubería es de 1 columna


def test_fuera_del_mapa_por_la_izquierda_es_solido():
    env = env_for(FLAT, start_col=0)
    obs, _ = env.reset()
    solid = solid_channel(obs)
    assert (solid[:, :GRID_COLS_BEHIND] == 1).all()
    assert solid[:13, GRID_COLS_BEHIND].sum() == 0           # la columna de Mario: aire


def test_goomba_activo_aparece_y_aplastado_desaparece():
    env = env_for(FLAT, start_col=3, goomba_cols=[10])
    obs, _ = env.reset()
    # Antes del primer frame el goomba aún no está activo: no se ve.
    assert enemy_channel(obs).sum() == 0
    assert obs[ENEMIES_OFFSET + ENEMY_FEATURES - 1] == 0

    obs, *_ = env.step(0)            # se activa al entrar en la vista
    enemy = enemy_channel(obs)
    assert enemy.sum() == 1
    assert enemy[12, 10 - 3 + GRID_COLS_BEHIND] == 1         # fila del suelo, 7 por delante
    dx, dy, direction, present = obs[ENEMIES_OFFSET:ENEMIES_OFFSET + ENEMY_FEATURES]
    assert present == 1 and dx > 0 and direction == -1 and dy == pytest.approx(0)

    # Un goomba aplastado no cuenta (INTERFACES.md §4). Se marca a mano para no
    # depender de la puntería del pisotón.
    env.game.goombas[0].squashed = True
    obs, *_ = env.step(0)
    assert enemy_channel(obs).sum() == 0
    assert obs[ENEMIES_OFFSET + ENEMY_FEATURES - 1] == 0


def test_tres_goombas_mas_cercanos_ordenados():
    # 4 goombas dentro de la vista (se activan hasta la columna 16 con cam_x = 0):
    # la observación guarda los 3 más cercanos, del más cercano al más lejano.
    env = env_for(FLAT, start_col=3, goomba_cols=[16, 8, 14, 11])
    env.reset()
    obs, *_ = env.step(0)
    assert len([g for g in env.game.goombas if g.active]) == 4
    dxs = [obs[ENEMIES_OFFSET + i * ENEMY_FEATURES] for i in range(3)]
    presents = [obs[ENEMIES_OFFSET + i * ENEMY_FEATURES + 3] for i in range(3)]
    assert presents == [1, 1, 1]
    mario_x = env.game.mario.x
    nearest = sorted(g.x for g in env.game.goombas)[:3]
    assert dxs == pytest.approx([(x - mario_x) / SCREEN_W for x in nearest])


def test_variables_de_estado():
    env = env_for(FLAT, start_col=3)
    obs, _ = env.reset()
    assert obs[ON_GROUND] == 1 and obs[VX] == 0 and obs[JUMP_HELD] == 0
    assert 0 < obs[Y] < 1 and 0 <= obs[FRAC_X] < 1

    obs, *_ = env.step(5)            # saltar
    assert obs[JUMP_HELD] == 1
    assert obs[ON_GROUND] == 0 and obs[VY] < 0
    obs, *_ = env.step(0)            # soltar
    assert obs[JUMP_HELD] == 0

    env.reset()
    obs, *_ = env.step(3)            # → + correr
    assert obs[VX] > 0


def test_check_env_con_motor_real_y_con_stub():
    check_env(MarioEnv(game_factory=lambda: MarioGame(build_level_1())))
    check_env(MarioEnv(game_factory=StubMarioGame))


def test_obs_type_desconocido():
    with pytest.raises(ValueError):
        MarioEnv(game_factory=StubMarioGame, obs_type="ram")
