"""Tests de la aleatoriedad para robustez (P3.9): acciones pegajosas y
velocidad variable de los goombas."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from mario_env import ENV_ID, MarioEnv
from mario_game.constants import GOOMBA_SPEED
from mario_stub import StubMarioGame

# Una secuencia que mezcla acciones: con acciones pegajosas, repetir la anterior
# en algunos frames cambia la trayectoria.
ACTION_SEQ = [3, 0, 1, 4, 0, 3, 6, 2] * 25
SPEED_RANGE = (0.3, 1.5)


def trajectory(env, seed, actions=ACTION_SEQ):
    """x de Mario tras cada paso de un episodio con semilla `seed`."""
    env.reset(seed=seed)
    xs = []
    for a in actions:
        _, _, terminated, truncated, info = env.step(a)
        xs.append(info["x_pos"])
        if terminated or truncated:
            break
    return xs


def test_acciones_pegajosas_dependen_de_la_semilla():
    env = MarioEnv(game_factory=StubMarioGame, sticky_prob=0.25)
    assert trajectory(env, 1) == trajectory(env, 1)       # misma semilla, mismo episodio
    assert trajectory(env, 1) != trajectory(env, 2)       # semillas distintas, distinto


def test_dos_semillas_dan_episodios_distintos_con_el_motor_real():
    # Criterio de P3.9 con todo activado, en el nivel 1 por defecto.
    env = MarioEnv(sticky_prob=0.25, goomba_speed_range=SPEED_RANGE)
    a, b = trajectory(env, 10), trajectory(env, 11)
    assert a != b
    assert trajectory(env, 10) == a


def test_sin_aleatoriedad_la_semilla_no_importa():
    env = MarioEnv()
    assert trajectory(env, 1) == trajectory(env, 2)


def test_sin_aleatoriedad_no_se_consumen_numeros_aleatorios():
    env = MarioEnv()
    env.reset(seed=5)
    state = env.np_random.bit_generator.state
    for a in ACTION_SEQ[:50]:
        env.step(a)
    env.reset()
    assert env.np_random.bit_generator.state == state


def test_la_accion_pegajosa_empieza_en_nada():
    # Con sticky_prob alto casi todos los frames repiten la anterior, y tras
    # reset() la anterior es "nada": Mario apenas se mueve al principio.
    sticky = MarioEnv(game_factory=StubMarioGame, sticky_prob=0.9)
    normal = MarioEnv(game_factory=StubMarioGame)
    x_sticky = trajectory(sticky, 0, actions=[3])[0]
    x_normal = trajectory(normal, 0, actions=[3])[0]
    assert x_sticky < x_normal


def test_acciones_pegajosas_por_frame_mezclan_acciones_en_un_paso():
    # En un paso de 4 frames, algunos frames usan la acción anterior: con el
    # stub, la distancia de un paso queda entre la de una acción y la de la otra.
    env = MarioEnv(game_factory=StubMarioGame, sticky_prob=0.5)
    env.reset(seed=0)
    distances = set()
    for i in range(200):                     # alterna → (1.5 px/frame) y → + correr (2.5)
        x0 = env.game.mario.x
        env.step(1 if i % 2 else 3)
        distances.add(round(env.game.mario.x - x0, 3))
    assert any(6 < d < 10 for d in distances)  # pasos con frames de las dos acciones


def test_velocidad_de_goombas_en_el_rango():
    env = MarioEnv(goomba_speed_range=SPEED_RANGE)
    speeds = []
    for seed in range(20):
        _, info = env.reset(seed=seed)
        lo, hi = SPEED_RANGE
        assert lo <= info["goomba_speed"] <= hi
        # Se aplica en este mismo reset(): todos los goombas la llevan.
        assert all(g.speed == info["goomba_speed"] for g in env.game.goombas)
        speeds.append(info["goomba_speed"])
    assert len(set(speeds)) == 20


def test_velocidad_de_goombas_reproducible():
    env = MarioEnv(goomba_speed_range=SPEED_RANGE)
    _, a = env.reset(seed=7)
    _, b = env.reset(seed=7)
    assert a["goomba_speed"] == b["goomba_speed"]


def test_por_defecto_la_velocidad_es_la_del_contrato():
    env = MarioEnv()
    for seed in range(3):
        _, info = env.reset(seed=seed)
        assert info["goomba_speed"] == GOOMBA_SPEED
        assert all(g.speed == GOOMBA_SPEED for g in env.game.goombas)


def test_check_env_con_la_aleatoriedad_activada():
    check_env(MarioEnv(sticky_prob=0.25, goomba_speed_range=SPEED_RANGE))
    check_env(gym.make(ENV_ID, sticky_prob=0.25, goomba_speed_range=SPEED_RANGE,
                       random_start=True).unwrapped)


@pytest.mark.parametrize("kwargs", [
    {"sticky_prob": -0.1}, {"sticky_prob": 1.0},
    {"goomba_speed_range": (0.0, 1.0)}, {"goomba_speed_range": (1.5, 0.3)},
])
def test_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        MarioEnv(game_factory=StubMarioGame, **kwargs)


def test_info_trae_goomba_speed():
    env = MarioEnv(game_factory=StubMarioGame)
    _, info = env.reset()
    assert isinstance(info["goomba_speed"], float)
    _, _, _, _, info = env.step(np.int64(1))
    assert info["goomba_speed"] == GOOMBA_SPEED
