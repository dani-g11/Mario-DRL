"""Tests de las utilidades de entrenamiento y evaluación (Persona 4).

Comprueban lo que no necesita horas de entrenamiento:

- Las fábricas de entornos (make_mario_venv) devuelven VecEnv con la forma de
  observación correcta para grid y para pixels (con y sin apilado de frames).
- El callback de métricas (P4.2) calcula progress_mean, flag_rate y el reparto
  de muertes a partir del `info` del contrato, usando un entorno falso (como
  pide P4.2: "las métricas aparecen con un entorno falso").

No se importa stable_baselines3 si no está instalado: estos tests se saltan.
Así la suite del motor y del entorno (Personas 1-3) sigue pasando sin torch.
"""

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3",
                          reason="stable-baselines3 no instalado (dependencia de P4)")

from mario_training import (ENV_ID, MarioMetricsCallback, make_cartpole_venv,
                            make_mario_venv)
from mario_env import GRID_OBS_DIM, PIXELS_SIZE


def test_env_id_registrado():
    import gymnasium as gym
    assert ENV_ID in gym.registry


def test_venv_grid_forma():
    venv = make_mario_venv(n_envs=2, seed=0, obs_type="grid", vec="dummy")
    try:
        obs = venv.reset()
        assert obs.shape == (2, GRID_OBS_DIM)
        assert obs.dtype == np.float32
    finally:
        venv.close()


def test_venv_pixels_transpuesto_y_apilado():
    # pixels pasa por VecTransposeImage (canales primero) y VecFrameStack.
    venv = make_mario_venv(n_envs=1, seed=0, obs_type="pixels", frame_stack=4,
                           vec="dummy")
    try:
        obs = venv.reset()
        # 4 frames apilados en el eje de canales: (n_env, 4, 84, 84).
        assert obs.shape == (1, 4, PIXELS_SIZE, PIXELS_SIZE)
        assert obs.dtype == np.uint8
    finally:
        venv.close()


def test_cartpole_venv_funciona():
    venv = make_cartpole_venv(n_envs=2, seed=0, vec="dummy")
    try:
        obs = venv.reset()
        assert obs.shape[0] == 2
    finally:
        venv.close()


class _FakeLogger:
    """Logger de mentira que guarda lo último que se registró por clave."""

    def __init__(self):
        self.values = {}

    def record(self, key, value):
        self.values[key] = value


class _FakeModel:
    """Modelo falso: el callback de SB3 lee su logger (`logger` es una propiedad
    de BaseCallback que delega en `self.model.logger`)."""

    def __init__(self, logger):
        self.logger = logger


def _make_callback():
    """Un callback con un modelo y un logger falsos (entorno falso de P4.2)."""
    cb = MarioMetricsCallback(window=100)
    cb.model = _FakeModel(_FakeLogger())
    return cb


def _run_callback(episodes):
    """Pasa una lista de episodios (dicts de info) por el callback y devuelve lo
    que registró. Cada episodio se marca como done en un paso."""
    cb = _make_callback()
    for info in episodes:
        cb.locals = {"dones": [True], "infos": [info]}
        cb._on_step()
    return cb.model.logger.values


def test_callback_progress_y_flag_rate():
    # Entorno falso (P4.2): 3 episodios con progreso y banderas conocidos.
    episodes = [
        {"progress": 1.0, "flag_get": True, "death_reason": None},
        {"progress": 0.5, "flag_get": False, "death_reason": "goomba"},
        {"progress": 0.3, "flag_get": False, "death_reason": "pit"},
    ]
    values = _run_callback(episodes)
    assert values["mario/progress_mean"] == pytest.approx((1.0 + 0.5 + 0.3) / 3)
    assert values["mario/flag_rate"] == pytest.approx(1 / 3)
    assert values["mario/death_goomba"] == pytest.approx(1 / 3)
    assert values["mario/death_pit"] == pytest.approx(1 / 3)
    assert values["mario/death_time"] == pytest.approx(0.0)


def test_callback_ignora_pasos_sin_done():
    cb = _make_callback()
    # Ningún done: no registra nada (no hay episodios terminados todavía).
    cb.locals = {"dones": [False, False], "infos": [{}, {}]}
    cb._on_step()
    assert cb.model.logger.values == {}


def test_callback_todas_banderas():
    episodes = [{"progress": 1.0, "flag_get": True, "death_reason": None}] * 5
    values = _run_callback(episodes)
    assert values["mario/flag_rate"] == pytest.approx(1.0)
    assert values["mario/progress_mean"] == pytest.approx(1.0)
    for reason in ("pit", "goomba", "time"):
        assert values[f"mario/death_{reason}"] == pytest.approx(0.0)
