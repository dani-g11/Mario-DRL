"""Tests del diccionario `info` y del registro MarioPygame-v0 (P3.6)."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from helpers import make_level
from mario_env import ENV_ID, MarioEnv
from mario_stub import StubMarioGame

# Claves obligatorias del contrato (INTERFACES.md §7) y su tipo.
CONTRACT_KEYS = {"x_pos": float, "progress": float, "flag_get": bool, "dead": bool}


def assert_contract_info(info):
    for key, kind in CONTRACT_KEYS.items():
        assert isinstance(info[key], kind), key
    assert info["death_reason"] in (None, "pit", "goomba", "time")


def test_reset_y_step_traen_las_claves_del_contrato():
    env = MarioEnv(game_factory=StubMarioGame)
    _, info = env.reset()
    assert_contract_info(info)
    assert info["dead"] is False and info["flag_get"] is False
    assert info["death_reason"] is None
    x0 = info["x_pos"]

    _, _, _, _, info = env.step(1)
    assert_contract_info(info)
    assert info["x_pos"] == pytest.approx(x0 + 6)
    assert info["x_pos"] == env.game.mario.x and info["progress"] == env.game.progress
    assert info["steps"] == 1


def test_info_al_ganar():
    level = make_level(["#" * 20, "#" * 20], start_col=2, flag_col=8)
    env = MarioEnv(game_factory=lambda: StubMarioGame(level=level))
    env.reset()
    terminated = False
    while not terminated:
        _, _, terminated, _, info = env.step(3)
    assert info["flag_get"] is True and info["dead"] is False
    assert info["progress"] == 1.0
    assert info["death_reason"] is None


def test_info_al_morir_por_tiempo():
    env = MarioEnv(game_factory=lambda: StubMarioGame(time_limit=4))
    env.reset()
    _, _, terminated, _, info = env.step(0)
    assert terminated
    assert info["dead"] is True and info["flag_get"] is False
    assert info["death_reason"] == "time"
    assert info["time_left"] == 0


def test_entorno_registrado():
    assert ENV_ID == "MarioPygame-v0"
    assert ENV_ID in gym.registry


def test_gym_make_funciona():
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)
    assert isinstance(env.unwrapped, MarioEnv)
    assert env.observation_space.contains(obs)
    assert_contract_info(info)
    obs, reward, terminated, truncated, info = env.step(1)
    assert env.observation_space.contains(obs)
    assert_contract_info(info)
    env.close()


def test_gym_make_pasa_los_argumentos():
    env = gym.make(ENV_ID, obs_type="pixels", max_steps=7)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84, 1) and obs.dtype == np.uint8
    assert env.unwrapped.max_steps == 7
    env.close()


def test_check_env_a_traves_de_gym_make():
    # Con spec (creado con gym.make), check_env prueba también los modos de render.
    check_env(gym.make(ENV_ID).unwrapped)
