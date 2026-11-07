"""mario_training.py — utilidades compartidas de entrenamiento y evaluación (P4).

Reúne lo que usan `train.py` y `evaluate.py`:

- Fábricas de entornos vectorizados para Mario y para CartPole, con `Monitor`,
  semillas por entorno y apilado de frames opcional (`VecFrameStack`, P4.7).

Todo lo de pygame y SDL sin pantalla se configura aquí al importar, para que
también lo hereden los procesos hijos de SubprocVecEnv.
"""

from __future__ import annotations

import os

# SDL sin pantalla también en los procesos hijos de SubprocVecEnv: en Windows y
# macOS "spawn" vuelve a importar este módulo en cada hijo, y si pygame no
# encuentra el driver dummy el proceso muere al crear el entorno. Ponerlo aquí,
# antes de cualquier import que arrastre pygame, lo garantiza en todos lados.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecFrameStack, VecTransposeImage)

# Importar mario_env registra "MarioPygame-v0" en Gymnasium (INTERFACES.md §7).
import mario_env
from mario_env import ENV_ID  # noqa: F401  (se reexporta para train/evaluate)

__all__ = [
    "ENV_ID",
    "make_mario_env",
    "make_mario_venv",
    "make_cartpole_venv",
]


def make_mario_env(seed=None, obs_type="grid", level=1, random_start=False,
                   random_start_prob=0.5, sticky_prob=0.0,
                   goomba_speed_range=None, render_mode=None, **kwargs):
    """Crea un `MarioPygame-v0` envuelto en `Monitor` (para que SB3 registre la
    recompensa y la duración de cada episodio). Devuelve una función sin
    argumentos (lo que esperan los VecEnv de SB3)."""
    def _init():
        # Importar mario_env DENTRO de la fábrica, no solo arriba del módulo:
        # con SubprocVecEnv y "spawn" (Windows/macOS), cada proceso hijo empieza
        # limpio y re-importa el módulo __main__, que no siempre arrastra
        # mario_env. Sin este import, el hijo no tiene registrado "MarioPygame-v0"
        # y gym.make falla con NameNotFound.
        import mario_env  # noqa: F401
        env = gym.make(
            ENV_ID, obs_type=obs_type, level=level, random_start=random_start,
            random_start_prob=random_start_prob, sticky_prob=sticky_prob,
            goomba_speed_range=goomba_speed_range, render_mode=render_mode,
            **kwargs,
        )
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _init


def _make_venv(env_fns, vec, start_method=None):
    if vec == "subproc" and len(env_fns) > 1:
        # "spawn" funciona en Windows, macOS y Linux. fork daría problemas con
        # pygame/torch en algunos sistemas.
        return SubprocVecEnv(env_fns, start_method=start_method or "spawn")
    return DummyVecEnv(env_fns)


def make_mario_venv(n_envs=8, seed=0, obs_type="grid", level=1, frame_stack=1,
                    random_start=False, random_start_prob=0.5, sticky_prob=0.0,
                    goomba_speed_range=None, vec="subproc", render_mode=None):
    """Entorno vectorizado de Mario para entrenar o evaluar en lote.

    Cada subentorno recibe una semilla distinta (seed, seed+1, ...) para que no
    jueguen episodios idénticos. Con obs_type="pixels" se transponen los canales
    para PyTorch (VecTransposeImage) y, si frame_stack>1, se apilan frames
    (P4.7)."""
    env_fns = [
        make_mario_env(
            seed=(None if seed is None else seed + i), obs_type=obs_type,
            level=level, random_start=random_start,
            random_start_prob=random_start_prob, sticky_prob=sticky_prob,
            goomba_speed_range=goomba_speed_range, render_mode=render_mode,
        )
        for i in range(n_envs)
    ]
    venv = _make_venv(env_fns, vec)
    if obs_type == "pixels":
        # (H, W, C) -> (C, H, W) para las CNN de SB3.
        venv = VecTransposeImage(venv)
        if frame_stack > 1:
            venv = VecFrameStack(venv, n_stack=frame_stack, channels_order="first")
    elif frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    return venv


def make_cartpole_venv(n_envs=8, seed=0, vec="subproc"):
    """CartPole-v1 vectorizado, para probar el pipeline de PPO (P4.1)."""
    def _init(rank):
        def _f():
            env = gym.make("CartPole-v1")
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _f

    return _make_venv([_init(i) for i in range(n_envs)], vec)
