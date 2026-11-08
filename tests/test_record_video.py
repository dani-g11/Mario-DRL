"""Tests de las opciones de aleatoriedad de scripts/record_video.py (P4.9).

Solo comprueban cómo se construye el entorno a partir de la línea de comandos;
grabar un vídeo de verdad necesita un modelo entrenado.
"""

import argparse
import importlib.util
from pathlib import Path

import gymnasium as gym
import pytest

import mario_env  # noqa: F401  registra MarioPygame-v0

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "record_video.py"
_spec = importlib.util.spec_from_file_location("record_video", _PATH)
record_video = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(record_video)


def _args(**overrides):
    base = dict(obs="grid", level=1, sticky_prob=0.0, goomba_speed_min=None,
                goomba_speed_max=None, random_start=False, random_start_prob=0.5)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_sin_opciones_graba_el_entorno_determinista():
    assert record_video.env_kwargs(_args()) == {
        "obs_type": "grid", "level": 1, "render_mode": "rgb_array"}


def test_pasa_las_opciones_de_aleatoriedad():
    kwargs = record_video.env_kwargs(_args(
        level=2, sticky_prob=0.1, goomba_speed_min=0.3, goomba_speed_max=1.5,
        random_start=True, random_start_prob=0.7))
    assert kwargs["level"] == 2
    assert kwargs["sticky_prob"] == 0.1
    assert kwargs["goomba_speed_range"] == (0.3, 1.5)
    assert kwargs["random_start"] is True
    assert kwargs["random_start_prob"] == 0.7


@pytest.mark.parametrize("lo, hi", [(0.3, None), (None, 1.5)])
def test_velocidad_de_goomba_exige_minimo_y_maximo(lo, hi):
    with pytest.raises(SystemExit):
        record_video.env_kwargs(_args(goomba_speed_min=lo, goomba_speed_max=hi))


def test_el_entorno_sortea_la_velocidad_de_goomba():
    kwargs = record_video.env_kwargs(_args(goomba_speed_min=0.3, goomba_speed_max=1.5))
    env = gym.make(mario_env.ENV_ID, **kwargs)
    speeds = {round(env.reset(seed=s)[1]["goomba_speed"], 6) for s in range(5)}
    env.close()
    assert all(0.3 <= v <= 1.5 for v in speeds)
    assert len(speeds) > 1      # cambia entre episodios
