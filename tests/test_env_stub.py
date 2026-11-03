"""Tests del stub del juego (P3.1).

El stub (mario_stub.StubMarioGame) debe cumplir el contrato del §8 de
INTERFACES.md: mismos métodos, firmas y campos públicos, y avanzar en línea
recta a velocidad constante.
"""

import pygame
import pytest

from mario_game.constants import (RUN_MAX_SPEED, SCREEN_H, SCREEN_W, TILE,
                                   WALK_MAX_SPEED)
from helpers import make_level
from mario_stub import StubMarioGame


def test_reset_deja_estado_limpio():
    g = StubMarioGame()
    g.step(False, True, False, False)   # ensucia algo de estado
    g.reset()
    assert g.dead is False
    assert g.won is False
    assert g.death_reason is None
    assert g.mario.on_ground is True
    # Mario en start_col, centrado, con los pies en el suelo.
    col = g.level.start_col
    assert g.mario.x == pytest.approx(col * TILE + (TILE - g.mario.w) / 2)


def test_walk_y_run_avanzan_a_velocidad_constante():
    g = StubMarioGame()
    x0 = g.mario.x
    g.step(False, True, False, False)
    assert g.mario.x - x0 == pytest.approx(WALK_MAX_SPEED)
    assert g.mario.vx == pytest.approx(WALK_MAX_SPEED)

    g.reset()
    x0 = g.mario.x
    g.step(False, True, False, True)
    assert g.mario.x - x0 == pytest.approx(RUN_MAX_SPEED)
    assert g.mario.vx == pytest.approx(RUN_MAX_SPEED)


def test_progress_y_cam_x_no_disminuyen():
    g = StubMarioGame()
    prev_progress, prev_cam = g.progress, g.cam_x
    for _ in range(300):
        g.step(False, True, False, True)
        assert g.progress >= prev_progress
        assert g.cam_x >= prev_cam
        prev_progress, prev_cam = g.progress, g.cam_x


def test_start_col_lejano_da_progress_positivo():
    g = StubMarioGame()
    lejano = g.level.checkpoints[-1]
    g.reset(start_col=lejano)
    assert g.progress > 0
    assert g.cam_x > 0


def test_llega_a_la_bandera():
    level = make_level(["#" * 20, "#" * 20], start_col=2, flag_col=8)
    g = StubMarioGame(level=level)
    for _ in range(100):
        g.step(False, True, False, True)
        if g.won:
            break
    assert g.won is True
    assert g.mario.right > level.flag_col * TILE


def test_muerte_por_tiempo():
    g = StubMarioGame(time_limit=10)
    for _ in range(10):
        g.step(False, False, False, False)
    assert g.dead is True
    assert g.death_reason == "time"


def test_tras_terminar_step_no_cambia_nada():
    level = make_level(["#" * 20, "#" * 20], start_col=2, flag_col=8)
    g = StubMarioGame(level=level)
    while not g.won:
        g.step(False, True, False, True)
    snapshot = (g.mario.x, g.frame, g.progress, g.cam_x)
    g.step(False, True, False, True)
    assert (g.mario.x, g.frame, g.progress, g.cam_x) == snapshot


def test_render_no_lanza():
    g = StubMarioGame()
    surface = pygame.Surface((SCREEN_W, SCREEN_H))
    g.render(surface)          # no debe lanzar
