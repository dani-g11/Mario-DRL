"""P1.8: goombas — activación al acercarse, patrulla, giro en paredes y
rebote entre ellos.

Criterio del plan: un goomba gira al chocar con una tubería.
Mario se queda quieto en todos estos tests (la interacción Mario-goomba es C.7).
"""

from helpers import make_level, overlaps_solid
from mario_game.constants import (GOOMBA_H, GOOMBA_SPEED, ROWS, SCREEN_W,
                                  TILE)
from mario_game.game import MarioGame

FLOOR_Y = 13 * TILE


def new_game(rows, goomba_cols, start_col=1, **kwargs):
    level = make_level(rows, goomba_cols=goomba_cols, start_col=start_col)
    return MarioGame(level, **kwargs)


def idle(game, frames):
    """Mario no pulsa nada; comprueba que ningún goomba entra en un sólido."""
    for _ in range(frames):
        game.step(False, False, False, False)
        for g in game.goombas:
            assert not overlaps_solid(g, game.level)


# --- Aparición ---------------------------------------------------------------

def test_reset_spawns_goombas_on_surface_of_their_columns():
    game = new_game(["#" * 30] * 2, goomba_cols=[10, 20])
    assert [g.x for g in game.goombas] == [10 * TILE, 20 * TILE]
    for g in game.goombas:
        assert g.bottom == FLOOR_Y and g.h == GOOMBA_H
        assert g.direction == -1 and g.speed == GOOMBA_SPEED
        assert not g.active and not g.squashed


def test_goombas_too_close_to_start_are_not_spawned():
    game = new_game(["#" * 30] * 2, goomba_cols=[3, 4, 5, 6], start_col=1)
    assert [g.x // TILE for g in game.goombas] == [5, 6]   # cols > 1 + 3


def test_goomba_speed_is_applied_on_next_reset():
    game = new_game(["#" * 30] * 2, goomba_cols=[10], goomba_speed=0.8)
    assert game.goombas[0].speed == 0.8
    game.goomba_speed = 0.3
    game.reset()
    assert game.goombas[0].speed == 0.3


# --- Activación --------------------------------------------------------------

def test_far_goomba_waits_until_camera_gets_close():
    game = new_game(["#" * 100] * 2, goomba_cols=[60])
    g = game.goombas[0]
    x0 = g.x
    idle(game, 120)
    assert not g.active and g.x == x0             # lejos: no se mueve

    for _ in range(600):                           # Mario avanza
        game.step(False, True, False, True)
        if g.active:
            break
    assert g.active
    assert g.x < game.cam_x + SCREEN_W + TILE      # umbral del contrato


def test_activation_threshold_is_one_tile_past_the_screen():
    # Con cam_x = 0, la zona de activación termina en x = 256 + 16 = 272.
    game = new_game(["#" * 40] * 2, goomba_cols=[16, 17, 18])
    game.step(False, False, False, False)
    near, edge, far = game.goombas
    assert near.active                             # x = 256 < 272
    assert not edge.active                         # x = 272, no es < 272
    assert not far.active


def test_active_goomba_patrols_at_its_speed():
    game = new_game(["#" * 30] * 2, goomba_cols=[10])
    g = game.goombas[0]
    idle(game, 1)
    x = g.x
    idle(game, 10)
    assert g.x == x - 10 * GOOMBA_SPEED and g.on_ground


# --- Paredes (criterio del plan) ---------------------------------------------

def test_goomba_turns_when_hitting_a_pipe():
    rows = ["......P" + "." * 13,
            "......P" + "." * 13] + ["#" * 20] * 2
    game = new_game(rows, goomba_cols=[12], start_col=1)
    g = game.goombas[0]
    turned_at = None
    for frame in range(200):
        idle(game, 1)
        if g.direction == 1 and turned_at is None:
            turned_at = frame
            assert g.x == 7 * TILE                 # pegado a la cara derecha de la tubería
    assert turned_at is not None
    assert g.x > 7 * TILE                          # ya se aleja de la tubería


def test_goomba_turns_at_walls_on_both_sides():
    # Mario (col 1) queda fuera del recinto entre las tuberías de las cols 3 y 12.
    rows = ["...P" + "." * 8 + "P...",
            "...P" + "." * 8 + "P..."] + ["#" * 16] * 2
    game = new_game(rows, goomba_cols=[8], start_col=1)
    g = game.goombas[0]
    directions = []
    for _ in range(600):
        idle(game, 1)
        if not directions or directions[-1] != g.direction:
            directions.append(g.direction)
    assert not game.dead
    assert directions[:3] == [-1, 1, -1]           # rebota de pared a pared
    assert 4 * TILE <= g.x <= 12 * TILE - g.w


# --- Rebote entre goombas ----------------------------------------------------

def test_two_goombas_bounce_off_each_other():
    game = new_game(["#" * 40] * 2, goomba_cols=[10, 13])
    a, b = game.goombas
    b.direction = -1
    a.direction = 1                                # van uno hacia el otro
    for _ in range(200):
        idle(game, 1)
        assert not a.overlaps(b) or (a.direction, b.direction) == (-1, 1)
        if (a.direction, b.direction) == (-1, 1):
            break
    assert (a.direction, b.direction) == (-1, 1)   # se alejan
    idle(game, 30)
    assert not a.overlaps(b) and a.x < b.x          # no se quedan atascados


def test_deeply_overlapping_goombas_separate_instead_of_jittering():
    # Caso real: uno cae encima de otro desde un escalón o tienen velocidades
    # distintas. Invertir la dirección en cada frame los dejaría temblando.
    game = new_game(["#" * 40] * 2, goomba_cols=[10, 11])
    a, b = game.goombas
    b.x = a.x + 4                                  # 12 px de solapamiento
    for _ in range(60):
        idle(game, 1)
    assert not a.overlaps(b)
    assert a.direction == -1 and b.direction == 1


# --- Caídas ------------------------------------------------------------------

def test_goomba_walks_off_a_ledge_and_lands_below():
    rows = ["......" + "#" * 10,
            "......" + "#" * 10] + ["#" * 16] * 2      # escalón de 2 tiles
    game = new_game(rows, goomba_cols=[8], start_col=1)
    g = game.goombas[0]
    idle(game, 200)
    assert g.on_ground and g.bottom == FLOOR_Y and g.x < 6 * TILE


def test_goomba_falls_into_pit_and_is_removed():
    rows = ["#" * 8 + "..." + "#" * 9] * 2
    game = new_game(rows, goomba_cols=[12], start_col=1)
    idle(game, 400)
    assert game.goombas == []


def test_inactive_goombas_do_not_fall():
    # Un goomba fuera de pantalla se queda congelado en su sitio.
    game = new_game(["#" * 100] * 2, goomba_cols=[80])
    y = game.goombas[0].y
    idle(game, 50)
    assert game.goombas[0].y == y and game.goombas[0].y < ROWS * TILE
