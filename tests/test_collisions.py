"""P1.4: Body y colisión por ejes contra tiles sólidos.

Criterio del plan: un cuerpo no atraviesa paredes, suelo ni techo.
Coordenadas de referencia: con el suelo en las filas 13-14, la cara superior
del suelo está en y = 13 * 16 = 208.
"""

import pytest

from helpers import make_level, overlaps_solid
from mario_game.constants import (GRAVITY, MARIO_H, MARIO_W, MAX_FALL_SPEED,
                                  ROWS, RUN_MAX_SPEED, TILE)
from mario_game.game import Body, Mario, to_cell

FLOOR_Y = 13 * TILE   # 208


def fall(body, level, frames):
    """Aplica gravedad simple y mueve en y; comprueba que nunca se solapa."""
    for _ in range(frames):
        body.vy = min(body.vy + GRAVITY, MAX_FALL_SPEED)
        body.move_y(level)
        assert not overlaps_solid(body, level)


def push_x(body, level, vx, frames):
    """Empuja horizontalmente con velocidad constante; devuelve si hubo choque."""
    hit = False
    for _ in range(frames):
        body.vx = vx
        hit |= body.move_x(level)
        assert not overlaps_solid(body, level)
    return hit


# --- Representación (C.1) ---------------------------------------------------

def test_body_edges():
    b = Body(10, 20, 12, 16)
    assert (b.x, b.y, b.right, b.bottom) == (10.0, 20.0, 22.0, 36.0)
    assert (b.vx, b.vy, b.on_ground) == (0.0, 0.0, False)


def test_overlap_requires_shared_area():
    a = Body(0, 0, 16, 16)
    assert a.overlaps(Body(8, 8, 16, 16))
    assert not a.overlaps(Body(16, 0, 16, 16))   # solo se tocan en el borde
    assert not a.overlaps(Body(0, 16, 16, 16))


def test_slots_reject_typos():
    b = Body(0, 0, 16, 16)
    with pytest.raises(AttributeError):
        b.on_gruond = True


def test_mario_uses_contract_box():
    m = Mario(0, 0)
    assert (m.w, m.h) == (MARIO_W, MARIO_H)
    assert m.facing == 1 and m.skidding is False


def test_to_cell_floors_negative_values():
    assert to_cell(0) == 0 and to_cell(15.9) == 0 and to_cell(16) == 1
    assert to_cell(-0.5) == -1


# --- Suelo -------------------------------------------------------------------

def test_lands_exactly_on_floor():
    level = make_level(["##########"] * 2)
    b = Body(20, 50, MARIO_W, MARIO_H)
    fall(b, level, 120)
    assert b.bottom == FLOOR_Y
    assert b.on_ground and b.vy == 0


def test_falls_through_pit_and_leaves_map():
    level = make_level(["###...####"] * 2)
    b = Body(3 * TILE + 2, FLOOR_Y - MARIO_H, MARIO_W, MARIO_H)  # sobre el hueco
    fall(b, level, 120)
    assert b.y > ROWS * TILE      # por debajo del mapa es vacío
    assert not b.on_ground


# --- Paredes -----------------------------------------------------------------

LEVEL_WITH_PIPES = [
    "..P...P...",
    "..P...P...",
    "##########",
    "##########",
]


def test_stops_at_wall_on_the_right():
    level = make_level(LEVEL_WITH_PIPES)
    b = Body(3 * TILE, FLOOR_Y - MARIO_H, MARIO_W, MARIO_H)
    assert push_x(b, level, RUN_MAX_SPEED, 60)
    assert b.right == 6 * TILE and b.vx == 0


def test_stops_at_wall_on_the_left():
    level = make_level(LEVEL_WITH_PIPES)
    b = Body(5 * TILE, FLOOR_Y - MARIO_H, MARIO_W, MARIO_H)
    assert push_x(b, level, -RUN_MAX_SPEED, 60)
    assert b.x == 3 * TILE and b.vx == 0


def test_left_edge_of_map_is_solid():
    level = make_level(["##########"] * 2)
    b = Body(4, FLOOR_Y - MARIO_H, MARIO_W, MARIO_H)
    push_x(b, level, -RUN_MAX_SPEED, 10)
    assert b.x == 0


def test_touching_a_wall_does_not_block_falling():
    # Caja pegada a la tubería (right == 96) cayendo por su lado: no se engancha.
    level = make_level(LEVEL_WITH_PIPES)
    b = Body(6 * TILE - MARIO_W, 40, MARIO_W, MARIO_H)
    fall(b, level, 120)
    assert b.bottom == FLOOR_Y and b.right == 6 * TILE


# --- Techo -------------------------------------------------------------------

def test_head_bump_stops_at_ceiling_and_reports_cell():
    level = make_level(["..B.......", "..........", "..........",
                        "..........", "##########", "##########"])
    brick_row = ROWS - 6                                  # fila 9
    b = Body(2 * TILE + 2, FLOOR_Y - MARIO_H, MARIO_W, MARIO_H)
    b.vy = -4.5
    hits = []
    for _ in range(20):
        hits = b.move_y(level) or hits
        assert not overlaps_solid(b, level)
        if b.vy == 0:
            break
        b.vy = -4.5                                       # sigue subiendo
    assert b.y == (brick_row + 1) * TILE
    assert hits == [(brick_row, 2)]


def test_head_bump_under_two_blocks_reports_both():
    level = make_level([".BB.......", ".........."] + ["##########"] * 2)
    row = ROWS - 4
    b = Body(TILE + 10, (row + 1) * TILE + 1, MARIO_W, MARIO_H)  # entre col 1 y 2
    b.vy = -4.5
    assert b.move_y(level) == [(row, 1), (row, 2)]


# --- Invariante general -----------------------------------------------------

@pytest.mark.parametrize("vx", [-2.5, -1.5, -0.7, 0.0, 0.7, 1.5, 2.5])
@pytest.mark.parametrize("start_col", [1, 4, 8])
def test_never_ends_a_frame_inside_a_solid(vx, start_col):
    level = make_level([
        "....BBB.....",
        "............",
        ".....H......",
        "..P..H...P..",
        "..P..HH..P..",
        "#####..#####",
        "#####..#####",
    ])
    b = Body(start_col * TILE + 2, 10 * TILE, MARIO_W, MARIO_H)
    for frame in range(300):
        if frame % 40 == 0:
            b.vy = -4.5                   # "salta" periódicamente
        b.vy = min(b.vy + GRAVITY, MAX_FALL_SPEED)
        b.vx = vx
        b.move_x(level)
        b.move_y(level)
        assert not overlaps_solid(b, level), (frame, b.x, b.y)
        if b.y > ROWS * TILE:
            break
