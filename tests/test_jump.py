"""P1.6: salto por flanco, gravedad variable y velocidad máxima de caída.

Criterio del plan: salto de al menos 4 tiles; no salta de nuevo sin soltar.
Todos los tests usan el motor real (MarioGame.step), no el modelo en papel.
Suelo en las filas 13-14: los pies de Mario en reposo están en y = 208.
"""

from helpers import make_level
from mario_game.constants import (MAX_FALL_SPEED, MAX_PIT_TILES,
                                  MIN_JUMP_TILES, ROWS, TILE)
from mario_game.game import MarioGame

FLOOR_Y = 13 * TILE
FLAT = ["#" * 60] * 2


def new_game(rows=FLAT, start_col=5):
    return MarioGame(make_level(rows, start_col=start_col))


def jump_rise(game, hold_frames):
    """Pulsa salto `hold_frames` frames y devuelve cuánto subieron los pies (px)."""
    start = game.mario.bottom
    lowest_bottom = start
    for frame in range(200):
        game.step(False, False, frame < hold_frames, False)
        lowest_bottom = min(lowest_bottom, game.mario.bottom)
        if frame > 0 and game.mario.on_ground:
            break
    return start - lowest_bottom


# --- Altura (criterio del plan) ---------------------------------------------

def test_full_jump_rises_at_least_4_tiles():
    rise = jump_rise(new_game(), hold_frames=200)
    assert rise >= MIN_JUMP_TILES * TILE, rise


def test_jump_onto_a_4_tile_column_from_standing_next_to_it():
    # Columna de 4 tiles de alto (filas 9-12) y 3 de ancho en las cols 8-10.
    rows = ["." * 8 + "HHH" + "." * 9] * 4 + ["#" * 20] * 2
    game = new_game(rows, start_col=6)
    for _ in range(30):                              # caminar hasta la pared
        game.step(False, True, False, False)
    assert game.mario.right == 8 * TILE              # pegado a la columna
    for frame in range(120):                          # saltar y seguir empujando
        game.step(False, True, True, False)
        if frame > 5 and game.mario.on_ground:
            break
    assert game.mario.bottom == (13 - 4) * TILE      # de pie encima (y = 144)
    assert game.mario.x >= 8 * TILE


def test_short_tap_jumps_lower_than_holding():
    tap = jump_rise(new_game(), hold_frames=1)
    full = jump_rise(new_game(), hold_frames=200)
    assert 0 < tap < full


def test_release_mid_jump_gives_intermediate_height():
    tap = jump_rise(new_game(), hold_frames=1)
    mid = jump_rise(new_game(), hold_frames=12)
    full = jump_rise(new_game(), hold_frames=200)
    assert tap < mid < full


# --- Flanco: no se encadenan saltos -----------------------------------------

def count_takeoffs(game, jump_inputs):
    takeoffs = 0
    for jump in jump_inputs:
        was_on_ground = game.mario.on_ground
        game.step(False, False, jump, False)
        if was_on_ground and game.mario.vy < 0:
            takeoffs += 1
    return takeoffs


def test_holding_jump_does_not_jump_again_after_landing():
    assert count_takeoffs(new_game(), [True] * 300) == 1


def test_release_and_press_again_jumps_again():
    inputs = [True] * 150 + [False] + [True] * 150
    assert count_takeoffs(new_game(), inputs) == 2


def test_pressing_jump_while_falling_does_not_jump_on_landing():
    # Pulsa en el aire (no hay salto doble) y mantiene hasta aterrizar.
    game = new_game()
    for _ in range(40):
        game.step(False, False, True, False)           # salto completo, pasa la cima
    for _ in range(2):
        game.step(False, False, False, False)          # suelta mientras cae
    assert not game.mario.on_ground and game.mario.vy > 0
    assert count_takeoffs(game, [True] * 100) == 0


def test_no_jump_when_standing_in_the_air():
    game = new_game(["###....###"] * 2, start_col=1)
    m = game.mario
    m.x, m.y, m.on_ground = 4 * TILE + 2, 100.0, False  # sobre el foso, en el aire
    game.step(False, False, True, False)
    assert m.vy > 0


# --- Gravedad y techo -------------------------------------------------------

def test_fall_speed_is_capped():
    game = new_game(["....######"] * 2, start_col=6)
    m = game.mario
    m.x, m.y, m.on_ground = 2.0, -200.0, False          # muy alto sobre el foso
    speeds = []
    for _ in range(150):
        game.step(False, False, False, False)
        speeds.append(m.vy)
    assert max(speeds) == MAX_FALL_SPEED
    assert m.y > ROWS * TILE


def test_head_bump_stops_rise_and_mario_falls_back():
    # Ladrillo justo 2 tiles por encima de la cabeza de Mario.
    rows = ["....." + "B" + "....", "." * 10, "." * 10, "#" * 10, "#" * 10]
    game = new_game(rows, start_col=5)
    brick_bottom = (ROWS - 5 + 1) * TILE
    for _ in range(120):
        game.step(False, False, True, False)
        assert game.mario.y >= brick_bottom             # nunca entra en el ladrillo
    assert game.mario.on_ground and game.mario.bottom == FLOOR_Y


# --- Foso de 3 tiles (aviso temprano; el test definitivo va en P1.10) -------

def test_running_jump_clears_a_3_tile_pit():
    pit_start = 20
    rows = ["#" * pit_start + "." * MAX_PIT_TILES + "#" * 20] * 2
    game = new_game(rows, start_col=3)
    jumped = False
    for _ in range(400):
        # Salta cuando la punta de Mario ya está sobre el foso.
        press = game.mario.right > pit_start * TILE
        jumped |= press
        game.step(False, True, press, True)
        if jumped and game.mario.on_ground:
            break
    assert game.mario.y < ROWS * TILE                    # no cayó
    assert game.mario.x >= (pit_start + MAX_PIT_TILES) * TILE - game.mario.w
    assert game.mario.on_ground
