"""P1.9: reglas — pisotón, muerte por enemigo, foso y tiempo, bloques ?,
bandera y puntuación.

Criterio del plan: un test de cada regla con un escenario mínimo.
Suelo en las filas 13-14 (superficie en y = 208).
"""

import pytest

from helpers import make_level
from mario_game.constants import (FPS, ROWS, SCORE_COIN, SCORE_FLAG,
                                  SCORE_PER_SECOND_LEFT, SCORE_STOMP,
                                  SQUASH_FRAMES, T_BRICK, T_QUESTION, T_USED,
                                  TILE)
from mario_game.game import MarioGame

FLAT = ["#" * 40] * 2


def new_game(rows=FLAT, goomba_cols=(), start_col=1, flag_col=None):
    level = make_level(rows, goomba_cols=goomba_cols, start_col=start_col,
                       flag_col=flag_col)
    return MarioGame(level)


def idle(game, frames=1):
    for _ in range(frames):
        game.step(False, False, False, False)


def drop_mario_on(game, goomba, height=40):
    """Coloca a Mario en el aire justo encima del goomba."""
    m = game.mario
    m.x = goomba.x + (goomba.w - m.w) / 2
    m.y = goomba.y - m.h - height
    m.vx, m.vy, m.on_ground = 0.0, 0.0, False


# --- Pisotón -----------------------------------------------------------------

def test_stomp_squashes_goomba_scores_and_bounces():
    game = new_game(goomba_cols=[10])
    g = game.goombas[0]
    idle(game)                                        # se activa
    drop_mario_on(game, g)
    for _ in range(60):
        idle(game)
        if g.squashed:
            break
    assert g.squashed and not game.dead
    assert game.score == SCORE_STOMP
    assert game.mario.vy < 0                          # rebote hacia arriba


def test_squashed_goomba_is_harmless_and_disappears():
    game = new_game(goomba_cols=[10])
    g = game.goombas[0]
    idle(game)
    g.squash()
    game.mario.x = g.x                                 # Mario encima del aplastado
    idle(game, SQUASH_FRAMES - 1)
    assert not game.dead and g in game.goombas
    idle(game)
    assert g not in game.goombas


def test_stomp_boundary_uses_previous_position_of_feet():
    game = new_game(goomba_cols=[10])
    g = game.goombas[0]
    g.active = True
    m = game.mario
    m.x, m.y, m.vy = g.x, g.y + 4, 2.0                # se solapan; Mario cae

    game._check_goomba_contacts(prev_bottom=g.y)       # venía desde arriba
    assert g.squashed and not game.dead

    game = new_game(goomba_cols=[10])
    g = game.goombas[0]
    g.active = True
    m = game.mario
    m.x, m.y, m.vy = g.x, g.y + 4, 2.0
    game._check_goomba_contacts(prev_bottom=g.y + 1)   # venía de un lado
    assert game.dead and game.death_reason == "goomba"


# --- Muerte por goomba -------------------------------------------------------

def test_goomba_walking_into_mario_kills_him():
    game = new_game(goomba_cols=[10])
    for _ in range(600):
        idle(game)
        if game.dead:
            break
    assert game.dead and game.death_reason == "goomba" and not game.won


def test_touching_goomba_while_rising_kills():
    # Subiendo (vy < 0) nunca es pisotón, aunque los pies estuvieran arriba.
    game = new_game(goomba_cols=[10])
    g = game.goombas[0]
    g.active = True
    m = game.mario
    m.x, m.y, m.vy = g.x, g.y + 4, -3.0
    game._check_goomba_contacts(prev_bottom=g.y)
    assert game.dead and game.death_reason == "goomba"


# --- Foso y tiempo -----------------------------------------------------------

def test_falling_into_pit_kills_and_freezes():
    game = new_game(["#" * 6 + "...." + "#" * 30] * 2)
    for _ in range(300):
        game.step(False, True, False, False)
        if game.dead:
            break
    assert game.dead and game.death_reason == "pit"
    assert game.mario.y >= ROWS * TILE
    frame = game.frame
    idle(game, 5)
    assert game.frame == frame                         # congelado


def test_time_runs_out():
    game = new_game()
    game.time_left = 3
    idle(game, 2)
    assert game.time_left == 1 and not game.dead
    idle(game)
    assert game.dead and game.death_reason == "time"


def test_time_decreases_one_per_frame():
    game = new_game()
    start = game.time_left
    idle(game, 60)
    assert game.time_left == start - 60


# --- Bloques ? ---------------------------------------------------------------

# Bloques en la fila 9: dos tiles de aire entre la cabeza de Mario y el bloque.
def block_rows(blocks):
    return [blocks, "." * 40, "." * 40, "." * 40] + FLAT


def jump_once(game, frames=60):
    for f in range(frames):
        game.step(False, False, f < 30, False)


def test_question_block_gives_coin_once_and_becomes_used():
    game = new_game(block_rows("." + "?" + "." * 38))
    row = ROWS - 6
    jump_once(game)
    assert game.level.tiles[row][1] == T_USED
    assert (game.coins, game.score) == (1, SCORE_COIN)
    jump_once(game)                                    # golpear de nuevo: nada
    assert (game.coins, game.score) == (1, SCORE_COIN)


def test_brick_does_nothing():
    game = new_game(block_rows("." + "B" + "." * 38))
    jump_once(game)
    assert game.level.tiles[ROWS - 6][1] == T_BRICK and game.coins == 0


def test_only_block_under_mario_center_is_activated():
    game = new_game(block_rows("??" + "." * 38))
    game.mario.x = 1 * TILE - 4    # caja en x 12..24: cabeza bajo cols 0 y 1; centro (18) en col 1
    jump_once(game)
    row = ROWS - 6
    assert game.level.tiles[row][0] == T_QUESTION
    assert game.level.tiles[row][1] == T_USED


def test_reset_restores_question_blocks():
    game = new_game(block_rows("." + "?" + "." * 38))
    tiles_obj = game.level.tiles
    jump_once(game)
    game.reset()
    assert game.level.tiles[ROWS - 6][1] == T_QUESTION
    assert game.level.tiles is tiles_obj               # mismo objeto, restaurado
    assert (game.coins, game.score) == (0, 0)


# --- Bandera y progreso ------------------------------------------------------

def run_to_flag(game, frames=1000):
    for _ in range(frames):
        game.step(False, True, False, True)
        if game.won or game.dead:
            return


def test_touching_flag_wins_with_score_and_full_progress():
    game = new_game(flag_col=30)
    run_to_flag(game)
    assert game.won and not game.dead and game.death_reason is None
    assert game.mario.right > 30 * TILE
    assert game.progress == 1.0
    assert game.score == SCORE_FLAG + SCORE_PER_SECOND_LEFT * (game.time_left // FPS)
    frame = game.frame
    idle(game, 5)
    assert game.frame == frame                         # congelado


def test_progress_is_monotonic_and_bounded():
    game = new_game(flag_col=30)
    history = []
    for _ in range(100):
        game.step(False, True, False, True)
        history.append(game.progress)
    for _ in range(100):
        game.step(True, False, False, True)            # vuelve atrás
        history.append(game.progress)
    assert all(0.0 <= p <= 1.0 for p in history)
    assert all(b >= a for a, b in zip(history, history[1:]))


def test_progress_starts_from_start_col():
    game = new_game(flag_col=30, start_col=15)
    assert game.progress == pytest.approx(game.mario.x / (30 * TILE))


# --- Validación del nivel (INTERFACES.md v0.2) ------------------------------

def test_goomba_over_pit_raises_value_error():
    with pytest.raises(ValueError, match="foso"):
        new_game(["#" * 10 + "..." + "#" * 27] * 2, goomba_cols=[11])


def test_goomba_outside_level_raises_value_error():
    with pytest.raises(ValueError, match="fuera"):
        new_game(goomba_cols=[99])


def test_start_col_outside_level_raises_value_error():
    game = new_game()
    with pytest.raises(ValueError, match="fuera"):
        game.reset(start_col=-1)
