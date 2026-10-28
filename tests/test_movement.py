"""P1.5: movimiento horizontal (aceleración, carrera, fricción y derrape).

Criterio del plan: velocidad máxima 1.5 caminando y 2.5 corriendo.
Además se prueban reset() y el estado congelado de step(), que ya forman
parte de la API pública (INTERFACES.md §4).
"""

import pytest

from helpers import make_level
from mario_game.constants import (MARIO_H, MARIO_W, RUN_MAX_SPEED, TILE,
                                  TIME_LIMIT_FRAMES, WALK_MAX_SPEED)
from mario_game.game import MarioGame

FLAT = ["#" * 200] * 2          # suelo plano de 200 tiles; superficie en la fila 13


def new_game(rows=FLAT, start_col=2):
    return MarioGame(make_level(rows, start_col=start_col))


def hold(game, frames, left=False, right=False, jump=False, run=False):
    """Mantiene los botones `frames` frames y devuelve la lista de vx."""
    speeds = []
    for _ in range(frames):
        game.step(left, right, jump, run)
        speeds.append(game.mario.vx)
    return speeds


# --- reset() -----------------------------------------------------------------

def test_reset_places_mario_on_surface_of_start_col():
    game = new_game(start_col=5)
    m = game.mario
    assert m.x == 5 * TILE + (TILE - MARIO_W) / 2
    assert m.bottom == 13 * TILE and m.on_ground
    assert (m.vx, m.vy) == (0.0, 0.0)


def test_reset_initial_public_fields():
    game = new_game()
    assert game.goombas == [] and game.cam_x == 0.0
    assert (game.dead, game.won, game.death_reason) == (False, False, None)
    # progress = x de Mario / x de la bandera (contrato): pequeño pero no 0.
    flag_x = game.level.flag_col * TILE
    assert game.progress == game.mario.x / flag_x and game.jump_held is False
    assert game.time_left == TIME_LIMIT_FRAMES and game.frame == 0


def test_reset_accepts_start_col_and_rejects_pit():
    game = new_game(["####...#####"] * 2)
    game.reset(start_col=8)
    assert game.mario.x == 8 * TILE + 2
    with pytest.raises(ValueError):
        game.reset(start_col=5)


def test_reset_restores_after_moving():
    game = new_game()
    hold(game, 50, right=True, run=True)
    game.reset()
    assert game.mario.x == 2 * TILE + 2 and game.mario.vx == 0 and game.frame == 0


def test_step_does_nothing_after_dead_or_won():
    for flag in ("dead", "won"):
        game = new_game()
        setattr(game, flag, True)
        x, frame = game.mario.x, game.frame
        hold(game, 10, right=True, run=True)
        assert game.mario.x == x and game.frame == frame


def test_jump_held_remembers_last_input():
    game = new_game()
    game.step(False, False, True, False)
    assert game.jump_held is True
    game.step(False, False, False, False)
    assert game.jump_held is False


# --- Velocidades máximas (criterio del plan) ----------------------------------

@pytest.mark.parametrize("left, right", [(False, True), (True, False)])
def test_walk_max_speed_is_1_5(left, right):
    # Hacia la izquierda Mario acaba chocando con el borde de la cámara (C.5),
    # por eso se comprueba el máximo alcanzado y no el último frame.
    speeds = hold(new_game(start_col=100), 120, left=left, right=right)
    assert max(abs(v) for v in speeds) == WALK_MAX_SPEED


@pytest.mark.parametrize("left, right", [(False, True), (True, False)])
def test_run_max_speed_is_2_5(left, right):
    speeds = hold(new_game(start_col=100), 120, left=left, right=right, run=True)
    assert max(abs(v) for v in speeds) == RUN_MAX_SPEED


def test_releasing_run_drops_to_walk_limit_immediately():
    game = new_game()
    hold(game, 60, right=True, run=True)
    speeds = hold(game, 30, right=True, run=False)
    assert max(speeds) <= WALK_MAX_SPEED


def test_acceleration_is_gradual():
    speeds = hold(new_game(), 60, right=True, run=True)
    assert speeds[0] < WALK_MAX_SPEED                       # no arranca de golpe
    assert all(b >= a for a, b in zip(speeds, speeds[1:]))  # nunca frena


# --- Fricción, inercia y derrape -------------------------------------------

def test_friction_stops_mario_without_overshooting():
    game = new_game()
    hold(game, 60, right=True, run=True)
    x_before = game.mario.x
    speeds = hold(game, 120)                     # sin pulsar nada
    assert all(v >= 0 for v in speeds)           # nunca se va hacia atrás
    assert speeds[-1] == 0.0
    assert game.mario.x > x_before               # se desliza antes de parar


def test_pressing_both_directions_counts_as_none():
    game = new_game()
    hold(game, 60, right=True, run=True)
    a = hold(game, 5, left=True, right=True)
    b_game = new_game()
    hold(b_game, 60, right=True, run=True)
    b = hold(b_game, 5)
    assert a == b


def test_skid_brakes_faster_than_friction_then_reverses():
    def frames_to_stop(**buttons):
        game = new_game(start_col=50)
        hold(game, 60, right=True, run=True)
        for frame in range(1, 200):
            game.step(buttons.get("left", False), False, False, True)
            if game.mario.vx <= 0:
                return frame, game
        raise AssertionError("no frenó")

    skid_frames, game = frames_to_stop(left=True)
    friction_frames, _ = frames_to_stop()
    assert skid_frames < friction_frames
    hold(game, 30, left=True, run=True)
    assert game.mario.vx < 0 and game.mario.facing == -1


def test_skidding_flag_only_while_braking_on_ground():
    game = new_game(start_col=50)
    hold(game, 60, right=True, run=True)
    game.step(True, False, False, True)
    assert game.mario.skidding
    hold(game, 60, left=True, run=True)          # ya va hacia la izquierda
    assert not game.mario.skidding


def test_air_keeps_momentum_without_input():
    game = new_game()
    m = game.mario
    m.y, m.on_ground, m.vx = 50.0, False, 2.0    # en el aire, sin suelo cerca
    game.step(False, False, False, True)
    assert m.vx == 2.0 and not m.on_ground


def test_facing_follows_last_direction():
    game = new_game(start_col=50)
    hold(game, 3, left=True)
    assert game.mario.facing == -1
    hold(game, 3, right=True)
    assert game.mario.facing == 1


def test_mario_stays_on_flat_ground_while_moving():
    game = new_game()
    for _ in range(200):
        game.step(False, True, False, True)
        assert game.mario.on_ground and game.mario.bottom == 13 * TILE
    assert game.mario.h == MARIO_H
