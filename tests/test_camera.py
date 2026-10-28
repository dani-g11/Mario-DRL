"""P1.7: cámara que no retrocede y límite izquierdo de Mario.

Criterio del plan: Mario no sale por la izquierda de la pantalla.
Contrato (INTERFACES.md §4): 0 <= cam_x <= level.width*TILE - SCREEN_W y
cam_x nunca disminuye dentro de un episodio.
"""

from helpers import make_level
from mario_game.constants import SCREEN_W, TILE
from mario_game.game import MarioGame

WIDTH = 80
FLAT = ["#" * WIDTH] * 2
MAX_CAM = WIDTH * TILE - SCREEN_W


def new_game(rows=FLAT, start_col=2):
    return MarioGame(make_level(rows, start_col=start_col))


def mario_center(game):
    return game.mario.x + game.mario.w / 2


def run(game, frames, left=False, right=False, jump=False):
    """Corre `frames` frames; comprueba las invariantes de cámara en cada uno."""
    for _ in range(frames):
        before = game.cam_x
        game.step(left, right, jump, True)
        assert game.cam_x >= before                       # nunca retrocede
        assert 0.0 <= game.cam_x <= MAX_CAM               # dentro del nivel
        assert game.mario.x >= game.cam_x                 # no sale por la izquierda


def test_camera_starts_at_zero_near_level_start():
    assert new_game(start_col=2).cam_x == 0.0


def test_reset_centers_camera_on_mario_far_from_start():
    game = new_game(start_col=40)
    assert game.cam_x == mario_center(game) - SCREEN_W / 2


def test_camera_follows_mario_past_screen_center():
    game = new_game()
    run(game, 150, right=True)
    assert game.cam_x > 0
    assert mario_center(game) - game.cam_x == SCREEN_W / 2   # Mario centrado


def test_camera_does_not_go_back_when_mario_walks_left():
    game = new_game()
    run(game, 150, right=True)
    while game.mario.vx >= 0:          # el derrape aún lo lleva a la derecha
        run(game, 1, left=True)
    cam = game.cam_x
    run(game, 60, left=True)
    assert game.cam_x == cam


def test_mario_cannot_leave_through_left_edge_of_screen():
    game = new_game()
    run(game, 150, right=True)
    run(game, 300, left=True)                 # empuja contra el borde
    assert game.mario.x == game.cam_x
    assert game.mario.vx == 0.0


def test_left_edge_also_holds_while_jumping():
    game = new_game()
    run(game, 150, right=True)
    run(game, 300, left=True, jump=True)


def test_left_limit_at_level_start():
    game = new_game(start_col=0)
    run(game, 60, left=True)
    assert game.mario.x == 0.0


def test_camera_stops_at_end_of_level():
    game = new_game()
    run(game, 900, right=True)
    assert game.cam_x == MAX_CAM


def test_level_narrower_than_screen_keeps_camera_at_zero():
    game = new_game(["#" * 10] * 2, start_col=5)
    for _ in range(100):
        game.step(False, True, False, True)
        assert game.cam_x == 0.0


def test_reset_resets_camera():
    game = new_game()
    run(game, 150, right=True)
    game.reset()
    assert game.cam_x == 0.0
