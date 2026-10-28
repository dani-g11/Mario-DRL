"""P1.10: suite que verifica el contrato del motor (INTERFACES.md §4 y §5).

Si alguno de estos tests falla, el motor no se publica (riesgo "la física no
cumple los límites" del plan). Son tests de integración: usan solo la API
pública de MarioGame y niveles mínimos de tests/helpers.py.
"""

import inspect
import random

import pytest

from helpers import make_level, overlaps_solid
from mario_game.constants import (MAX_PIT_TILES, MIN_JUMP_TILES, ROWS,
                                  RUN_MAX_SPEED, SCREEN_W, TILE,
                                  WALK_MAX_SPEED)
from mario_game.game import Goomba, Mario, MarioGame

# Las 7 acciones del entorno (INTERFACES.md §7): (left, right, jump, run).
ACTIONS = [(False, False, False, False), (False, True, False, False),
           (False, True, True, False), (False, True, False, True),
           (False, True, True, True), (False, False, True, False),
           (True, False, False, False)]

# Nivel con todos los elementos: fosos de 1 a 3 tiles, tuberías, escaleras de
# hasta 4 tiles, bloques ? y ladrillos, y goombas.
COMPLEX = [
    "...........?B?........................................",
    "........................................HHHH..........",
    "........................................HHHH..........",
    "..............P...........P..........HHHHHHH..........",
    "..............P...........P..HH......HHHHHHH..........",
    "##########.########..#############...#################",
    "##########.########..#############...#################",
]
# Colocados para que el bot heurístico de abajo (que no reacciona en el aire)
# pueda ganar: con un goomba justo al pie de la escalera, el bot cae a su lado
# y muere por contacto lateral, que es el comportamiento correcto del motor.
COMPLEX_GOOMBAS = [7, 16, 28, 46, 50]


def complex_game(**kwargs):
    level = make_level(COMPLEX, goomba_cols=COMPLEX_GOOMBAS, start_col=1,
                       flag_col=52, checkpoints=[1, 23])
    return MarioGame(level, **kwargs)


def snapshot(game):
    """Estado público completo, comparable con ==."""
    m = game.mario
    return (
        (m.x, m.y, m.vx, m.vy, m.on_ground, m.facing, m.skidding),
        tuple((g.x, g.y, g.direction, g.active, g.squashed) for g in game.goombas),
        game.cam_x, game.dead, game.won, game.death_reason, game.progress,
        game.jump_held, game.score, game.coins, game.time_left, game.frame,
        tuple(tuple(r) for r in game.level.tiles),
    )


# --- §4: firmas y campos públicos -------------------------------------------

def test_public_method_signatures():
    assert list(inspect.signature(MarioGame.reset).parameters) == ["self", "start_col"]
    assert inspect.signature(MarioGame.reset).parameters["start_col"].default is None
    assert list(inspect.signature(MarioGame.step).parameters) == [
        "self", "left", "right", "jump", "run"]
    render = inspect.signature(MarioGame.render).parameters
    assert list(render) == ["self", "surface", "hud"] and render["hud"].default is True
    init = inspect.signature(MarioGame.__init__).parameters
    assert init["level"].default is None and "goomba_speed" in init


def test_public_fields_have_contract_types():
    game = complex_game()
    for _ in range(2):
        assert isinstance(game.mario, Mario)
        assert isinstance(game.goombas, list)
        assert all(isinstance(g, Goomba) for g in game.goombas)
        assert game.level is not None
        assert isinstance(game.cam_x, float)
        assert isinstance(game.dead, bool) and isinstance(game.won, bool)
        assert game.death_reason in (None, "pit", "goomba", "time")
        assert isinstance(game.progress, float)
        assert isinstance(game.jump_held, bool)
        for name in ("score", "coins", "time_left", "frame"):
            assert isinstance(getattr(game, name), int), name
        assert isinstance(game.goomba_speed, float)
        assert game.step(False, True, True, True) is None   # step devuelve None


def test_info_keys_can_be_built_from_public_fields():
    # Lo que hará P3 (INTERFACES.md §7).
    game = complex_game()
    game.step(False, True, False, True)
    info = {"x_pos": game.mario.x, "progress": game.progress,
            "flag_get": game.won, "dead": game.dead,
            "death_reason": game.death_reason}
    assert isinstance(info["x_pos"], float) and 0 <= info["progress"] <= 1


def test_reset_from_each_checkpoint():
    game = complex_game()
    for col in game.level.checkpoints:
        game.reset(start_col=col)
        assert game.mario.on_ground and not game.dead
        assert game.mario.x >= game.cam_x
        assert all(int(g.x // TILE) > col + MarioGame.GOOMBA_SAFE_COLS
                   for g in game.goombas)


def test_reset_after_episode_equals_fresh_game():
    game = complex_game()
    rng = random.Random(1)
    for _ in range(400):
        game.step(*ACTIONS[rng.randrange(7)])
    game.reset()
    assert snapshot(game) == snapshot(complex_game())


# --- §5: límites físicos garantizados ---------------------------------------

def test_jump_height_at_least_4_tiles():
    game = MarioGame(make_level(["#" * 20] * 2, start_col=5))
    floor = game.mario.bottom
    top = floor
    for _ in range(120):
        game.step(False, False, True, False)
        top = min(top, game.mario.bottom)
    assert floor - top >= MIN_JUMP_TILES * TILE


@pytest.mark.parametrize("jump_distance", [4, 20, 40])
def test_running_jump_onto_4_tile_step(jump_distance):
    wall = 20
    rows = ["." * wall + "H" * 10] * 4 + ["#" * (wall + 10)] * 2
    game = MarioGame(make_level(rows, start_col=2))
    jumped = False
    for _ in range(400):
        press = jumped or wall * TILE - game.mario.right <= jump_distance
        jumped |= press
        game.step(False, True, press, True)
        if jumped and game.mario.on_ground and game.mario.x > wall * TILE:
            break
    assert game.mario.bottom == (13 - 4) * TILE        # encima del escalón


@pytest.mark.parametrize("pit_width", range(1, MAX_PIT_TILES + 1))
@pytest.mark.parametrize("runway_tiles", [3, 6, 12])
@pytest.mark.parametrize("timing", ["front_over_pit", "half_over_pit"])
def test_running_jump_clears_pits_up_to_3_tiles(pit_width, runway_tiles, timing):
    pit = 2 + runway_tiles
    rows = ["#" * pit + "." * pit_width + "#" * 20] * 2
    game = MarioGame(make_level(rows, start_col=2))
    m = game.mario
    edge = pit * TILE
    trigger = edge if timing == "front_over_pit" else edge + m.w / 2
    jumped = False
    for _ in range(400):
        jumped |= m.right > trigger
        game.step(False, True, jumped, True)
        if jumped and m.on_ground:
            break
    assert not game.dead
    assert m.on_ground and m.x >= (pit + pit_width) * TILE - m.w


def test_speed_limits_hold_in_every_frame_of_random_play():
    game = complex_game()
    rng = random.Random(7)
    for _ in range(3000):
        a = ACTIONS[rng.randrange(7)]
        for _ in range(4):
            game.step(*a)
            limit = RUN_MAX_SPEED if a[3] else WALK_MAX_SPEED
            assert abs(game.mario.vx) <= limit
            assert abs(game.mario.vy) < TILE
        if game.dead or game.won:
            game.reset()


# --- Invariantes de toda partida --------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_invariants_under_random_play(seed):
    """Juego aleatorio con acciones del agente (4 frames cada una)."""
    game = complex_game()
    rng = random.Random(seed)
    episodes = 0
    for _ in range(1500):
        a = ACTIONS[rng.randrange(7)]
        for _ in range(4):
            cam, progress = game.cam_x, game.progress
            game.step(*a)
            m = game.mario
            if not game.dead:
                assert not overlaps_solid(m, game.level)
            for g in game.goombas:
                assert not overlaps_solid(g, game.level)
            assert game.cam_x >= cam and game.progress >= progress
            assert 0.0 <= game.cam_x <= game.level.width * TILE - SCREEN_W
            assert m.x >= game.cam_x
            assert 0.0 <= game.progress <= 1.0
            assert not (game.dead and game.won)
            assert (game.death_reason is None) == (not game.dead)
            if game.dead and game.death_reason == "pit":
                assert m.y >= ROWS * TILE
        if game.dead or game.won:
            episodes += 1
            game.reset(start_col=rng.choice(game.level.checkpoints))
    assert episodes > 0                                   # hubo muertes/victorias


def test_simulation_is_deterministic():
    def play(seed):
        game = complex_game(goomba_speed=0.7)
        rng = random.Random(seed)
        states = []
        for _ in range(600):
            game.step(*ACTIONS[rng.randrange(7)])
            states.append(snapshot(game))
            if game.dead or game.won:
                game.reset()
        return states

    assert play(3) == play(3)
    assert play(3) != play(4)


def obstacle_ahead(game, look_px=24):
    """Bot heurístico mínimo: ¿hay un foso, una pared o un goomba delante?"""
    m, level = game.mario, game.level
    feet_row = int((m.bottom - 1) // TILE)
    for px in (m.right + 1, m.right + look_px):
        col = int(px // TILE)
        if col >= level.width:
            return False
        if level.surface_row(col) is None or level.is_solid(feet_row, col):
            return True
    return any(g.active and not g.squashed and 0 < g.x - m.right < 48
               and abs(g.y - m.y) < TILE for g in game.goombas)


def test_heuristic_bot_completes_the_complex_level():
    """Humo de integración: un bot sencillo corre a la derecha y salta ante
    fosos, paredes y goombas. Comprueba que motor y reglas encajan en un nivel
    con todos los elementos y que se puede ganar desde cada checkpoint."""
    game = complex_game()
    for col in game.level.checkpoints:
        game.reset(start_col=col)
        jump = False
        for _ in range(3000):
            m = game.mario
            if m.on_ground:
                # En el suelo: pulsa si hay obstáculo (si el botón venía
                # pulsado, no despega hasta soltarlo: salto por flanco).
                jump = obstacle_ahead(game)
            else:
                # En el aire: mantiene mientras sube (salto completo) y suelta
                # al caer, para poder volver a saltar justo al aterrizar.
                jump = m.vy < 0
            game.step(False, True, jump, True)
            if game.won or game.dead:
                break
        assert game.won, (col, game.death_reason, game.mario.x / TILE)
        assert game.progress == 1.0
