"""Tests de mario_game/level.py (Persona 2): formato, bordes y diseño de los niveles.

Tres bloques:
- P2.2: `Level` (forma, is_solid, surface_row) con un nivel mínimo a mano.
- P2.3: `build_level_1()` se carga, es nuevo en cada llamada y se imprime en ASCII.
- P2.4: reglas de diseño de cada nivel (fosos, alturas, checkpoints, goombas).
  Cuando exista build_level_2() (P2.10), basta añadirlo a LEVEL_BUILDERS.
"""

import pytest

from helpers import overlaps_solid
from mario_game.constants import (MAX_PIT_TILES, MIN_JUMP_TILES, ROWS, T_BRICK,
                                  T_EMPTY, T_GROUND, T_HARD, T_PIPE,
                                  T_QUESTION, T_USED, TILE)
from mario_game.game import MarioGame
from mario_game.level import (Level, build_level_1, join_tramos,
                              tiles_from_ascii)

LEVEL_BUILDERS = [build_level_1]

# Nivel mínimo de 7 columnas (filas 9 a 14; por encima hay aire):
#   col 0: suelo con un bloque ? flotante en la fila 9
#   col 1: foso            col 2: suelo + tubería de 1
#   col 3: suelo + 2 bloques duros     col 4: suelo
#   col 5: foso            col 6: suelo
SMALL_ART = [
    "?......",
    ".......",
    "...H...",
    "..PH...",
    "#.###.#",
    "#.###.#",
]


def small_level(**kwargs) -> Level:
    defaults = dict(goomba_cols=[], start_col=0, flag_col=6, checkpoints=[0])
    defaults.update(kwargs)
    return Level(tiles_from_ascii(SMALL_ART), **defaults)


# --- P2.2: la clase Level -----------------------------------------------------

def test_shape_and_width():
    level = small_level()
    assert len(level.tiles) == ROWS
    assert level.width == 7
    assert all(len(row) == 7 for row in level.tiles)


def test_rejects_wrong_number_of_rows():
    with pytest.raises(ValueError, match="filas"):
        Level([[0] * 5] * 3, [], 0, 4, [0])


def test_rejects_ragged_rows():
    tiles = [[0] * 5 for _ in range(ROWS)]
    tiles[3] = [0] * 4
    with pytest.raises(ValueError, match="mismo"):
        Level(tiles, [], 0, 4, [0])


def test_tiles_are_mutable_lists():
    # El motor modifica level.tiles (bloques ?) y lo restaura en reset().
    level = Level(tuple(tuple(r) for r in tiles_from_ascii(SMALL_ART)), [], 0, 6, [0])
    level.tiles[9][0] = T_USED
    assert level.tiles[9][0] == T_USED


def test_is_solid_below_the_map_is_empty():
    level = small_level()
    assert not level.is_solid(ROWS, 0)
    assert not level.is_solid(ROWS + 5, 3)
    # La regla de abajo va antes que la de los lados: fuera del mapa por abajo
    # tampoco es pared.
    assert not level.is_solid(ROWS, -1)
    assert not level.is_solid(ROWS, level.width)


def test_is_solid_sides_are_walls():
    level = small_level()
    assert level.is_solid(5, -1)
    assert level.is_solid(5, level.width)
    assert level.is_solid(0, -1)
    assert level.is_solid(ROWS - 1, level.width + 3)


def test_is_solid_above_the_map_is_air():
    level = small_level()
    assert not level.is_solid(-1, 3)
    assert not level.is_solid(-10, 0)


def test_is_solid_follows_tile_codes():
    level = small_level()
    assert level.is_solid(13, 0)            # suelo
    assert level.is_solid(9, 0)             # bloque ?
    assert level.is_solid(12, 2)            # tubería
    assert level.is_solid(11, 3)            # bloque duro
    assert not level.is_solid(12, 0)        # aire
    assert not level.is_solid(14, 1)        # foso


def test_every_solid_code_is_solid():
    for code in (T_GROUND, T_BRICK, T_QUESTION, T_USED, T_PIPE, T_HARD):
        tiles = [[T_EMPTY] * 3 for _ in range(ROWS)]
        tiles[5][1] = code
        assert Level(tiles, [], 0, 2, [0]).is_solid(5, 1)


def test_surface_row():
    level = small_level()
    assert level.surface_row(0) == 13       # el bloque ? flotante no cuenta
    assert level.surface_row(1) is None     # foso
    assert level.surface_row(2) == 12       # suelo + tubería
    assert level.surface_row(3) == 11       # suelo + 2 bloques duros
    assert level.surface_row(4) == 13
    assert level.surface_row(5) is None
    assert level.surface_row(6) == 13


def test_surface_row_of_a_column_that_reaches_the_top():
    tiles = [[T_GROUND] * 3 for _ in range(ROWS)]
    assert Level(tiles, [], 0, 2, [0]).surface_row(1) == 0


# --- Conversión ASCII ---------------------------------------------------------

def test_tiles_from_ascii_pads_with_air_on_top():
    tiles = tiles_from_ascii(["#.#"])
    assert len(tiles) == ROWS
    assert tiles[-1] == [T_GROUND, T_EMPTY, T_GROUND]
    assert all(t == T_EMPTY for row in tiles[:-1] for t in row)


def test_tiles_from_ascii_rejects_unknown_characters_and_ragged_rows():
    with pytest.raises(ValueError, match="desconocido"):
        tiles_from_ascii(["#x#"])
    with pytest.raises(ValueError, match="medir"):
        tiles_from_ascii(["###", "##"])
    with pytest.raises(ValueError):
        tiles_from_ascii(["#"] * (ROWS + 1))


def test_join_tramos_aligns_at_the_bottom():
    rows = join_tramos([["#.", "##"], ["...", "###"]])
    assert len(rows) == ROWS
    assert rows[-2] == "#." + "..."      # el tramo 1 no tiene fila ahí: aire
    assert rows[-1] == "##" + "###"
    assert all(set(r) == {"."} for r in rows[:-2])


def test_to_ascii_round_trip():
    level = small_level()
    lines = level.to_ascii().splitlines()
    assert len(lines) == ROWS
    assert tiles_from_ascii(lines) == level.tiles


def test_to_ascii_markers_and_ruler():
    level = small_level(goomba_cols=[4], checkpoints=[0, 4])
    lines = level.to_ascii(markers=True, ruler=True).splitlines()
    assert len(lines) == ROWS + 3          # 15 filas + 2 de regla + 1 de marcas
    assert lines[1] == "0123456"
    marks = lines[-1]
    assert (marks[0], marks[4], marks[6]) == ("S", "C", "F")


# --- P2.3: build_level_1() ----------------------------------------------------

def test_build_level_1_has_the_contract_fields():
    level = build_level_1()
    assert isinstance(level, Level)
    assert len(level.tiles) == ROWS
    assert level.width == len(level.tiles[0])
    assert isinstance(level.goomba_cols, list)
    assert isinstance(level.checkpoints, list)
    assert 0 <= level.start_col < level.width
    assert 0 <= level.flag_col < level.width


def test_build_level_1_returns_a_new_level_each_call():
    a, b = build_level_1(), build_level_1()
    assert a is not b and a.tiles is not b.tiles
    a.tiles[9][5] = T_USED                  # el motor hace esto al golpear un bloque ?
    assert b.tiles[9][5] == T_QUESTION
    a.goomba_cols.clear()
    assert b.goomba_cols


def test_mariogame_without_level_uses_level_1():
    game = MarioGame()
    assert game.level.width == build_level_1().width


# --- P2.4: reglas de diseño (para cada nivel) ---------------------------------

@pytest.fixture(params=LEVEL_BUILDERS, ids=lambda f: f.__name__)
def level(request) -> Level:
    return request.param()


def surfaces(level):
    return [level.surface_row(c) for c in range(level.width)]


def test_checkpoints_are_sorted_and_include_the_start(level):
    assert level.checkpoints == sorted(set(level.checkpoints))
    assert level.start_col in level.checkpoints


def test_start_flag_and_goombas_are_on_solid_ground(level):
    for col in [level.start_col, level.flag_col, *level.goomba_cols]:
        row = level.surface_row(col)
        assert row is not None, f"la columna {col} es un foso"
        assert level.tiles[row][col] == T_GROUND, \
            f"la columna {col} no es suelo llano (tubería, escalera o bloque)"


def test_goombas_have_flat_ground_around_them(level):
    # De 6 columnas a su izquierda a 8 a su derecha, sin fosos ni escalones: el
    # encuentro con Mario y su aterrizaje tras el pisotón pasan sobre suelo llano.
    rows = surfaces(level)
    for col in level.goomba_cols:
        window = rows[max(col - 6, 0):col + 9]
        assert window[0] is not None and len(set(window)) == 1, \
            f"el goomba {col} está cerca de un foso o un obstáculo"


def test_goombas_are_spaced_out(level):
    cols = sorted(level.goomba_cols)
    assert all(b - a >= 10 for a, b in zip(cols, cols[1:])), \
        "dos goombas a menos de 10 columnas: hay que pisar los dos seguidos"


def test_checkpoints_are_not_on_pits_or_obstacles(level):
    for col in level.checkpoints:
        row = level.surface_row(col)
        assert row is not None, f"el checkpoint {col} cae sobre un foso"
        assert level.tiles[row][col] == T_GROUND, \
            f"el checkpoint {col} cae sobre un obstáculo"


def test_no_pit_is_wider_than_the_contract_limit(level):
    run = 0
    for col, row in enumerate(surfaces(level)):
        run = run + 1 if row is None else 0
        assert run <= MAX_PIT_TILES, f"foso de más de {MAX_PIT_TILES} tiles en la columna {col}"


def test_no_obstacle_is_taller_than_the_jump_limit(level):
    rows = surfaces(level)
    for col in range(1, level.width):
        if rows[col] is not None and rows[col - 1] is not None:
            rise = rows[col - 1] - rows[col]
            assert rise <= MIN_JUMP_TILES, f"escalón de {rise} tiles en la columna {col}"
    ground = max(r for r in rows if r is not None)
    assert ground - min(r for r in rows if r is not None) <= MIN_JUMP_TILES


def test_floating_blocks_leave_room_to_jump_under_them(level):
    # Mínimo 3 tiles de aire entre el suelo y cualquier bloque flotante (todo lo
    # que está por encima de surface_row es flotante).
    for col, surface in enumerate(surfaces(level)):
        if surface is None:
            continue
        for row in range(surface):
            if level.tiles[row][col] != T_EMPTY:
                assert surface - row - 1 >= 3, f"bloque a ras del suelo en ({row}, {col})"


def test_checkpoints_have_room_to_run_before_the_next_hazard(level):
    rows = surfaces(level)
    for col in level.checkpoints:
        for ahead in range(col + 1, level.width):
            hazard = rows[ahead] is None or rows[ahead] < rows[ahead - 1]
            if hazard:
                assert ahead - col >= 5, \
                    f"el checkpoint {col} está a {ahead - col} tiles del obstáculo de {ahead}"
                break


def test_the_flag_has_room_for_the_castle(level):
    assert level.width - level.flag_col >= 10


def test_engine_accepts_every_checkpoint(level):
    for col in level.checkpoints:
        game = MarioGame(level)
        game.reset(start_col=col)
        assert game.mario.on_ground
        assert not overlaps_solid(game.mario, game.level)
        assert not (game.dead or game.won)
        # Los goombas que aparecen están a más de 3 columnas de Mario y en suelo.
        for goomba in game.goombas:
            assert goomba.x // TILE > col + MarioGame.GOOMBA_SAFE_COLS
            assert not overlaps_solid(goomba, game.level)
