"""Tests de mario_game/sprites.py y mario_game/render.py (Persona 2, P2.5 a P2.7).

Todo se prueba sin ventana: se dibuja en superficies en memoria (conftest.py fija
SDL_VIDEODRIVER=dummy). El nombre del archivo es distinto de tests/test_render.py,
que prueba la delegación de MarioGame.render() con un render falso.
"""

import pytest

pygame = pytest.importorskip("pygame")

from mario_game import render, sprites  # noqa: E402
from mario_game.constants import (FPS, GOOMBA_H, GOOMBA_W, MARIO_H, MARIO_W,  # noqa: E402
                                  SCREEN_H, SCREEN_W, T_BRICK, T_GROUND, T_HARD,
                                  T_QUESTION, T_USED, TILE)
from mario_game.game import Goomba, MarioGame  # noqa: E402
from mario_game.level import Level, build_level_1, tiles_from_ascii  # noqa: E402


def new_surface():
    return pygame.Surface((SCREEN_W, SCREEN_H))


def pixels(surface) -> bytes:
    return pygame.image.tobytes(surface, "RGB")


def count_color(surface, color) -> int:
    return pygame.mask.from_threshold(surface, color, (1, 1, 1, 255)).count()


def snapshot(game):
    m = game.mario
    return (m.x, m.y, m.vx, m.vy, m.on_ground, m.facing, m.skidding, game.cam_x,
            game.frame, game.time_left, game.score, game.coins, game.progress,
            game.dead, game.won, [list(row) for row in game.level.tiles],
            [(g.x, g.y, g.vx, g.squashed) for g in game.goombas])


# --- P2.5: sprites ----------------------------------------------------------------

@pytest.mark.parametrize("pose", sprites.MARIO_POSES)
@pytest.mark.parametrize("facing", [1, -1])
def test_mario_sprites_match_the_collision_box(pose, facing):
    assert sprites.mario_sprite(pose, facing).get_size() == (MARIO_W, MARIO_H)


def test_mario_has_four_distinct_poses_and_can_face_left():
    assert len(sprites.MARIO_POSES) == 4
    images = {pose: pixels(sprites.mario_sprite(pose)) for pose in sprites.MARIO_POSES}
    assert len(set(images.values())) == 4
    for pose in sprites.MARIO_POSES:
        assert pixels(sprites.mario_sprite(pose, -1)) != images[pose]


def test_mario_sprite_is_cached_and_rejects_unknown_poses():
    assert sprites.mario_sprite("idle") is sprites.mario_sprite("idle")
    with pytest.raises(ValueError, match="pose"):
        sprites.mario_sprite("volando")


def test_goomba_has_two_states():
    walking, squashed = sprites.goomba_sprite(), sprites.goomba_sprite(squashed=True)
    assert walking.get_size() == (GOOMBA_W, GOOMBA_H)
    assert squashed.get_size() == (GOOMBA_W, GOOMBA_H // 2)
    assert pixels(walking) != pixels(sprites.goomba_sprite(flip=True))   # los pies alternan


def test_every_tile_has_a_16x16_sprite_and_they_are_all_different():
    codes = (T_GROUND, T_BRICK, T_QUESTION, T_USED, T_HARD)
    images = [sprites.tile_sprite(code) for code in codes]
    assert all(img.get_size() == (TILE, TILE) for img in images)
    assert len({pixels(img) for img in images}) == len(codes)


def test_pipe_has_four_different_quarters():
    quarters = [sprites.pipe_sprite(top, right) for top in (True, False) for right in (True, False)]
    assert all(q.get_size() == (TILE, TILE) for q in quarters)
    assert len({pygame.image.tobytes(q, "RGBA") for q in quarters}) == 4


def test_tile_sprite_rejects_unknown_codes_and_pipes():
    with pytest.raises(ValueError):
        sprites.tile_sprite(99)
    with pytest.raises(ValueError, match="pipe_sprite"):
        sprites.tile_sprite(5)


def test_surface_from_art_validates_its_input():
    with pytest.raises(ValueError, match="mide"):
        sprites.surface_from_art(["ab", "a"], {"a": (0, 0, 0), "b": (1, 1, 1)})
    with pytest.raises(ValueError, match="paleta"):
        sprites.surface_from_art(["az"], {"a": (0, 0, 0)})


def test_sprite_sheet_can_be_built():
    assert sprites.build_sheet(scale=2).get_width() > 100


# --- P2.6: escenario ----------------------------------------------------------------

def test_draw_rejects_a_surface_of_the_wrong_size():
    with pytest.raises(ValueError, match=f"{SCREEN_W}"):
        render.draw(pygame.Surface((100, 100)), MarioGame(build_level_1()))


def test_draw_is_deterministic():
    game = MarioGame(build_level_1())
    a, b = new_surface(), new_surface()
    render.draw(a, game)
    render.draw(b, game)
    assert pixels(a) == pixels(b)


def test_draw_does_not_change_the_game():
    game = MarioGame(build_level_1())
    for _ in range(90):
        game.step(False, True, False, True)
    before = snapshot(game)
    render.draw(new_surface(), game)
    render.draw(new_surface(), game, hud=False)
    assert snapshot(game) == before


def test_the_picture_follows_the_camera():
    game = MarioGame(build_level_1())
    a, b = new_surface(), new_surface()
    render.draw(a, game, hud=False)
    game.cam_x += 64
    render.draw(b, game, hud=False)
    assert pixels(a) != pixels(b)


def test_every_part_of_the_level_can_be_drawn():
    game = MarioGame(build_level_1())
    surface = new_surface()
    last = game.level.width * TILE - SCREEN_W
    for cam in list(range(0, last, 48)) + [last]:
        game.cam_x = float(cam)
        render.draw(surface, game)
        # Siempre hay suelo en las dos filas de abajo, salvo en los fosos.
        assert count_color(surface, sprites.SKY_COLOR) < SCREEN_W * SCREEN_H


def test_draws_the_flag_pole_and_the_castle_when_they_are_on_screen():
    game = MarioGame(build_level_1())
    surface = new_surface()
    game.cam_x = float(game.level.flag_col * TILE - 64)
    render.draw(surface, game, hud=False)
    assert count_color(surface, (152, 232, 120)) > 40          # el mástil
    assert count_color(surface, (200, 80, 32)) > 200           # los ladrillos del castillo
    game.cam_x = 0.0
    render.draw(surface, game, hud=False)
    assert count_color(surface, (152, 232, 120)) == 0


def test_pipe_quarters_are_chosen_by_their_neighbours():
    # Una tubería de 2 de ancho y 2 de alto en las columnas 1 y 2.
    level = Level(tiles_from_ascii([".PP.", ".PP.", "####"]), [], 0, 3, [0])
    tiles, rows = level.tiles, len(level.tiles)
    top_row = rows - 3
    assert render._pipe_image(tiles, top_row, 1) is sprites.pipe_sprite(True, False)
    assert render._pipe_image(tiles, top_row, 2) is sprites.pipe_sprite(True, True)
    assert render._pipe_image(tiles, top_row + 1, 1) is sprites.pipe_sprite(False, False)
    assert render._pipe_image(tiles, top_row + 1, 2) is sprites.pipe_sprite(False, True)


def test_a_hit_question_block_looks_different():
    game = MarioGame(build_level_1())
    before, after = new_surface(), new_surface()
    game.cam_x = 0.0
    render.draw(before, game, hud=False)
    game.level.tiles[9][5] = T_USED                 # lo que hace el motor al golpearlo
    render.draw(after, game, hud=False)
    assert pixels(before) != pixels(after)


# --- P2.7: personajes y HUD -------------------------------------------------------------

def test_mario_is_drawn_inside_his_collision_box():
    game = MarioGame(build_level_1())
    surface = new_surface()
    render.draw(surface, game, hud=False)
    m = game.mario
    box = pygame.Rect(int(m.x - game.cam_x), int(m.y), MARIO_W, MARIO_H)
    assert count_color(surface.subsurface(box), sprites.MARIO_RED) > 10


def test_mario_pose_follows_his_state():
    game = MarioGame(build_level_1())
    m = game.mario
    assert render._mario_pose(game) == "idle"
    m.vx = 2.0
    m.x = 0.0                                      # (int(x) // 6) % 2 == 0: paso abierto
    assert render._mario_pose(game) == "walk"
    m.skidding = True
    assert render._mario_pose(game) == "skid"
    m.on_ground = False
    assert render._mario_pose(game) == "jump"
    m.on_ground, m.skidding = True, False
    game.dead = True
    assert render._mario_pose(game) == "jump"


def test_goombas_and_squashed_goombas_are_drawn_at_their_position():
    game = MarioGame(build_level_1())
    surface = new_surface()
    game.cam_x = 0.0
    game.goombas = []
    render.draw(surface, game, hud=False)
    empty = pixels(surface)
    walking = Goomba(10 * TILE, 12 * TILE, 0.5)
    game.goombas = [walking]
    render.draw(surface, game, hud=False)
    with_walking = pixels(surface)
    walking.squash()
    render.draw(surface, game, hud=False)
    assert empty != with_walking != pixels(surface)


def test_goombas_outside_the_view_are_skipped():
    game = MarioGame(build_level_1())
    game.cam_x = 0.0
    game.goombas = []
    a, b = new_surface(), new_surface()
    render.draw(a, game, hud=False)
    game.goombas = [Goomba(100 * TILE, 12 * TILE, 0.5)]
    render.draw(b, game, hud=False)
    assert pixels(a) == pixels(b)


def test_hud_only_touches_the_top_strip():
    game = MarioGame(build_level_1())
    with_hud, without_hud = new_surface(), new_surface()
    render.draw(with_hud, game, hud=True)
    render.draw(without_hud, game, hud=False)
    assert pixels(with_hud) != pixels(without_hud)
    below = pygame.Rect(0, render.HUD_HEIGHT, SCREEN_W, SCREEN_H - render.HUD_HEIGHT)
    assert pixels(with_hud.subsurface(below)) == pixels(without_hud.subsurface(below))


def test_hud_text_formats():
    game = MarioGame(build_level_1())
    game.score, game.coins, game.progress, game.time_left = 1200, 3, 0.317, 245 * FPS + 17
    assert render.hud_text(game) == {"score": "001200", "coins": "X03",
                                     "progress": "31", "time": "245"}
    game.progress, game.time_left = 1.0, -5
    assert render.hud_text(game)["progress"] == "100"
    assert render.hud_text(game)["time"] == "000"


def test_font_has_every_character_the_hud_uses():
    for ch, rows in render._GLYPHS.items():
        assert len(rows) == 5 and all(len(r) == 3 for r in rows), f"glifo {ch!r} mal formado"
    game = MarioGame(build_level_1())
    used = set("SCORE COINS PROG TIME" + "".join(render.hud_text(game).values()))
    assert used <= set(render._GLYPHS)


def test_the_hud_changes_with_the_game_state():
    game = MarioGame(build_level_1())
    a, b = new_surface(), new_surface()
    render.draw(a, game)
    game.score, game.progress = 500, 0.5
    render.draw(b, game)
    assert pixels(a) != pixels(b)


def test_game_render_uses_this_module():
    game = MarioGame(build_level_1())
    surface = new_surface()
    game.render(surface)
    direct = new_surface()
    render.draw(direct, game)
    assert pixels(surface) == pixels(direct)
