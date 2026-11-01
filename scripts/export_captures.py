"""Exporta capturas PNG del render (P2.5, P2.6 y P2.7).

Genera, en la carpeta de salida:
  sprites.png         todos los sprites ampliados (P2.5)
  nivel1_tramoN.png   cada tramo del nivel 1, tal como lo dibuja el render (P2.6)
  entidades.png       Mario, goombas y HUD: el inicio de una partida y un estado
                      ficticio con Mario saltando y un goomba aplastado (P2.7)

Uso:  python scripts/export_captures.py [carpeta_de_salida]
      (por defecto docs/captures)
"""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")        # sin ventana
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame  # noqa: E402

from mario_game import sprites  # noqa: E402
from mario_game.constants import FPS, SCREEN_H, SCREEN_W, TILE  # noqa: E402
from mario_game.game import Goomba, MarioGame  # noqa: E402
from mario_game.level import TRAMO_ANCHO, build_level_1  # noqa: E402
from mario_game.render import GROUND_Y, draw  # noqa: E402

SCALE = 2


def save(surface, path, scale=SCALE):
    big = pygame.transform.scale(surface, (surface.get_width() * scale,
                                           surface.get_height() * scale))
    pygame.image.save(big, path)
    print(f"  {path}")


def render_strip(game, start_px, end_px):
    """El tramo [start_px, end_px) del nivel, uniendo varias vistas de pantalla."""
    level_px = game.level.width * TILE
    strip = pygame.Surface((end_px - start_px, SCREEN_H))
    view = pygame.Surface((SCREEN_W, SCREEN_H))
    for x0 in range(start_px, end_px, SCREEN_W):
        cam = min(x0, level_px - SCREEN_W)
        game.cam_x = float(cam)
        draw(view, game, hud=False)
        strip.blit(view, (cam - start_px, 0))
    return strip


def fictitious_game():
    """Estado inventado: Mario saltando junto a una tubería, un goomba y otro aplastado."""
    game = MarioGame(build_level_1())
    game.reset(start_col=22)
    mario = game.mario
    mario.x, mario.y = 27 * TILE, 9 * TILE
    mario.on_ground, mario.vx, mario.vy, mario.facing = False, 2.0, -1.0, 1
    walking = Goomba(24 * TILE, GROUND_Y - TILE, 0.5)
    squashed = Goomba(20 * TILE, GROUND_Y - TILE, 0.5)
    squashed.squash()
    game.goombas = [walking, squashed]
    game.cam_x = 18.0 * TILE
    game.frame = 8                      # segunda posición de los pies del goomba
    game.score, game.coins, game.progress = 1200, 3, 0.31
    game.time_left = 245 * FPS
    return game


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Guardando en {out_dir}:")
    save(sprites.build_sheet(scale=4), os.path.join(out_dir, "sprites.png"), scale=1)

    level_game = MarioGame(build_level_1())
    width_px = level_game.level.width * TILE
    for n, start in enumerate(range(0, width_px, TRAMO_ANCHO * TILE)):
        end = min(start + TRAMO_ANCHO * TILE, width_px)
        save(render_strip(level_game, start, end), os.path.join(out_dir, f"nivel1_tramo{n}.png"))

    scenes = pygame.Surface((SCREEN_W * 2 + 8, SCREEN_H))
    scenes.fill((0, 0, 0))
    start_view = pygame.Surface((SCREEN_W, SCREEN_H))
    draw(start_view, MarioGame(build_level_1()), hud=True)
    fiction_view = pygame.Surface((SCREEN_W, SCREEN_H))
    draw(fiction_view, fictitious_game(), hud=True)
    scenes.blit(start_view, (0, 0))
    scenes.blit(fiction_view, (SCREEN_W + 8, 0))
    save(scenes, os.path.join(out_dir, "entidades.png"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join("docs", "captures"))
