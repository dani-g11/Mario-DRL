"""Render de Mario DRL (Persona 2, P2.6 y P2.7): `draw(surface, game, hud)`.

`MarioGame.render()` delega aquí (INTERFACES.md §4). Lee los campos públicos del
juego y dibuja la vista actual en una superficie de SCREEN_W × SCREEN_H; el
escalado lo hace quien llama. No modifica el juego.

Orden de dibujo: cielo, nubes y colinas, castillo, bandera, tiles, goombas,
Mario y, si hud=True, el HUD. La bandera y el castillo no son tiles: salen de
`level.flag_col` (el castillo es solo decoración, a la derecha de la bandera).

Funciona sin ventana: no usa pygame.display, convert() ni pygame.font (el HUD usa
una fuente de píxeles propia), así que sirve para la observación `pixels` del
entorno y durante el entrenamiento.
"""

from functools import lru_cache

import pygame

from mario_game import sprites
from mario_game.constants import (FPS, ROWS, SCREEN_H, SCREEN_W, T_EMPTY,
                                  T_PIPE, TILE)

WHITE = (252, 252, 252)
BLACK = (0, 0, 0)
GROUND_Y = (ROWS - 2) * TILE            # y de la superficie del suelo normal (fila 13)
CASTLE_COL_OFFSET = 4                   # el castillo empieza 4 columnas tras la bandera
CASTLE_COLS = 5
FLAG_POLE_TILES = 10                    # alto del mástil
HUD_HEIGHT = 38                         # franja superior que ocupa el HUD


def draw(surface, game, hud: bool = True) -> None:
    """Dibuja la vista actual de `game` (desde `game.cam_x`) en `surface`."""
    _require_screen_size(surface)
    level = game.level
    cam = int(game.cam_x)
    surface.fill(sprites.SKY_COLOR)
    _draw_clouds(surface, level, cam)
    _draw_hills(surface, level, cam)
    _draw_castle(surface, level, cam)
    _draw_flag(surface, level, cam)
    _draw_tiles(surface, level, cam)
    _draw_goombas(surface, game, cam)
    _draw_mario(surface, game, cam)
    if hud:
        _draw_hud(surface, game)


def _require_screen_size(surface) -> None:
    if surface.get_size() != (SCREEN_W, SCREEN_H):
        raise ValueError(f"la superficie debe medir {SCREEN_W}×{SCREEN_H}, "
                         f"mide {surface.get_width()}×{surface.get_height()}")


# --- Escenario ------------------------------------------------------------------

def _ground_y(level, col: int) -> int:
    """y de la superficie del suelo en `col` (el suelo normal si la columna no sirve)."""
    if 0 <= col < level.width:
        row = level.surface_row(col)
        if row is not None:
            return row * TILE
    return GROUND_Y


def _draw_clouds(surface, level, cam: int) -> None:
    # Posiciones fijas calculadas con la columna, para que el fondo no cambie
    # entre episodios ni entre llamadas.
    for i in range(level.width // 9 + 1):
        x = (i * 9 + (i * 5) % 7) * TILE - cam
        y = HUD_HEIGHT + 8 + ((i * 3) % 4) * 14
        if -56 < x < SCREEN_W:
            pygame.draw.ellipse(surface, WHITE, (x, y + 10, 56, 18))
            pygame.draw.circle(surface, WHITE, (x + 18, y + 12), 11)
            pygame.draw.circle(surface, WHITE, (x + 36, y + 10), 13)


def _draw_hills(surface, level, cam: int) -> None:
    for j in range(level.width // 22 + 1):
        width, height = (96, 44) if j % 2 == 0 else (56, 26)
        x = (j * 22 + 3 + (j * 3) % 5) * TILE
        if not -width < x - cam < SCREEN_W:
            continue
        # Solo sobre suelo llano: encima de un foso quedaría la colina flotando.
        cols = range(x // TILE, (x + width - 1) // TILE + 1)
        if not all(0 <= c < level.width and level.surface_row(c) == ROWS - 2 for c in cols):
            continue
        rect = (x - cam, GROUND_Y - height, width, 2 * height)   # la mitad de abajo la tapa el suelo
        pygame.draw.ellipse(surface, (0, 168, 0), rect)
        pygame.draw.ellipse(surface, (0, 88, 0), rect, 1)


def _draw_castle(surface, level, cam: int) -> None:
    col0 = level.flag_col + CASTLE_COL_OFFSET
    x = col0 * TILE - cam
    if not -CASTLE_COLS * TILE < x < SCREEN_W:
        return
    base = _ground_y(level, col0 + CASTLE_COLS // 2)
    brick, joint = (200, 80, 32), (104, 32, 8)
    body = pygame.Rect(x, base - 48, CASTLE_COLS * TILE, 48)
    tower = pygame.Rect(x + 16, base - 80, 48, 32)
    for rect in (body, tower):
        pygame.draw.rect(surface, brick, rect)
        for k, y in enumerate(range(rect.top, rect.bottom, 8)):
            pygame.draw.line(surface, joint, (rect.left, y + 7), (rect.right - 1, y + 7))
            for jx in range(rect.left + (8 if k % 2 else 0), rect.right, 16):
                pygame.draw.line(surface, joint, (jx, y), (jx, y + 6))
        pygame.draw.rect(surface, joint, rect, 1)
    for mx, my in ((x, base - 56), (x + 72, base - 56),                       # almenas
                   (x + 16, base - 88), (x + 36, base - 88), (x + 56, base - 88)):
        pygame.draw.rect(surface, brick, (mx, my, 8, 8))
        pygame.draw.rect(surface, joint, (mx, my, 8, 8), 1)
    pygame.draw.rect(surface, BLACK, (x + 36, base - 70, 8, 12))              # ventana
    pygame.draw.circle(surface, BLACK, (x + 40, base - 24), 8)                # puerta
    pygame.draw.rect(surface, BLACK, (x + 32, base - 24, 16, 24))


def _draw_flag(surface, level, cam: int) -> None:
    # El mástil está en el borde izquierdo de la columna de la bandera: ahí es
    # donde la caja de Mario "toca" y el motor da la victoria (§6).
    x = level.flag_col * TILE - cam
    if not -TILE < x < SCREEN_W:
        return
    bottom = _ground_y(level, level.flag_col)
    top = max(bottom - FLAG_POLE_TILES * TILE, 0)
    pygame.draw.rect(surface, (152, 232, 120), (x, top, 2, bottom - top))
    pygame.draw.rect(surface, (0, 120, 0), (x + 1, top, 1, bottom - top))
    pygame.draw.circle(surface, (88, 216, 84), (x + 1, top - 1), 3)
    pygame.draw.polygon(surface, WHITE, [(x, top + 4), (x - 14, top + 10), (x, top + 16)])
    pygame.draw.circle(surface, (0, 120, 0), (x - 5, top + 10), 2)


def _draw_tiles(surface, level, cam: int) -> None:
    first = max(cam // TILE, 0)
    last = min((cam + SCREEN_W - 1) // TILE, level.width - 1)
    for row, tiles in enumerate(level.tiles):
        y = row * TILE
        for col in range(first, last + 1):
            code = tiles[col]
            if code == T_EMPTY:
                continue
            if code == T_PIPE:
                image = _pipe_image(level.tiles, row, col)
            else:
                image = sprites.tile_sprite(code)
            surface.blit(image, (col * TILE - cam, y))


def _pipe_image(tiles, row: int, col: int):
    """Qué cuarto de tubería va en (row, col): arriba o cuerpo, izquierda o derecha."""
    top = row == 0 or tiles[row - 1][col] != T_PIPE
    pipes_to_the_left = 0
    while col - pipes_to_the_left - 1 >= 0 and tiles[row][col - pipes_to_the_left - 1] == T_PIPE:
        pipes_to_the_left += 1
    return sprites.pipe_sprite(top, pipes_to_the_left % 2 == 1)


# --- Personajes -----------------------------------------------------------------

def _draw_goombas(surface, game, cam: int) -> None:
    flip = (game.frame // 8) % 2 == 1               # alterna los pies cada 8 frames
    for goomba in game.goombas:
        x = int(goomba.x) - cam
        if not -TILE < x < SCREEN_W:
            continue
        if goomba.squashed:
            # El sprite aplastado mide la mitad: se apoya en la base de la caja.
            surface.blit(sprites.goomba_sprite(squashed=True), (x, int(goomba.y) + TILE // 2))
        else:
            surface.blit(sprites.goomba_sprite(flip=flip), (x, int(goomba.y)))


def _mario_pose(game) -> str:
    mario = game.mario
    if game.dead or not mario.on_ground:
        return "jump"
    if mario.skidding:
        return "skid"
    if abs(mario.vx) > 0.05:
        return "walk" if (int(mario.x) // 6) % 2 == 0 else "idle"   # alterna al avanzar
    return "idle"


def _draw_mario(surface, game, cam: int) -> None:
    mario = game.mario
    sprite = sprites.mario_sprite(_mario_pose(game), mario.facing)
    surface.blit(sprite, (int(mario.x) - cam, int(mario.y)))


# --- HUD ------------------------------------------------------------------------

_GLYPH_SOURCE = {
    "0": "###/#.#/#.#/#.#/###", "1": ".#./##./.#./.#./###", "2": "###/..#/###/#../###",
    "3": "###/..#/###/..#/###", "4": "#.#/#.#/###/..#/..#", "5": "###/#../###/..#/###",
    "6": "###/#../###/#.#/###", "7": "###/..#/..#/.#./.#.", "8": "###/#.#/###/#.#/###",
    "9": "###/#.#/###/..#/###",
    "A": ".#./#.#/###/#.#/#.#", "B": "##./#.#/##./#.#/##.", "C": ".##/#../#../#../.##",
    "D": "##./#.#/#.#/#.#/##.", "E": "###/#../##./#../###", "F": "###/#../##./#../#..",
    "G": ".##/#../#.#/#.#/.##", "H": "#.#/#.#/###/#.#/#.#", "I": "###/.#./.#./.#./###",
    "J": "..#/..#/..#/#.#/.#.", "K": "#.#/#.#/##./#.#/#.#", "L": "#../#../#../#../###",
    "M": "#.#/###/###/#.#/#.#", "N": "##./#.#/#.#/#.#/#.#", "O": ".#./#.#/#.#/#.#/.#.",
    "P": "##./#.#/##./#../#..", "Q": ".#./#.#/#.#/##./.##", "R": "##./#.#/##./#.#/#.#",
    "S": ".##/#../.#./..#/##.", "T": "###/.#./.#./.#./.#.", "U": "#.#/#.#/#.#/#.#/###",
    "V": "#.#/#.#/#.#/#.#/.#.", "W": "#.#/#.#/###/###/#.#", "X": "#.#/#.#/.#./#.#/#.#",
    "Y": "#.#/#.#/.#./.#./.#.", "Z": "###/..#/.#./#../###",
    "%": "#.#/..#/.#./#../#.#", "-": ".../.../###/.../...", ":": ".../.#./.../.#./...",
    " ": ".../.../.../.../...",
}
_GLYPHS = {ch: art.split("/") for ch, art in _GLYPH_SOURCE.items()}


@lru_cache(maxsize=512)
def _text_surface(text: str, color: tuple, scale: int) -> pygame.Surface:
    """Texto en la fuente de píxeles 3×5, con sombra. Se guarda en caché: no modificar."""
    advance = 4 * scale
    surf = pygame.Surface((len(text) * advance + scale, 6 * scale), pygame.SRCALPHA)
    for offset, ink in ((scale, BLACK), (0, color)):             # primero la sombra
        for i, ch in enumerate(text.upper()):
            for gy, row in enumerate(_GLYPHS.get(ch, _GLYPHS[" "])):
                for gx, cell in enumerate(row):
                    if cell == "#":
                        pygame.draw.rect(surf, ink, (i * advance + gx * scale + offset,
                                                     gy * scale + offset, scale, scale))
    return surf


def hud_text(game) -> dict:
    """Los textos del HUD: puntuación, monedas, progreso (0 a 100) y segundos restantes."""
    return {
        "score": f"{game.score:06d}",
        "coins": f"X{game.coins:02d}",
        "progress": f"{int(game.progress * 100)}",
        "time": f"{max(game.time_left, 0) // FPS:03d}",
    }


def _draw_hud(surface, game) -> None:
    text = hud_text(game)
    for x, label, value in ((8, "SCORE", text["score"]), (72, "COINS", text["coins"]),
                            (136, "PROG", text["progress"]), (200, "TIME", text["time"])):
        surface.blit(_text_surface(label, WHITE, 2), (x, 3))
        surface.blit(_text_surface(value, WHITE, 2), (x, 15))
    # Barra de progreso: lo que ha avanzado Mario hacia la bandera (game.progress).
    bar = pygame.Rect(8, 30, SCREEN_W - 16, 4)
    pygame.draw.rect(surface, BLACK, bar.inflate(2, 2))
    pygame.draw.rect(surface, (64, 64, 64), bar)
    pygame.draw.rect(surface, (88, 216, 84), (bar.x, bar.y, int(bar.w * game.progress), bar.h))


# --- Carteles (pausa, fin de partida) ----------------------------------------------

def draw_banner(surface, title: str, lines=(), color: tuple = WHITE) -> None:
    """Cartel centrado sobre la imagen ya dibujada: `title` grande y `lines` debajo.

    Oscurece el fondo para que el texto se lea. Lo usa play.py en la pausa y al
    terminar la partida; el entorno de entrenamiento no lo necesita. La fuente solo
    tiene A-Z, 0-9, % - : y el espacio (sin tildes): cualquier otro carácter sale en blanco.
    """
    _require_screen_size(surface)
    shade = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    shade.fill((0, 0, 0, 120))
    surface.blit(shade, (0, 0))
    rows = [_text_surface(title, color, 3)] + [_text_surface(line, WHITE, 2) for line in lines]
    gap = 8
    y = (SCREEN_H - sum(img.get_height() for img in rows) - gap * (len(rows) - 1)) // 2
    for img in rows:
        surface.blit(img, ((SCREEN_W - img.get_width()) // 2, y))
        y += img.get_height() + gap