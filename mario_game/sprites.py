"""Sprites y tiles de Mario DRL (Persona 2, P2.5).

Todo el pixel-art está definido aquí como texto: cada carácter de una cuadrícula
es un píxel y la paleta dice su color ('.' = transparente). No hay imágenes
externas. Las superficies se crean al primer uso (no al importar) y se guardan
en una caché.

No usa pygame.display ni convert(): funciona sin ventana, como el resto del
juego durante el entrenamiento.

Tamaños (los de las cajas de colisión del contrato, INTERFACES.md §2):
Mario 12×16, goomba 16×16 (aplastado 16×8), tiles 16×16.
"""

import pygame

from mario_game.constants import (GOOMBA_H, GOOMBA_W, MARIO_H, MARIO_W,
                                  T_BRICK, T_GROUND, T_HARD, T_PIPE,
                                  T_QUESTION, T_USED, TILE)

SKY_COLOR = (92, 148, 252)

MARIO_POSES = ("idle", "walk", "jump", "skid")
MARIO_RED = (216, 40, 0)          # los tests buscan este color para ver a Mario

# --- Paletas ------------------------------------------------------------------

MARIO_PALETTE = {
    "r": MARIO_RED,               # gorra y camisa
    "s": (252, 188, 176),         # piel
    "b": (136, 20, 0),            # pelo, bigote y zapatos
    "u": (0, 88, 248),            # peto
    "y": (252, 216, 56),          # botones
}
GOOMBA_PALETTE = {
    "b": (200, 96, 24),           # cuerpo
    "d": (104, 48, 8),            # contorno y pies
    "w": (252, 252, 252),         # ojos
    "k": (0, 0, 0),               # pupilas
    "t": (252, 188, 176),         # pie claro
}

# --- Mario (12×16): cabeza común + 9 filas de cuerpo según la pose ---------------

_MARIO_HEAD = [
    "....rrrrr...",
    "...rrrrrrrrr",
    "...bbbssbs..",
    "..bsbsssbsss",
    "..bsbbsssbss",
    "..bbssssbbbb",
    "....sssssss.",
]

_MARIO_BODY = {
    "idle": [
        "...rrrurrr..",
        "..rrruurrr..",
        ".rrrruuuurrr",
        ".ssruyuuyuss",
        ".sssuuuuusss",
        "..uuuuuuuu..",
        "..uuu..uuu..",
        ".bbbb..bbbb.",
        "bbbbb..bbbbb",
    ],
    "walk": [
        "...rrrurrr..",
        "..rrruurrr..",
        ".rrrruuuurrr",
        ".sssuyuuyus.",
        "..suuuuuuus.",
        "...uuuuuuu..",
        "...uuuu.uu..",
        "..bbbbb.bbb.",
        "..bbb....bbb",
    ],
    "jump": [
        "...rrrurrrss",
        "..rrruurrrss",
        ".rrrruuuurss",
        "sssruyuuyus.",
        "ss.uuuuuuu..",
        "...uuuuuuuu.",
        "..uuuu..uuu.",
        ".bbbb....bbb",
        "bbbb......bb",
    ],
    "skid": [
        "...rrrurrr..",
        "..rrruurrr..",
        "ssrrruuurrr.",
        "sssruyuuyus.",
        ".ssuuuuuuus.",
        "..uuuuuuuu..",
        "..uuuu.uuu..",
        "..bbbbbbbb..",
        ".bbbbbbbbbb.",
    ],
}

# --- Goomba (16×16): la mitad izquierda se refleja; los pies son distintos en
# cada pie para que, al voltearlo, parezca que camina ------------------------------

_GOOMBA_HALF = [
    "......dd",
    "....dddb",
    "..ddbbbb",
    ".dbbbbbb",
    ".dbbbbbb",
    "dbbbbbbb",
    "dbbddbbb",
    "dbwwwdbb",
    "dbwwkkbb",
    "dbwwkkbb",
    ".dbbbbbb",
    "..ddbttt",
    "....tttt",
]
_GOOMBA_FEET = [
    "..ddddd..ddddd..",
    ".dddddd...ddddd.",
    "dddddd.....ddddd",
]
_GOOMBA_SQUASHED_HALF = [
    ".....ddd",
    "..dddbbb",
    ".dbbbbbb",
    "dbbwkbbb",
    "dbbbbbbb",
    ".dbbtttt",
    "..ddtttt",
    "dddddddd",
]


def _mirror(half_rows: list[str]) -> list[str]:
    return [row + row[::-1] for row in half_rows]


MARIO_ART = {pose: _MARIO_HEAD + body for pose, body in _MARIO_BODY.items()}
GOOMBA_ART = _mirror(_GOOMBA_HALF) + _GOOMBA_FEET
GOOMBA_SQUASHED_ART = _mirror(_GOOMBA_SQUASHED_HALF)


def surface_from_art(rows: list[str], palette: dict) -> pygame.Surface:
    """Superficie con transparencia a partir de filas de texto."""
    width = len(rows[0])
    surf = pygame.Surface((width, len(rows)), pygame.SRCALPHA)
    for y, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"fila {y} de un sprite mide {len(row)}, se esperaba {width}")
        for x, ch in enumerate(row):
            if ch != ".":
                if ch not in palette:
                    raise ValueError(f"color {ch!r} sin definir en la paleta (fila {y})")
                surf.set_at((x, y), palette[ch])
    return surf


# --- Tiles (16×16), dibujados con rectángulos ---------------------------------------

def _fill(surf, color, rect):
    pygame.draw.rect(surf, color, rect)


def _bricks(base, line, course_h) -> pygame.Surface:
    """Ladrillos: hiladas de `course_h` px con las juntas desplazadas."""
    surf = pygame.Surface((TILE, TILE))
    surf.fill(base)
    for course, y0 in enumerate(range(0, TILE, course_h)):
        _fill(surf, line, (0, y0 + course_h - 1, TILE, 1))                 # junta horizontal
        joints = (7,) if course % 2 == 0 else (3, 11)
        for x in joints:
            _fill(surf, line, (x, y0, 1, course_h - 1))                    # junta vertical
    return surf


def _ground_tile():
    surf = _bricks((176, 100, 40), (80, 40, 8), 8)
    _fill(surf, (224, 156, 84), (0, 0, TILE, 1))                           # brillo arriba
    return surf


def _brick_tile():
    return _bricks((216, 96, 40), (104, 32, 8), 4)


_QUESTION_GLYPH = [          # cada # se dibuja como un cuadrado de 2×2 px
    ".##.",
    "#..#",
    "..#.",
    ".#..",
    "....",
    ".#..",
]


def _question_tile(used: bool):
    base, edge, mark = (((156, 100, 52), (80, 40, 8), (80, 40, 8)) if used
                        else ((252, 184, 56), (136, 76, 0), (136, 76, 0)))
    surf = pygame.Surface((TILE, TILE))
    surf.fill(edge)
    _fill(surf, base, (1, 1, TILE - 2, TILE - 2))
    for x, y in ((2, 2), (TILE - 3, 2), (2, TILE - 3), (TILE - 3, TILE - 3)):
        surf.set_at((x, y), edge)                                          # remaches
    if not used:
        for gy, row in enumerate(_QUESTION_GLYPH):
            for gx, ch in enumerate(row):
                if ch == "#":
                    _fill(surf, mark, (4 + gx * 2, 2 + gy * 2, 2, 2))
    return surf


def _hard_tile():
    surf = pygame.Surface((TILE, TILE))
    surf.fill((176, 176, 188))
    _fill(surf, (236, 236, 244), (0, 0, TILE, 2))
    _fill(surf, (236, 236, 244), (0, 0, 2, TILE))
    _fill(surf, (88, 88, 104), (0, TILE - 2, TILE, 2))
    _fill(surf, (88, 88, 104), (TILE - 2, 0, 2, TILE))
    _fill(surf, (140, 140, 156), (4, 4, TILE - 8, TILE - 8))
    return surf


def _pipe_tile(top: bool, right: bool):
    """Un cuarto de tubería de 2×2 tiles: borde superior con labio y cuerpo."""
    green, light, shade, dark = (0, 168, 0), (136, 232, 72), (0, 120, 0), (0, 56, 0)
    # El cuerpo es 2 px más estrecho que el labio, por los lados exteriores.
    x0 = 0 if top else (0 if right else 2)
    x1 = TILE if top else (TILE - 2 if right else TILE)
    surf = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
    _fill(surf, green, (x0, 0, x1 - x0, TILE))
    if right:
        _fill(surf, shade, (x1 - 6, 0, 5, TILE))                           # sombra
        _fill(surf, dark, (x1 - 1, 0, 1, TILE))                            # contorno
    else:
        _fill(surf, dark, (x0, 0, 1, TILE))
        _fill(surf, light, (x0 + 3, 0, 2, TILE))                           # brillo
    if top:
        _fill(surf, dark, (x0, 0, x1 - x0, 1))
        _fill(surf, dark, (x0, TILE - 1, x1 - x0, 1))
    return surf


# --- API pública (con caché) ---------------------------------------------------------

_cache: dict = {}


def _cached(key, build):
    if key not in _cache:
        _cache[key] = build()
    return _cache[key]


def mario_sprite(pose: str, facing: int = 1) -> pygame.Surface:
    """Mario en una de MARIO_POSES. facing = +1 (derecha) o -1 (izquierda)."""
    if pose not in MARIO_ART:
        raise ValueError(f"pose desconocida: {pose!r} (usa una de {MARIO_POSES})")
    right = _cached(("mario", pose), lambda: surface_from_art(MARIO_ART[pose], MARIO_PALETTE))
    if facing >= 0:
        return right
    return _cached(("mario", pose, "flip"), lambda: pygame.transform.flip(right, True, False))


def goomba_sprite(squashed: bool = False, flip: bool = False) -> pygame.Surface:
    """Goomba caminando (16×16) o aplastado (16×8). flip alterna los pies."""
    if squashed:
        return _cached(("goomba", "squashed"),
                       lambda: surface_from_art(GOOMBA_SQUASHED_ART, GOOMBA_PALETTE))
    walk = _cached(("goomba", "walk"), lambda: surface_from_art(GOOMBA_ART, GOOMBA_PALETTE))
    if not flip:
        return walk
    return _cached(("goomba", "walk", "flip"), lambda: pygame.transform.flip(walk, True, False))


_TILE_BUILDERS = {
    T_GROUND: _ground_tile,
    T_BRICK: _brick_tile,
    T_QUESTION: lambda: _question_tile(False),
    T_USED: lambda: _question_tile(True),
    T_HARD: _hard_tile,
}


def tile_sprite(code: int) -> pygame.Surface:
    """Tile de un código del contrato (no sirve para T_PIPE: usa pipe_sprite)."""
    if code == T_PIPE:
        raise ValueError("la tubería se dibuja con pipe_sprite(top, right)")
    if code not in _TILE_BUILDERS:
        raise ValueError(f"código de tile sin sprite: {code}")
    return _cached(("tile", code), _TILE_BUILDERS[code])


def pipe_sprite(top: bool, right: bool) -> pygame.Surface:
    """Un cuarto de tubería: top = borde superior; right = mitad derecha."""
    return _cached(("pipe", top, right), lambda: _pipe_tile(top, right))


def build_sheet(scale: int = 4) -> pygame.Surface:
    """Hoja con todos los sprites ampliados (para comprobarlos a ojo, P2.5)."""
    pad = 6
    items = []
    for pose in MARIO_POSES:
        items.append(mario_sprite(pose, 1))
    items.append(mario_sprite("walk", -1))
    items += [goomba_sprite(False, False), goomba_sprite(False, True), goomba_sprite(True)]
    tiles = [tile_sprite(c) for c in (T_GROUND, T_BRICK, T_QUESTION, T_USED, T_HARD)]
    pipes = [pipe_sprite(True, False), pipe_sprite(True, True),
             pipe_sprite(False, False), pipe_sprite(False, True)]

    def row_width(row):
        return sum(s.get_width() * scale + pad for s in row) + pad

    rows = [items, tiles + pipes]
    width = max(row_width(r) for r in rows)
    height = sum(max(s.get_height() for s in r) * scale + pad for r in rows) + pad
    sheet = pygame.Surface((width, height))
    sheet.fill(SKY_COLOR)
    y = pad
    for row in rows:
        x = pad
        for s in row:
            big = pygame.transform.scale(s, (s.get_width() * scale, s.get_height() * scale))
            sheet.blit(big, (x, y + (max(t.get_height() for t in row) - s.get_height()) * scale))
            x += big.get_width() + pad
        y += max(s.get_height() for s in row) * scale + pad
    return sheet


# Tamaños que debe respetar el render (las cajas de colisión del contrato).
MARIO_SIZE = (MARIO_W, MARIO_H)
GOOMBA_SIZE = (GOOMBA_W, GOOMBA_H)
