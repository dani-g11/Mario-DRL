"""Utilidades compartidas por los tests del motor.

FakeLevel es un doble de prueba que cumple el contrato de Level descrito en
INTERFACES.md §3. NO sustituye a mario_game/level.py (Persona 2): solo permite
probar el motor con niveles mínimos escritos en ASCII sin depender de ese
archivo.
"""

from mario_game.constants import (ROWS, SOLID_TILES, T_BRICK, T_EMPTY,
                                  T_GROUND, T_HARD, T_PIPE, T_QUESTION, T_USED)
from mario_game.game import EPS, to_cell

CHAR_TO_TILE = {
    ".": T_EMPTY, "#": T_GROUND, "B": T_BRICK, "?": T_QUESTION,
    "U": T_USED, "P": T_PIPE, "H": T_HARD,
}


class FakeLevel:
    def __init__(self, tiles, goomba_cols=(), start_col=0, flag_col=None,
                 checkpoints=None):
        self.tiles = tiles
        self.width = len(tiles[0])
        self.goomba_cols = list(goomba_cols)
        self.start_col = start_col
        self.flag_col = self.width - 1 if flag_col is None else flag_col
        self.checkpoints = [start_col] if checkpoints is None else list(checkpoints)

    def is_solid(self, row: int, col: int) -> bool:
        if row >= ROWS:
            return False
        if col < 0 or col >= self.width:
            return True
        if row < 0:
            return False
        return self.tiles[row][col] in SOLID_TILES

    def surface_row(self, col: int):
        if not self.is_solid(ROWS - 1, col):
            return None
        row = ROWS - 1
        while row > 0 and self.is_solid(row - 1, col):
            row -= 1
        return row


def make_level(rows: list[str], **kwargs) -> FakeLevel:
    """Crea un FakeLevel a partir de filas ASCII.

    Las filas dadas se alinean abajo; por encima se rellena con aire hasta
    completar ROWS filas. Ejemplo de suelo plano de 2 tiles de alto:
        make_level(["##########", "##########"])
    """
    width = len(rows[0])
    assert all(len(r) == width for r in rows), "todas las filas deben medir igual"
    padded = ["." * width] * (ROWS - len(rows)) + rows
    tiles = [[CHAR_TO_TILE[ch] for ch in r] for r in padded]
    return FakeLevel(tiles, **kwargs)


def overlaps_solid(body, level) -> bool:
    """True si la caja del cuerpo comparte área con algún tile sólido."""
    for row in range(to_cell(body.y), to_cell(body.bottom - EPS) + 1):
        for col in range(to_cell(body.x), to_cell(body.right - EPS) + 1):
            if level.is_solid(row, col):
                return True
    return False
