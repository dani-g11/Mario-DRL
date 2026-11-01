"""Niveles de Mario DRL (Persona 2): la clase `Level` y `build_level_N()`.

El contrato está en INTERFACES.md §3. Este módulo solo importa `constants`:
no importa `game.py` ni pygame, para que el entorno (P3) pueda simular sin
gráficos.

Los niveles se escriben en ASCII con las mismas letras que usan los tests del
motor (tests/helpers.py):

    .  aire           #  suelo             B  ladrillo
    ?  bloque ?       U  bloque ? usado    P  tubería
    H  bloque duro / escalera

Para ver un nivel: `python -m mario_game.level`
"""

from mario_game.constants import (ROWS, SOLID_TILES, T_BRICK, T_EMPTY,
                                  T_GROUND, T_HARD, T_PIPE, T_QUESTION, T_USED)

CHAR_TO_TILE = {
    ".": T_EMPTY, "#": T_GROUND, "B": T_BRICK, "?": T_QUESTION,
    "U": T_USED, "P": T_PIPE, "H": T_HARD,
}
TILE_TO_CHAR = {tile: ch for ch, tile in CHAR_TO_TILE.items()}


# --- Conversión ASCII <-> tiles ---------------------------------------------

def tiles_from_ascii(rows: list[str]) -> list[list[int]]:
    """Convierte filas ASCII en la matriz de códigos de tile.

    Las filas se alinean abajo y por encima se rellena con aire hasta completar
    ROWS filas, así basta escribir las filas que tienen algo.
    """
    if not rows or len(rows) > ROWS:
        raise ValueError(f"se esperan entre 1 y {ROWS} filas, hay {len(rows)}")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("todas las filas ASCII deben medir lo mismo")
    padded = ["." * width] * (ROWS - len(rows)) + list(rows)
    try:
        return [[CHAR_TO_TILE[ch] for ch in row] for row in padded]
    except KeyError as exc:
        raise ValueError(f"carácter de tile desconocido: {exc.args[0]!r}") from None


def join_tramos(tramos: list[list[str]]) -> list[str]:
    """Une tramos (bloques de filas ASCII) de izquierda a derecha.

    Cada tramo puede listar solo sus filas de abajo; se alinean con el fondo del
    nivel. Devuelve ROWS filas ASCII.
    """
    padded = []
    for tramo in tramos:
        width = len(tramo[0])
        if any(len(row) != width for row in tramo):
            raise ValueError("las filas de un tramo deben medir lo mismo")
        padded.append(["." * width] * (ROWS - len(tramo)) + list(tramo))
    return ["".join(parts) for parts in zip(*padded)]


# --- Level (INTERFACES.md §3) -----------------------------------------------

class Level:
    """Un nivel: matriz de tiles más las posiciones de inicio, bandera y goombas."""

    def __init__(self, tiles, goomba_cols, start_col, flag_col, checkpoints):
        tiles = [list(row) for row in tiles]    # listas: el motor las modifica
        if len(tiles) != ROWS:
            raise ValueError(f"el nivel debe tener {ROWS} filas, tiene {len(tiles)}")
        width = len(tiles[0])
        if width == 0 or any(len(row) != width for row in tiles):
            raise ValueError("todas las filas deben medir lo mismo y no estar vacías")
        self.tiles = tiles                       # tiles[row][col]
        self.width = width                       # ancho en tiles
        self.goomba_cols = list(goomba_cols)     # columna de cada goomba
        self.start_col = start_col               # inicio por defecto
        self.flag_col = flag_col                 # columna de la bandera
        self.checkpoints = list(checkpoints)     # inicios seguros, ordenados

    def is_solid(self, row: int, col: int) -> bool:
        """True si la celda es sólida. El orden de las reglas es el del contrato:
        por abajo es vacío (se cae a un foso), por los lados es pared."""
        if row >= ROWS:
            return False
        if col < 0 or col >= self.width:
            return True
        if row < 0:
            return False
        return self.tiles[row][col] in SOLID_TILES

    def surface_row(self, col: int) -> int | None:
        """Fila del sólido más alto de la pila continua que empieza abajo del todo.

        None si la columna es un foso (la fila de abajo no es sólida). Los
        bloques flotantes no cuentan. Precondición: 0 <= col < width.
        """
        if not self.is_solid(ROWS - 1, col):
            return None
        row = ROWS - 1
        while row > 0 and self.is_solid(row - 1, col):
            row -= 1
        return row

    def to_ascii(self, markers: bool = False, ruler: bool = False,
                 start: int = 0, stop: int | None = None) -> str:
        """El nivel (o las columnas start..stop) como texto.

        markers: añade una línea con S inicio, C checkpoints, g goombas y F bandera.
        ruler: añade dos líneas con los números de columna (decenas y unidades).
        """
        stop = self.width if stop is None else min(stop, self.width)
        lines = []
        if ruler:
            lines.append("".join(str(c // 10 % 10) if c % 10 == 0 else " "
                                 for c in range(start, stop)))
            lines.append("".join(str(c % 10) for c in range(start, stop)))
        lines += ["".join(TILE_TO_CHAR[t] for t in row[start:stop])
                  for row in self.tiles]
        if markers:
            marks = {c: "g" for c in self.goomba_cols}
            marks.update({c: "C" for c in self.checkpoints})
            marks[self.start_col] = "S"
            marks[self.flag_col] = "F"
            lines.append("".join(marks.get(c, " ") for c in range(start, stop)))
        return "\n".join(lines)


# --- Nivel 1 ------------------------------------------------------------------
# Suelo de 2 tiles (superficie en la fila 13). Lo que hay en cada tramo:
#   0: bloques ?, un goomba y una tubería de 2.   1: foso de 2 y tubería de 3.
#   2: tubería de 2, foso de 3 y pirámide de bloques duros (altura máxima 4).
#   3: foso de 2, foso de 3 (con 10 tiles de suelo entre los dos) y una tubería de 2.
#   4: escalera de 4 escalones y la bandera, con sitio a la derecha para el castillo.
# Límites del contrato: ningún obstáculo pasa de 4 tiles de alto ni de 3 de foso.
# Los bloques flotantes están en la fila 9 (3 tiles libres encima del suelo) y
# solo en un tramo sin enemigos ni obstáculos cerca: al saltar, la cabeza de
# Mario chocaría con ellos y el salto sería más corto.

TRAMO_ANCHO = 40

TRAMOS_NIVEL_1 = [
    # tramo 0 (cols 0-39)
    [
        ".....?B?B?..............................",
        "........................................",
        "..............................PP........",
        "..............................PP........",
        "########################################",
        "########################################",
    ],
    # tramo 1 (cols 40-79)
    [
        "......................PP................",
        "......................PP................",
        "......................PP................",
        "############..##########################",
        "############..##########################",
    ],
    # tramo 2 (cols 80-119)
    [
        "...............................HH.......",
        "..............................HHHH......",
        "..........PP.................HHHHHH.....",
        "..........PP................HHHHHHHH....",
        "####################...#################",
        "####################...#################",
    ],
    # tramo 3 (cols 120-159)
    [
        "..............................PP........",
        "..............................PP........",
        "########..##########...#################",
        "########..##########...#################",
    ],
    # tramo 4 (cols 160-199)
    [
        "...................H....................",
        "..................HH....................",
        ".................HHH....................",
        "................HHHH....................",
        "########################################",
        "########################################",
    ],
]

LEVEL_1_START_COL = 2
LEVEL_1_FLAG_COL = 186
# Cada goomba tiene suelo llano de 6 columnas a su izquierda a 8 a su derecha: se
# encuentra con Mario, y este aterriza tras el pisotón, lejos de fosos y tuberías.
# Van separados al menos 10 columnas para no tener que pisar dos seguidos, y
# lejos del aterrizaje de los saltos de tubería (si no, Mario cae sobre ellos).
LEVEL_1_GOOMBA_COLS = [18, 43, 76, 164]
# Inicios seguros: suelo llano, con espacio para coger carrerilla antes del
# siguiente obstáculo (lo comprueba tests/test_level.py).
LEVEL_1_CHECKPOINTS = [2, 36, 55, 65, 94, 117, 134, 144, 156]


def build_level_1() -> Level:
    """Nivel 1. Devuelve un `Level` nuevo en cada llamada (INTERFACES.md §3)."""
    return Level(
        tiles=tiles_from_ascii(join_tramos(TRAMOS_NIVEL_1)),
        goomba_cols=LEVEL_1_GOOMBA_COLS,
        start_col=LEVEL_1_START_COL,
        flag_col=LEVEL_1_FLAG_COL,
        checkpoints=LEVEL_1_CHECKPOINTS,
    )


if __name__ == "__main__":
    level = build_level_1()
    print(f"Nivel 1: {level.width} columnas, {len(level.goomba_cols)} goombas, "
          f"bandera en la columna {level.flag_col}")
    for first in range(0, level.width, TRAMO_ANCHO):
        last = min(first + TRAMO_ANCHO, level.width) - 1
        print(f"\n--- columnas {first}-{last} ---")
        print(level.to_ascii(markers=True, ruler=True, start=first, stop=last + 1))
