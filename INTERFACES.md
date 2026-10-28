# INTERFACES.md — Contrato de interfaces de Mario DRL

Versión: **0.4 (borrador, pendiente de aprobación de las 4 personas)**
Motor que lo implementa: `mario_game.__version__ == "1.0.0"`
Responsable del documento: Persona 1 (motor del juego)

Este documento es el contrato entre los cuatro módulos. Cada persona puede
programar contra él (o contra un stub que lo cumpla) sin esperar el código real
de las demás.

## Convenciones

- **[PROFESOR]**: requisito explícito del documento "Plan de trabajo — Mario DRL".
  No se puede cambiar sin consultar al profesor.
- **[PROPUESTA]**: decisión del equipo para completar lo que el plan no
  especifica. Se puede cambiar con acuerdo de las 4 personas y una nueva versión
  de este archivo (ver "Control de cambios").
- Las unidades son **píxeles** (px) y **frames**. Un frame = 1/60 s.
- Las coordenadas tienen el origen arriba a la izquierda; `x` crece hacia la
  derecha y `y` hacia abajo. La fila 0 es la fila superior del nivel.

---

## 1. Módulos y responsables [PROFESOR]

| Módulo | Responsable |
|---|---|
| `mario_game/game.py`, `mario_game/constants.py` | Persona 1 |
| `mario_game/level.py`, `mario_game/sprites.py`, `mario_game/render.py`, `play.py` | Persona 2 |
| `mario_env.py` | Persona 3 |
| `train.py`, `evaluate.py`, informe | Persona 4 |

Ubicación de `constants.py` dentro de `mario_game/` y del render en
`mario_game/render.py`: [PROPUESTA].

Dirección de las dependencias [PROPUESTA]:

```
constants.py  ←  level.py  ←  game.py  ←  mario_env.py  ←  train.py / evaluate.py
      ↑                          ↑
   render.py  ───(lee campos)────┘
```

- `constants.py` no importa nada del proyecto.
- `level.py` no importa `game.py`.
- `game.py` importa `render.py` solo dentro de `MarioGame.render()`, para que la
  simulación funcione sin gráficos.

---

## 2. Constantes compartidas (`mario_game/constants.py`)

Estas constantes forman parte del contrato. Todos los módulos deben importarlas
en lugar de copiar los números.

| Nombre | Valor | Origen | Significado |
|---|---|---|---|
| `FPS` | `60` | [PROFESOR] | Un `step()` = un frame a 60 FPS |
| `ROWS` | `15` | [PROFESOR] | Filas del nivel |
| `TILE` | `16` | [PROPUESTA] | Lado de un tile en px |
| `SCREEN_W` | `256` | [PROPUESTA] | Ancho de la vista (16 tiles) |
| `SCREEN_H` | `240` | [PROPUESTA] | Alto de la vista (15 tiles) |
| `WALK_MAX_SPEED` | `1.5` | [PROFESOR] | px/frame, velocidad máxima caminando |
| `RUN_MAX_SPEED` | `2.5` | [PROFESOR] | px/frame, velocidad máxima corriendo |
| `MIN_JUMP_TILES` | `4` | [PROFESOR] | Altura mínima garantizada del salto, en tiles |
| `MAX_PIT_TILES` | `3` | [PROFESOR] | Foso más ancho garantizado saltable corriendo |
| `TIME_LIMIT_FRAMES` | `18000` | [PROPUESTA] | 300 s de juego |
| `MARIO_W`, `MARIO_H` | `12`, `16` | [PROPUESTA] | Caja de colisión de Mario |
| `GOOMBA_W`, `GOOMBA_H` | `16`, `16` | [PROPUESTA] | Caja de colisión del goomba |
| `GOOMBA_SPEED` | `0.5` | [PROPUESTA] | px/frame, velocidad por defecto del goomba |

Los parámetros internos de la física (gravedad, aceleración, fricción, impulso
de salto, velocidad máxima de caída) también viven en `constants.py`, pero
**no son contrato**: Persona 1 puede ajustarlos mientras se cumplan los límites
de la sección 5. Nadie debe diseñar el nivel en función de ellos.

### Códigos de tile [PROPUESTA]

| Constante | Código | ¿Sólido? | Uso |
|---|---|---|---|
| `T_EMPTY` | `0` | No | Aire |
| `T_GROUND` | `1` | Sí | Suelo |
| `T_BRICK` | `2` | Sí | Ladrillo (no se rompe: no hay power-ups) |
| `T_QUESTION` | `3` | Sí | Bloque ? con moneda |
| `T_USED` | `4` | Sí | Bloque ? ya golpeado |
| `T_PIPE` | `5` | Sí | Tubería |
| `T_HARD` | `6` | Sí | Bloque duro / escalera |

`SOLID_TILES = {T_GROUND, T_BRICK, T_QUESTION, T_USED, T_PIPE, T_HARD}`

La bandera y el castillo **no son tiles**: la bandera se define con
`Level.flag_col`, y el castillo es solo decorativo (lo dibuja el render).

---

## 3. Formato del nivel (`mario_game/level.py`, Persona 2)

### Campos y métodos [PROFESOR]

`Level.tiles`, `goomba_cols`, `start_col`, `flag_col`, `checkpoints`,
`is_solid(row, col)` y `surface_row(col)`.

### Tipos y semántica [PROPUESTA salvo indicación]

```python
class Level:
    def __init__(self, tiles, goomba_cols, start_col, flag_col, checkpoints): ...

    tiles: list[list[int]]   # tiles[row][col]; len(tiles) == ROWS; todas las filas del mismo ancho
    width: int               # == len(tiles[0]), en tiles
    goomba_cols: list[int]   # columna de aparición de cada goomba
    start_col: int           # columna de inicio por defecto
    flag_col: int            # columna de la bandera
    checkpoints: list[int]   # columnas seguras de inicio, ordenadas; incluye start_col

    def is_solid(self, row: int, col: int) -> bool: ...
    def surface_row(self, col: int) -> int | None: ...

def build_level_1() -> Level: ...   # [PROFESOR] P2.3
def build_level_2() -> Level: ...   # [PROFESOR] P2.10, nivel de generalización (P3.10)
```

Cada llamada a `build_level_N()` debe devolver un objeto `Level` **nuevo** (ver
"Un `Level` por `MarioGame`" en §4). [PROPUESTA]

`is_solid(row, col)` se evalúa en este orden:

1. `row >= ROWS` → `False`. Por abajo es vacío y se puede caer a un foso. [PROFESOR]
2. `col < 0` o `col >= width` → `True`. Por los lados es sólido. [PROFESOR]
3. `row < 0` → `False`. Por encima del mapa hay aire.
4. Si no, `tiles[row][col] in SOLID_TILES`.

`surface_row(col)` devuelve la fila superior de la columna de suelo que empieza
abajo del todo:

- Si `tiles[ROWS-1][col]` no es sólido, la columna es un foso y devuelve `None`.
- Si no, sube mientras haya sólidos y devuelve la fila del sólido más alto de esa
  pila continua. Los bloques flotantes por encima no cuentan.
- Precondición: `0 <= col < width`.

Ejemplo (columna con suelo de 2 tiles y un bloque ? flotante en la fila 9):

```
fila  9: ?   ← no cuenta
fila 10-12: aire
fila 13: suelo   ← surface_row(col) == 13
fila 14: suelo
```

Reglas de diseño del nivel [PROFESOR, P2.1 y P2.4]:

- Ningún obstáculo supera 4 tiles de alto ni 3 tiles de foso.
- Ningún checkpoint cae sobre un foso ni dentro de un obstáculo.

Además, el motor puede **modificar `level.tiles`** (bloque ? → bloque usado)
durante un episodio y lo restaura en `reset()`. [PROPUESTA]

---

## 4. API del juego (`mario_game/game.py`, Persona 1)

### Métodos [PROFESOR]

```python
class MarioGame:
    def __init__(self, level: Level | None = None,
                 goomba_speed: float = GOOMBA_SPEED): ...       # [PROPUESTA]
    def reset(self, start_col: int | None = None) -> None: ...  # [PROFESOR]
    def step(self, left: bool, right: bool, jump: bool, run: bool) -> None: ...  # [PROFESOR]
    def render(self, surface, hud: bool = True) -> None: ...    # [PROFESOR]
```

- `__init__`: con `level=None` usa `build_level_1()`. Llama a `reset()`
  internamente, así que el juego queda listo tras construirlo. [PROPUESTA]
- `reset(start_col=None)`: con `None` usa `level.start_col`. [PROFESOR]
  Detalles [PROPUESTA]:
  - Restaura los tiles originales, el tiempo, la puntuación y las banderas de estado.
  - Coloca a Mario centrado en `start_col`, con los pies sobre
    `surface_row(start_col)`, en reposo y en el suelo.
  - Lanza `ValueError` si `surface_row(start_col)` es `None`.
  - Crea un goomba por cada columna de `goomba_cols` que sea mayor que
    `start_col + 3`, para no morir en el primer frame.
  - **Validación de goombas** [PROPUESTA v0.2, pendiente de aprobación de
    P2, P3 y P4]: si una columna de `goomba_cols` que debe crearse es un foso
    (`surface_row(col) is None`) o está fuera de `[0, level.width)`, `reset()`
    lanza `ValueError`. El motor no ignora en silencio los enemigos mal
    colocados. Garantizar posiciones válidas es responsabilidad del diseño del
    nivel (P2.4). La misma regla de rango se aplica a `start_col`.
  - Ajusta `cam_x` a la posición de Mario.
- `step(left, right, jump, run)`: avanza exactamente un frame. Devuelve `None`; el
  estado se lee de los campos públicos. Si `dead` o `won` ya son `True`, no hace
  nada. [Un frame: PROFESOR. Lo demás: PROPUESTA]
- `render(surface, hud=True)`: dibuja la vista actual (desde `cam_x`) en
  `surface`, que debe medir `SCREEN_W × SCREEN_H`. El escalado lo hace quien
  llama. Delega en `mario_game.render.draw(surface, game, hud)`, que es de
  Persona 2. No modifica el estado del juego. [PROPUESTA]
  Si `mario_game/render.py` aún no existe, lanza `ImportError` con un mensaje
  claro. [PROPUESTA v0.3]
- **Un `Level` por `MarioGame`** [PROPUESTA v0.3, pendiente de P2, P3 y P4]:
  el motor modifica `level.tiles` (bloques ?) y lo restaura en `reset()`. Dos
  juegos no deben compartir el mismo objeto `Level`: cada entorno llama a
  `build_level_1()` por su cuenta.

### Campos públicos [PROFESOR: los nombres; PROPUESTA: los tipos y la semántica]

Estos campos son de solo lectura para los demás módulos, salvo
`goomba_speed`.

| Campo | Tipo | Semántica |
|---|---|---|
| `mario` | `Mario` | Ver abajo |
| `goombas` | `list[Goomba]` | Goombas aún existentes (incluye los aplastados durante su animación) |
| `level` | `Level` | Nivel en uso (con los tiles modificados del episodio actual) |
| `cam_x` | `float` | Borde izquierdo de la vista en px. `0 <= cam_x <= level.width*TILE - SCREEN_W`. Nunca disminuye dentro de un episodio |
| `dead` | `bool` | Mario murió. Se mantiene hasta `reset()` |
| `won` | `bool` | Mario tocó la bandera. Se mantiene hasta `reset()`. Nunca es `True` a la vez que `dead` |
| `death_reason` | `str \| None` | `None`, `"pit"`, `"goomba"` o `"time"` |
| `progress` | `float` | `max_x_alcanzado / (flag_col*TILE)`, en `[0, 1]`. Nunca disminuye dentro de un episodio. Vale `1.0` si `won` |
| `jump_held` | `bool` | Valor de `jump` en el `step()` anterior. Sirve para detectar flancos (pulsar frente a mantener) |

Campos adicionales [PROPUESTA] que necesitan el HUD (P2.7) y las animaciones:

| Campo | Tipo | Semántica |
|---|---|---|
| `score` | `int` | Puntuación |
| `coins` | `int` | Monedas recogidas |
| `time_left` | `int` | Frames restantes (`segundos = time_left // FPS`) |
| `frame` | `int` | Frames transcurridos desde `reset()` |
| `goomba_speed` | `float` | Velocidad de los goombas. Se puede escribir, y se aplica en el siguiente `reset()` (para P3.9) |

### `Mario` y `Goomba` [PROPUESTA]

Ambos heredan de `Body`:

```python
class Body:
    x: float          # px, esquina superior izquierda
    y: float
    w: int            # px
    h: int
    vx: float         # px/frame, positivo hacia la derecha
    vy: float         # px/frame, positivo hacia abajo
    on_ground: bool

class Mario(Body):
    facing: int       # +1 derecha, -1 izquierda
    skidding: bool    # derrapando (para el sprite)

class Goomba(Body):
    direction: int    # +1 / -1
    speed: float      # px/frame
    active: bool      # se ha activado (empieza a moverse)
    squashed: bool    # aplastado: no colisiona; se elimina de la lista a los 30 frames
```

Los goombas con `squashed=True` deben ignorarse en la observación de P3.

`Body` también ofrece las propiedades de solo lectura `right` (`x + w`) y
`bottom` (`y + h`), útiles para el render y la observación. [PROPUESTA]

### Resumen: qué pide el plan y qué añadimos

Lo que pide el plan del profesor **para el motor** y dónde está:

| Exigido por el plan | Dónde |
|---|---|
| `MarioGame.reset(start_col=None)` | `game.py` |
| `MarioGame.step(left, right, jump, run)`, un frame a 60 FPS | `game.py` |
| `MarioGame.render(surface, hud=True)` | `game.py` (delega en `render.py` de P2) |
| Campos `mario`, `goombas`, `level`, `cam_x`, `dead`, `won`, `death_reason`, `progress`, `jump_held` | `game.py` |
| Constantes: tamaño de tile, gravedad, velocidades, salto (P1.3) | `constants.py` |
| `Body` y colisión por ejes (P1.4) | `game.py` |

Lo que **añadimos** nosotros [PROPUESTA] porque el plan no lo detallaba y otro
módulo lo necesita:

| Añadido | Para quién / por qué |
|---|---|
| `MarioGame(level=None, goomba_speed=GOOMBA_SPEED)` | Varios niveles (P3.10) y velocidad variable de goombas (P3.9) |
| Campos `score`, `coins`, `time_left`, `frame` | HUD (P2.7) y animaciones |
| Campo `goomba_speed` (escribible) | Aleatoriedad (P3.9) |
| Clases `Mario`, `Goomba` y sus campos (§4); `Body.right`, `Body.bottom` | Render (P2.7) y observación (P3.3) |
| `MarioGame.GOOMBA_SAFE_COLS = 3` | Regla de aparición de goombas en `reset()` |
| `Level.width` y el constructor `Level(...)` | Motor y tests con niveles mínimos |
| Constantes `TILE`, `SCREEN_W/H`, códigos `T_*`, cajas de colisión, `TIME_LIMIT_FRAMES`, `GOOMBA_SPEED`, `SCORE_*` | Lenguaje común de los 4 módulos |
| `ValueError` en `reset()` y `ImportError` en `render()` | Errores claros en vez de fallos silenciosos (v0.2, v0.3) |
| `mario_game.__version__` | Saber qué versión del motor se usa |

**Fuera del contrato**: todo lo que empieza por `_` (`_update_camera`,
`_check_goomba_contacts`…) y las funciones auxiliares de `game.py`
(`to_cell`, `EPS`, `separate_goombas`, `Body.move_x/move_y/overlaps`,
`Mario.apply_horizontal_input/apply_jump_and_gravity`, `Goomba.update/squash`)
y las constantes marcadas INTERNA en `constants.py`. Se pueden leer, pero
pueden cambiar sin aviso: no programes contra ellas.

---

## 5. Límites físicos garantizados [PROFESOR]

El nivel se diseña con estos límites, no con el motor terminado. La suite de
P1.10 los verifica, y el motor 1.0 no se publica si algún test falla.

| Límite | Valor | Cómo se verifica [PROPUESTA] |
|---|---|---|
| Altura de salto | ≥ 4 tiles | Saltando desde el suelo con `jump` mantenido, los pies suben ≥ 64 px. Además, Mario se sube encima de una columna de 4 tiles |
| Foso saltable corriendo | hasta 3 tiles | En un nivel con suelo, un foso de 3 tiles y suelo, Mario empieza lejos con →+correr y salta en el borde. Aterriza al otro lado sin morir |
| Velocidad caminando | máx. 1.5 px/frame | `abs(vx) <= 1.5` en todos los frames con `run=False`, y se alcanza ≈ 1.5 |
| Velocidad corriendo | máx. 2.5 px/frame | `abs(vx) <= 2.5` en todos los frames con `run=True`, y se alcanza ≈ 2.5 |
| Salto por flanco | — | Mantener `jump` no produce un segundo salto al aterrizar; hay que soltar y volver a pulsar |
| Sin atravesar tiles | — | Ningún cuerpo termina un frame solapado con un tile sólido |

Garantías adicionales del motor [PROPUESTA]:

- `abs(vx) < TILE` y `abs(vy) < TILE` siempre. Así ningún cuerpo atraviesa un
  tile en un solo frame.
- El motor es **determinista**: con el mismo nivel, la misma `goomba_speed` y la
  misma secuencia de entradas, produce el mismo resultado. No usa `random`.
- `step()` no usa pygame. Funciona sin pantalla y sin `pygame.init()`.

---

## 6. Reglas del juego

| Regla | Origen | Detalle [PROPUESTA] |
|---|---|---|
| Muerte por foso | [PROFESOR] | `mario.y >= ROWS*TILE` → `dead=True`, `death_reason="pit"` |
| Muerte por goomba | [PROFESOR] | Contacto con un goomba activo no aplastado que no sea un pisotón → `"goomba"` |
| Muerte por tiempo | [PROFESOR] | `time_left` llega a 0 → `"time"` |
| Pisotón | [PROFESOR] | Mario cae (`vy > 0`) y en el frame anterior sus pies estaban por encima del techo del goomba. El goomba queda aplastado y Mario rebota |
| Bloques ? | [PROFESOR] | Golpe desde abajo → `T_USED`, +1 moneda. Sin champiñones ni power-ups |
| Bandera | [PROFESOR] | La caja de Mario alcanza la columna `flag_col` → `won=True` en ese frame, sin animación |
| Cámara | [PROFESOR] | Nunca retrocede |
| Límite izquierdo | [PROFESOR] | `mario.x >= cam_x`; si choca con ese borde, `vx = 0` |
| Goombas: activación | [PROFESOR] | Se activa cuando `x < cam_x + SCREEN_W + TILE` y permanece activo |
| Goombas: patrulla | [PROFESOR] | Empieza hacia la izquierda; le afecta la gravedad; cae por los bordes y por los fosos; se elimina al salir por abajo |
| Goombas: paredes | [PROFESOR] | Al chocar con un sólido (por ejemplo, una tubería) invierte `direction` |
| Goombas entre sí | [PROFESOR] | Al solaparse, cada uno se aleja del otro: el de la izquierda pasa a `direction = -1` y el de la derecha a `+1` (no se invierte a ciegas, para que no se queden atascados) |

Puntuación [PROPUESTA; el plan no da valores]:

| Evento | Puntos |
|---|---|
| Pisotón a goomba | 100 |
| Moneda de bloque ? | 200 |
| Bandera | 1000 + 10 × segundos restantes |

> La puntuación es solo para el HUD y el modo humano. **No es la recompensa del
> agente** (ver sección 7).

---

## 7. API del entorno (`mario_env.py`, Persona 3)

Esta sección deja fijo lo que P3 y P4 necesitan acordar entre sí. Los detalles
internos son responsabilidad de Persona 3.

| Elemento | Definición | Origen |
|---|---|---|
| Id de registro | `MarioPygame-v0` | [PROFESOR] |
| Acciones | `Discrete(7)`; cada acción se mantiene 4 frames | [PROFESOR] |
| Observación `grid` | Rejilla de tiles alrededor de Mario (sólidos y enemigos) más velocidad, si está en el suelo y enemigos cercanos. Forma exacta: la define P3 | [PROFESOR] / pendiente P3 |
| Observación `pixels` | `uint8` de forma `(84, 84, 1)`, en grises | [PROFESOR] |
| Recompensa | avance en x / 10, −0.01 por paso, −15 al morir, +50 al tocar la bandera | [PROFESOR] |
| Terminación | muerte (foso, goomba o tiempo) o bandera | [PROFESOR] |
| Truncado | 150 pasos sin avanzar; pasos máximos (valor a definir por P3) | [PROFESOR] / pendiente P3 |
| Inicio aleatorio | `random_start`: `reset(start_col)` con una columna de `level.checkpoints` | [PROFESOR] |
| Aleatoriedad | acciones pegajosas y velocidad variable de goombas (vía `game.goomba_speed`) | [PROFESOR] / [PROPUESTA] el mecanismo |
| Varios niveles | parámetro `level` | [PROFESOR] |

Tabla de acciones (traducción a `step(left, right, jump, run)`) [PROFESOR: el orden; PROPUESTA: los índices]:

| Índice | Acción | left | right | jump | run |
|---|---|---|---|---|---|
| 0 | nada | F | F | F | F |
| 1 | → | F | T | F | F |
| 2 | → + saltar | F | T | T | F |
| 3 | → + correr | F | T | F | T |
| 4 | → + correr + saltar | F | T | T | T |
| 5 | saltar | F | F | T | F |
| 6 | ← | T | F | F | F |

Diccionario `info` [PROFESOR: las claves; PROPUESTA: la procedencia]:

| Clave | Tipo | Procedencia |
|---|---|---|
| `x_pos` | `float` | `game.mario.x` |
| `progress` | `float` | `game.progress` |
| `flag_get` | `bool` | `game.won` |
| `dead` | `bool` | `game.dead` |
| `death_reason` | `str \| None` | `game.death_reason` |

Nota para P3: "avance en x" puede calcularse como `Δ mario.x` o como
`Δ max_x`. La elección es de P3, pero debe documentarse aquí cuando se tome.

---

## 8. Comportamiento mínimo de un stub conforme (P3.1, P4.2)

Un stub de `MarioGame` cumple el contrato si:

- Tiene los mismos métodos, firmas y campos públicos, con los tipos de la sección 4.
- `reset()` pone `dead=False`, `won=False` y `death_reason=None`, y deja a Mario
  en `start_col` con `on_ground=True`. `cam_x` y `progress` se calculan como en
  §4: con un `start_col` lejano no valen 0.
- `step(right=True)` aumenta `mario.x` a una velocidad constante ≤ 1.5 px/frame
  (≤ 2.5 con `run=True`). `progress` y `cam_x` se actualizan y no disminuyen.
- Al llegar a `flag_col*TILE`: `won=True`. Al agotar `time_left`: `dead=True`,
  `death_reason="time"`.
- Tras `dead` o `won`, `step()` no cambia nada.
- `render(surface)` puede limitarse a rellenar `surface` con un color y dibujar
  un rectángulo para Mario.

---

## 9. Control de cambios

- Cualquier cambio en este archivo requiere el acuerdo de las 4 personas y un
  aumento de versión. [PROFESOR]
- Cambios compatibles (añadir un campo opcional): versión menor (0.1 → 0.2).
  Cambios incompatibles (renombrar, cambiar un tipo o un significado): versión
  mayor, y avisar a quien tenga un stub.

| Versión | Fecha | Cambio | Aprobado por |
|---|---|---|---|
| 0.1 | 2026-10-27 | Borrador inicial | P1 (pendiente P2, P3, P4) |
| 0.2 | 2026-10-27 | §4 `reset()`: `ValueError` si un goomba o `start_col` cae sobre un foso o fuera del nivel (cambio compatible) | P1 (pendiente P2, P3, P4) |
| 0.3 | 2026-10-27 | §4: `render()` lanza `ImportError` claro si falta `render.py`; un `Level` por `MarioGame` (cambio compatible) | P1 (pendiente P2, P3, P4) |
| 0.4 | 2026-10-27 | Revisión contra el plan: tabla "qué pide el plan y qué añadimos", `build_level_2()`, `Body.right/bottom`, texto del rebote entre goombas y del stub ajustados a la implementación (cambio compatible) | P1 (pendiente P2, P3, P4) |
