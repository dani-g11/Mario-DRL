"""Constantes compartidas del juego Mario DRL.

Este módulo no importa nada del proyecto: es la única fuente de números que
comparten el motor (P1), el nivel y el render (P2) y el entorno (P3).

Unidades: píxeles (px) y frames (1 frame = 1/60 s). Velocidades en px/frame,
aceleraciones en px/frame². Eje y hacia abajo.

Las constantes marcadas "CONTRATO" están fijadas en INTERFACES.md y no se
cambian sin acuerdo de las 4 personas. Las marcadas "INTERNA" son ajustes de la
física que Persona 1 puede cambiar mientras se cumplan los límites del contrato.
"""

# --- Pantalla y rejilla (CONTRATO) ------------------------------------------
FPS = 60
TILE = 16                 # lado de un tile en px
ROWS = 15                 # filas del nivel
SCREEN_W = 256            # 16 tiles de ancho
SCREEN_H = 240            # 15 tiles de alto (= ROWS * TILE)

# --- Límites garantizados (CONTRATO) ----------------------------------------
WALK_MAX_SPEED = 1.5      # px/frame
RUN_MAX_SPEED = 2.5       # px/frame
MIN_JUMP_TILES = 4        # el salto sube al menos 4 tiles
MAX_PIT_TILES = 3         # un foso de 3 tiles se salta corriendo

# --- Cajas de colisión (CONTRATO) -------------------------------------------
MARIO_W = 12
MARIO_H = 16
GOOMBA_W = 16
GOOMBA_H = 16

# --- Reglas (CONTRATO) ------------------------------------------------------
TIME_LIMIT_FRAMES = 300 * FPS   # 300 s de juego
GOOMBA_SPEED = 0.5              # px/frame, valor por defecto

# --- Códigos de tile (CONTRATO) ---------------------------------------------
T_EMPTY = 0
T_GROUND = 1
T_BRICK = 2
T_QUESTION = 3
T_USED = 4
T_PIPE = 5
T_HARD = 6
SOLID_TILES = frozenset({T_GROUND, T_BRICK, T_QUESTION, T_USED, T_PIPE, T_HARD})

# --- Física horizontal (INTERNA) --------------------------------------------
ACCEL = 0.08              # px/frame² al pulsar una dirección
FRICTION = 0.06           # px/frame² de frenado sin pulsar nada (en el suelo)
SKID_DECEL = 0.15         # px/frame² al pulsar la dirección contraria (derrape)

# --- Puntuación (CONTRATO, valores propuestos en INTERFACES.md §6) ----------
# Solo para el HUD y el modo humano: NO es la recompensa del agente (§7).
SCORE_STOMP = 100
SCORE_COIN = 200
SCORE_FLAG = 1000
SCORE_PER_SECOND_LEFT = 10

# --- Reglas (INTERNA) -------------------------------------------------------
STOMP_BOUNCE = 3.0        # px/frame hacia arriba tras pisar un goomba
SQUASH_FRAMES = 30        # frames que se ve el goomba aplastado antes de borrarlo

# --- Física vertical (INTERNA) ----------------------------------------------
JUMP_SPEED = 4.5          # px/frame, velocidad inicial hacia arriba
GRAVITY_HOLD = 0.125      # px/frame² subiendo con el botón de salto pulsado
GRAVITY = 0.45            # px/frame² en el resto de casos
MAX_FALL_SPEED = 4.5      # px/frame, velocidad terminal de caída
