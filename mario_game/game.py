"""Motor del juego Mario DRL (Persona 1).

Etapa C.1: representación física de los personajes.
Etapa C.2: colisión por ejes contra tiles sólidos.
Etapa C.3: movimiento horizontal y esqueleto de MarioGame (reset/step).
Etapa C.4: salto por flanco, gravedad variable y velocidad terminal.
Etapa C.5: cámara que no retrocede y límite izquierdo de Mario.
Etapa C.6: goombas (activación, patrulla, giro en paredes, rebote entre ellos).
Etapa C.7: reglas (pisotón, muertes, bloques ?, bandera, puntuación, progreso).
Etapa C.8: render() (delegado en mario_game/render.py) y optimizaciones.
"""

import importlib

from mario_game.constants import (ACCEL, FPS, FRICTION, GOOMBA_H,
                                  GOOMBA_SPEED, GOOMBA_W, GRAVITY,
                                  GRAVITY_HOLD, JUMP_SPEED, MARIO_H, MARIO_W,
                                  MAX_FALL_SPEED, ROWS, RUN_MAX_SPEED,
                                  SCORE_COIN, SCORE_FLAG, SCORE_PER_SECOND_LEFT,
                                  SCORE_STOMP, SCREEN_W, SKID_DECEL,
                                  SQUASH_FRAMES, STOMP_BOUNCE, T_QUESTION,
                                  T_USED, TILE, TIME_LIMIT_FRAMES,
                                  WALK_MAX_SPEED)

# Margen para calcular el último píxel ocupado por una caja. Una caja con
# right = 32.0 ocupa hasta el píxel 31.999..., que está en la columna 1, no en
# la 2. Sin este margen, un cuerpo apoyado justo al lado de una pared
# "detectaría" la pared aunque no la toque.
EPS = 1e-6


def to_cell(px: float) -> int:
    """Convierte una coordenada en px al índice de tile que la contiene.

    Se usa // (división entera hacia abajo) y no int(px / TILE): int() trunca
    hacia cero, y para px = -0.5 daría 0 en vez de -1.
    """
    return int(px // TILE)


class Body:
    """Caja rectangular que se mueve: la base de Mario y de los goombas.

    (x, y) es la esquina superior izquierda en px; el eje y crece hacia abajo.
    vx, vy están en px/frame. on_ground indica si el cuerpo está apoyado en
    un tile sólido al terminar el último frame.
    """

    # __slots__ fija la lista de atributos: escribir mal un nombre
    # (p. ej. body.on_gruond = True) da error en vez de crear un atributo nuevo
    # en silencio, y el acceso es algo más rápido.
    __slots__ = ("x", "y", "w", "h", "vx", "vy", "on_ground")

    def __init__(self, x: float, y: float, w: int, h: int) -> None:
        self.x = float(x)
        self.y = float(y)
        self.w = w
        self.h = h
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False

    # Bordes de la caja. right y bottom son exclusivos: un cuerpo con x=0, w=12
    # ocupa los píxeles 0..11 y right vale 12.
    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def overlaps(self, other: "Body") -> bool:
        """True si las dos cajas comparten área (tocarse en el borde no cuenta)."""
        # Se escribe con x + w en vez de las propiedades right/bottom: es la
        # función más llamada del motor y así evita 4 llamadas por comparación.
        return (self.x < other.x + other.w and other.x < self.x + self.w
                and self.y < other.y + other.h and other.y < self.y + self.h)

    # --- Colisión por ejes ---------------------------------------------------
    # Se mueve primero en x y se corrige; después en y y se corrige. Al separar
    # los ejes, cada corrección sabe en qué dirección empujar el cuerpo
    # (contra la pared o contra el suelo) sin tener que adivinarlo.
    # Funciona porque ninguna velocidad llega a TILE (ver test_constants): en un
    # frame el cuerpo solo puede entrar en la primera fila/columna de tiles.

    # Rendimiento: move_x y move_y son lo más ejecutado del motor (una vez por
    # cuerpo y frame). Por eso escriben `int(px // TILE)` y `self.x + self.w`
    # directamente en lugar de llamar a to_cell() y a las propiedades
    # right/bottom: el resultado es idéntico y ahorra llamadas a funciones.

    def move_x(self, level) -> bool:
        """Avanza vx en x y lo saca de cualquier sólido. True si chocó."""
        vx = self.vx
        if vx == 0:
            return False
        self.x += vx
        if vx > 0:
            col = int((self.x + self.w - EPS) // TILE)   # columna del borde derecho
        else:
            col = int(self.x // TILE)                    # columna del borde izquierdo
        top_row = int(self.y // TILE)
        bottom_row = int((self.y + self.h - EPS) // TILE)
        for row in range(top_row, bottom_row + 1):
            if level.is_solid(row, col):
                if vx > 0:
                    self.x = col * TILE - self.w  # pegado a la cara izquierda del tile
                else:
                    self.x = (col + 1) * TILE     # pegado a la cara derecha del tile
                self.vx = 0.0
                return True
        return False

    def move_y(self, level) -> list[tuple[int, int]]:
        """Avanza vy en y, lo saca de cualquier sólido y actualiza on_ground.

        Devuelve las celdas (row, col) golpeadas con la cabeza al subir; la
        etapa C.7 las usa para los bloques ?. Al caer o sin choque devuelve [].
        """
        self.on_ground = False
        vy = self.vy
        if vy == 0:
            return []
        self.y += vy
        left_col = int(self.x // TILE)
        right_col = int((self.x + self.w - EPS) // TILE)
        if vy > 0:
            row = int((self.y + self.h - EPS) // TILE)   # fila de los pies
            for col in range(left_col, right_col + 1):
                if level.is_solid(row, col):
                    self.y = row * TILE - self.h         # apoyado sobre el tile
                    self.on_ground = True
                    self.vy = 0.0
                    break
            return []
        row = int(self.y // TILE)                         # fila de la cabeza
        hits = [(row, col) for col in range(left_col, right_col + 1)
                if level.is_solid(row, col)]
        if hits:
            self.y = (row + 1) * TILE                     # cabeza contra el techo
            self.vy = 0.0
        return hits


class Mario(Body):
    """Mario: un Body con orientación y estado de derrape (para los sprites)."""

    __slots__ = ("facing", "skidding")

    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y, MARIO_W, MARIO_H)
        self.facing = 1        # +1 mira a la derecha, -1 a la izquierda
        self.skidding = False

    def apply_horizontal_input(self, left: bool, right: bool, run: bool) -> None:
        """Actualiza vx según los botones. No mueve a Mario (eso es move_x).

        Tres casos:
        - Pulsa hacia donde ya se mueve (o está quieto): acelera con ACCEL.
        - Pulsa la dirección contraria: derrape, frena con SKID_DECEL (más
          fuerte que la fricción) y después empieza a moverse al otro lado.
        - No pulsa nada (o pulsa ambas): fricción en el suelo; en el aire
          conserva la inercia, como en el Mario original.
        Al final se limita |vx| al máximo de caminar o correr según `run`.
        """
        direction = int(right) - int(left)       # +1, -1 o 0 (ambas = 0)
        self.skidding = False

        if direction != 0:
            self.facing = direction
            if self.vx * direction < 0:          # signos opuestos: derrape
                self.vx += direction * SKID_DECEL
                self.skidding = self.on_ground
            else:
                self.vx += direction * ACCEL
        elif self.on_ground:
            # Fricción sin pasarse de cero (si no, Mario temblaría a los lados).
            speed = max(abs(self.vx) - FRICTION, 0.0)
            self.vx = speed if self.vx > 0 else -speed

        # Límite de velocidad del contrato. Se aplica en cada frame, también al
        # soltar "correr": así |vx| <= 1.5 siempre que run sea False.
        max_speed = RUN_MAX_SPEED if run else WALK_MAX_SPEED
        self.vx = max(-max_speed, min(self.vx, max_speed))

    def apply_jump_and_gravity(self, jump: bool, jump_held: bool) -> None:
        """Actualiza vy: inicia el salto si corresponde y aplica la gravedad.

        - Salto por flanco: solo despega si `jump` está pulsado AHORA y NO lo
          estaba en el frame anterior (`jump_held`), y además está en el suelo.
          Mantener el botón no encadena saltos al aterrizar.
        - Gravedad variable: mientras sube con el botón pulsado la gravedad es
          GRAVITY_HOLD (baja); al soltar o al empezar a caer es GRAVITY. Así la
          duración de la pulsación controla la altura.
        - Velocidad terminal: vy nunca supera MAX_FALL_SPEED.
        El orden (impulso, luego gravedad) es el mismo que usa test_constants.
        """
        if jump and not jump_held and self.on_ground:
            self.vy = -JUMP_SPEED
            self.on_ground = False

        gravity = GRAVITY_HOLD if (jump and self.vy < 0) else GRAVITY
        self.vy = min(self.vy + gravity, MAX_FALL_SPEED)


class Goomba(Body):
    """Enemigo que patrulla: camina en línea recta y se da la vuelta al chocar.

    Reutiliza la física de Body: gravedad, move_x y move_y. No gira en los
    bordes de las plataformas (cae, como en el Mario original).
    """

    __slots__ = ("direction", "speed", "active", "squashed", "_squash_timer")

    def __init__(self, x: float, y: float, speed: float) -> None:
        super().__init__(x, y, GOOMBA_W, GOOMBA_H)
        self.direction = -1      # empieza caminando hacia Mario (izquierda)
        self.speed = speed       # px/frame; P3 puede modificarla tras reset()
        self.active = False      # quieto hasta que la cámara se acerca
        self.squashed = False    # pisado: no colisiona ni se mueve
        self._squash_timer = 0   # frames que le quedan visible tras el pisotón

    def squash(self) -> None:
        self.squashed = True
        self.vx = self.vy = 0.0
        self._squash_timer = SQUASH_FRAMES

    def update(self, level) -> None:
        """Un frame de patrulla: velocidad, gravedad, movimiento y giro."""
        self.vx = self.direction * self.speed
        self.vy = min(self.vy + GRAVITY, MAX_FALL_SPEED)
        if self.move_x(level):           # chocó con una pared o tubería
            self.direction = -self.direction
        self.move_y(level)


def separate_goombas(goombas: list[Goomba]) -> None:
    """Rebote entre goombas: si dos se solapan, cada uno se aleja del otro.

    Se fija la dirección según la posición (el de la izquierda va a la
    izquierda) en vez de invertirla. Invertir sin más falla si siguen
    solapados al frame siguiente: volverían a invertir y quedarían atascados
    temblando uno contra otro.
    """
    # Optimización: ordenados por x, en cuanto un goomba empieza a la derecha
    # del borde de `a` ninguno de los siguientes puede tocarlo, y se corta el
    # bucle. Así se evita comparar todos los pares (n² comparaciones).
    alive = sorted((g for g in goombas if g.active and not g.squashed),
                   key=lambda g: g.x)
    n = len(alive)
    for i in range(n):
        a = alive[i]
        a_right = a.x + a.w
        for j in range(i + 1, n):
            b = alive[j]
            if b.x >= a_right:
                break
            if a.overlaps(b):            # a.x <= b.x: a es el de la izquierda
                a.direction = -1
                b.direction = 1


class MarioGame:
    """Motor del juego. API pública fijada en INTERFACES.md §4.

    reset, step y render completos: física, cámara, goombas y reglas.
    render() delega el dibujo en mario_game/render.py (Persona 2).
    """

    # Un goomba no aparece si está a 3 columnas o menos del inicio de Mario,
    # para no morir en el primer frame (INTERFACES.md §4, reset).
    GOOMBA_SAFE_COLS = 3

    def __init__(self, level=None, goomba_speed: float = GOOMBA_SPEED) -> None:
        if level is None:
            # Import diferido: level.py es de Persona 2 y los tests del motor
            # no deben depender de él. Hasta que exista, el editor marca la
            # línea como módulo no encontrado; el comentario silencia ese aviso.
            try:
                from mario_game.level import build_level_1  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "mario_game/level.py (Persona 2) aún no existe: "
                    "pasa un nivel explícito, MarioGame(level=...)") from exc
            level = build_level_1()
        self.level = level
        self.goomba_speed = goomba_speed
        self._original_tiles = [list(r) for r in level.tiles]
        self.reset()

    def _surface_row_or_error(self, col: int, what: str) -> int:
        """Fila de suelo de `col`; ValueError si es un foso o está fuera del nivel.

        Regla de INTERFACES.md §4 (validación, v0.2): el motor no ignora en
        silencio posiciones inválidas del nivel.
        """
        if not 0 <= col < self.level.width:
            raise ValueError(f"{what}={col} está fuera del nivel (ancho {self.level.width})")
        row = self.level.surface_row(col)
        if row is None:
            raise ValueError(f"{what}={col} está sobre un foso")
        return row

    def reset(self, start_col: int | None = None) -> None:
        # Restaura los bloques ? golpeados en el episodio anterior. Se copia
        # fila a fila dentro de las mismas listas para no cambiar el objeto
        # level.tiles que otros módulos puedan tener referenciado.
        for row_tiles, original in zip(self.level.tiles, self._original_tiles):
            row_tiles[:] = original

        col = self.level.start_col if start_col is None else start_col
        row = self._surface_row_or_error(col, "start_col")

        # Mario centrado en la columna, con los pies sobre el suelo.
        self.mario = Mario(col * TILE + (TILE - MARIO_W) / 2, row * TILE - MARIO_H)
        self.mario.on_ground = True

        self.goombas = [self._spawn_goomba(c) for c in self.level.goomba_cols
                        if c > col + self.GOOMBA_SAFE_COLS]
        # Máximo de la cámara: su borde derecho no pasa del final del nivel. Si
        # el nivel es más estrecho que la pantalla, la cámara se queda en 0.
        self._max_cam_x = float(max(0, self.level.width * TILE - SCREEN_W))
        self.cam_x = 0.0
        self._update_camera()              # centra la cámara en el punto de inicio
        self.dead = False
        self.won = False
        self.death_reason = None
        self._max_x = self.mario.x
        self.progress = 0.0
        self._update_progress()            # con start_col lejano no empieza en 0
        self.jump_held = False
        self.score = 0
        self.coins = 0
        self.time_left = TIME_LIMIT_FRAMES
        self.frame = 0

    def step(self, left: bool, right: bool, jump: bool, run: bool) -> None:
        """Avanza exactamente un frame (1/60 s)."""
        if self.dead or self.won:
            return                       # estado congelado hasta reset()

        m = self.mario
        prev_bottom = m.bottom

        # 1. Física de Mario
        m.apply_horizontal_input(left, right, run)
        m.apply_jump_and_gravity(jump, self.jump_held)
        m.move_x(self.level)
        self._clamp_to_camera_left()
        self._hit_blocks(m.move_y(self.level))

        # 2. Mundo
        self._update_camera()
        self._update_goombas()

        # 3. Reglas: la primera que se cumple decide el final del frame.
        self._check_goomba_contacts(prev_bottom)
        if not self.dead and m.y >= ROWS * TILE:
            self._die("pit")
        if not self.dead:
            self._check_flag()
        self._update_progress()

        self.time_left -= 1
        if self.time_left <= 0 and not (self.dead or self.won):
            self._die("time")

        self.jump_held = jump
        self.frame += 1

    def render(self, surface, hud: bool = True) -> None:
        """Dibuja la vista actual en `surface` (SCREEN_W × SCREEN_H).

        El dibujo es responsabilidad de Persona 2: este método solo delega en
        mario_game.render.draw(surface, game, hud). El import es diferido para
        que step() y los tests funcionen sin pygame ni pantalla. No modifica
        el estado del juego.
        """
        try:
            renderer = importlib.import_module("mario_game.render")
        except ModuleNotFoundError as exc:
            if exc.name != "mario_game.render":
                raise                    # falta otra cosa (p. ej. pygame): error real
            raise ImportError(
                "mario_game/render.py (Persona 2) aún no existe: "
                "MarioGame.render() necesita render.draw(surface, game, hud)") from exc
        renderer.draw(surface, self, hud)

    # --- Goombas (C.6) -------------------------------------------------------

    def _spawn_goomba(self, col: int) -> Goomba:
        row = self._surface_row_or_error(col, "goomba_cols")
        g = Goomba(col * TILE + (TILE - GOOMBA_W) / 2, row * TILE - GOOMBA_H,
                   self.goomba_speed)
        g.on_ground = True
        return g

    def _update_goombas(self) -> None:
        """Activa, mueve y hace rebotar a los goombas; elimina los caídos."""
        activation_x = self.cam_x + SCREEN_W + TILE
        for g in self.goombas:
            if not g.active and g.x < activation_x:
                g.active = True        # una vez activo, sigue activo
            if g.active and not g.squashed:
                g.update(self.level)
            elif g.squashed:
                g._squash_timer -= 1
        separate_goombas(self.goombas)
        # Dejan de existir los que cayeron por un foso y los aplastados cuya
        # animación terminó. Se crea una lista nueva en vez de borrar dentro
        # del for (borrar mientras se recorre una lista salta elementos).
        self.goombas = [g for g in self.goombas
                        if g.y < ROWS * TILE
                        and not (g.squashed and g._squash_timer <= 0)]

    def _check_goomba_contacts(self, prev_bottom: float) -> None:
        """Pisotón o muerte al tocar un goomba activo y no aplastado.

        Es pisotón si Mario cae (vy > 0) y al empezar el frame sus pies
        estaban a la altura del techo del goomba o por encima. Mirar la
        posición ANTERIOR evita que un roce lateral durante una caída cuente
        como pisotón.
        """
        m = self.mario
        stomped = False
        for g in self.goombas:
            if not g.active or g.squashed or not m.overlaps(g):
                continue
            if m.vy > 0 and prev_bottom <= g.y:
                g.squash()
                self.score += SCORE_STOMP
                stomped = True
            else:
                self._die("goomba")
                return
        if stomped:
            m.vy = -STOMP_BOUNCE
            m.on_ground = False

    # --- Reglas (C.7) --------------------------------------------------------

    def _die(self, reason: str) -> None:
        self.dead = True
        self.death_reason = reason

    def _hit_blocks(self, hits: list[tuple[int, int]]) -> None:
        """Golpe con la cabeza: un bloque ? da una moneda y pasa a usado.

        Si la cabeza toca dos celdas, se activa la que está bajo el centro de
        Mario (como en el original); si ninguna lo está, la primera.
        """
        if not hits:
            return
        center_col = to_cell(self.mario.x + self.mario.w / 2)
        row, col = next((h for h in hits if h[1] == center_col), hits[0])
        if 0 <= col < self.level.width and self.level.tiles[row][col] == T_QUESTION:
            self.level.tiles[row][col] = T_USED
            self.coins += 1
            self.score += SCORE_COIN

    def _check_flag(self) -> None:
        """Victoria cuando la caja de Mario entra en la columna de la bandera."""
        if self.mario.right > self.level.flag_col * TILE:
            self.won = True
            self.score += SCORE_FLAG + SCORE_PER_SECOND_LEFT * (self.time_left // FPS)

    def _update_progress(self) -> None:
        """progress = máximo x alcanzado / x de la bandera, en [0, 1]."""
        self._max_x = max(self._max_x, self.mario.x)
        if self.won:
            self.progress = 1.0
        else:
            flag_x = self.level.flag_col * TILE
            self.progress = min(self._max_x / flag_x, 1.0) if flag_x > 0 else 1.0

    # --- Cámara (C.5) --------------------------------------------------------

    def _update_camera(self) -> None:
        """Sigue a Mario cuando pasa del centro de la pantalla; nunca retrocede.

        cam_x es el borde izquierdo de la vista. El objetivo es que el centro
        de Mario quede en el centro de la pantalla; con max() la cámara solo
        avanza, y se limita a [0, _max_cam_x].
        """
        target = self.mario.x + self.mario.w / 2 - SCREEN_W / 2
        self.cam_x = min(max(self.cam_x, target, 0.0), self._max_cam_x)

    def _clamp_to_camera_left(self) -> None:
        """Impide que Mario salga por el borde izquierdo de la pantalla.

        Es seguro empujarlo a cam_x sin revisar colisiones: venía de una
        posición libre a la derecha y se movió menos de su ancho, así que la
        caja en x = cam_x está dentro de la zona que ya recorrió.
        """
        m = self.mario
        if m.x < self.cam_x:
            m.x = self.cam_x
            if m.vx < 0:
                m.vx = 0.0
