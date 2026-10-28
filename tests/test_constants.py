"""P1.3: las constantes cumplen los límites del contrato "en papel".

Aquí no se usa el motor: se simula a mano la misma regla de integración que
usará step() (primero se actualiza la velocidad y luego la posición, cada
frame). Si alguien cambia las constantes y rompe un límite, estos tests fallan
antes de llegar a la física real. Los tests con el motor real llegan en C.4.
"""

from mario_game import constants as C


def jump_profile(hold_jump: bool = True) -> list[float]:
    """Altura de los pies sobre el suelo (px) en cada frame de un salto.

    Termina cuando Mario vuelve a la altura de partida.
    """
    vy = -C.JUMP_SPEED
    height = 0.0
    heights = []
    while True:
        rising_with_button = hold_jump and vy < 0
        vy += C.GRAVITY_HOLD if rising_with_button else C.GRAVITY
        vy = min(vy, C.MAX_FALL_SPEED)
        height -= vy            # y crece hacia abajo; la altura es -y
        if height <= 0:
            return heights
        heights.append(height)


def test_contract_values_unchanged():
    assert C.WALK_MAX_SPEED == 1.5
    assert C.RUN_MAX_SPEED == 2.5
    assert C.MIN_JUMP_TILES == 4
    assert C.MAX_PIT_TILES == 3
    assert C.ROWS == 15
    assert C.SCREEN_H == C.ROWS * C.TILE


def test_speeds_smaller_than_a_tile():
    # Si un cuerpo avanzara un tile o más en un frame podría "saltarse" una pared.
    for speed in (C.RUN_MAX_SPEED, C.JUMP_SPEED, C.MAX_FALL_SPEED):
        assert 0 < speed < C.TILE


def test_jump_reaches_min_height_on_paper():
    peak = max(jump_profile(hold_jump=True))
    assert peak >= C.MIN_JUMP_TILES * C.TILE, peak


def test_short_hop_is_lower_than_full_jump():
    # Gravedad variable: soltar el botón pronto da un salto más bajo.
    assert max(jump_profile(hold_jump=False)) < max(jump_profile(hold_jump=True))


def test_running_jump_clears_max_pit_on_paper():
    # Criterio conservador: Mario debe recorrer en el aire el foso completo
    # más su propio ancho (sale con toda la caja aún sobre el borde).
    airtime = len(jump_profile(hold_jump=True)) + 1
    distance = airtime * C.RUN_MAX_SPEED
    assert distance >= C.MAX_PIT_TILES * C.TILE + C.MARIO_W, distance


def test_tile_codes_are_unique_and_empty_is_not_solid():
    codes = [C.T_EMPTY, C.T_GROUND, C.T_BRICK, C.T_QUESTION,
             C.T_USED, C.T_PIPE, C.T_HARD]
    assert len(set(codes)) == len(codes)
    assert C.T_EMPTY not in C.SOLID_TILES
    assert set(codes) - {C.T_EMPTY} == C.SOLID_TILES
