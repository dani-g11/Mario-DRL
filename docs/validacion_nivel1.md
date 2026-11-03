# Validación de nivel 1

- Nivel de 200 columnas (3200 px), bandera en la columna 186, 4 goombas y 9 checkpoints.
- Un bot sencillo (corre a la derecha y salta ante paredes, fosos y goombas) intenta llegar a la bandera desde cada checkpoint, con el motor real.
- Velocidades de goomba probadas (px/frame): 0.3, 0.5, 1, 1.5.

| Tramo | Checkpoint | Goomba 0.3 | Goomba 0.5 | Goomba 1 | Goomba 1.5 |
|---|---|---|---|---|---|
| 0 | col. 2 | OK (21 s) | OK (21 s) | OK (21 s) | OK (21 s) |
| 0 | col. 36 | OK (17 s) | OK (17 s) | OK (17 s) | OK (17 s) |
| 1 | col. 55 | OK (14 s) | OK (14 s) | OK (14 s) | OK (14 s) |
| 1 | col. 65 | OK (13 s) | OK (13 s) | OK (13 s) | OK (13 s) |
| 2 | col. 94 | OK (11 s) | OK (11 s) | OK (11 s) | OK (11 s) |
| 2 | col. 117 | OK (8 s) | OK (8 s) | OK (8 s) | OK (8 s) |
| 3 | col. 134 | OK (6 s) | OK (6 s) | OK (6 s) | OK (6 s) |
| 3 | col. 144 | OK (5 s) | OK (5 s) | OK (5 s) | OK (5 s) |
| 3 | col. 156 | OK (3 s) | OK (3 s) | OK (3 s) | OK (3 s) |

**Resultado: 36 de 36 partidas ganadas. Todos los tramos son superables desde cada checkpoint.**
