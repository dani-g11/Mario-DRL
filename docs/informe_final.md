# Informe final — Mario DRL (Persona 4)

Entrenamiento, evaluación y resultados del agente que aprende a completar el
nivel por prueba y error (PPO + Stable-Baselines3). Este informe reúne el método
y cómo reproducir cada número; las tablas de resultados del entrenamiento largo
se rellenan al terminar las corridas con `scripts/run_experiments.py`.

> Las celdas marcadas con `—` o «(pendiente de la corrida)» se completan tras
> ejecutar el entrenamiento de 2 M de pasos. Todo lo demás son medidas reales ya
> tomadas en este repositorio.

## 1. Objetivo y criterio de éxito

Del plan de trabajo: el agente entrenado debe llegar a la bandera desde el inicio
en **al menos el 90 % de 20 episodios de evaluación**, también con aleatoriedad
activada (acciones pegajosas y velocidad variable de goombas), en menos de 3 h de
entrenamiento en un portátil.

El problema está formulado como un MDP (contrato en
[`INTERFACES.md`](../INTERFACES.md) §7):

- **Observación**: `grid` (vector de 498 valores con sólidos, enemigos y estado
  de Mario) o `pixels` (84×84 en grises, con apilado de 4 frames).
- **Acciones**: 7 discretas, cada una mantenida 4 frames (frame skip).
- **Recompensa**: Δx/10 por paso, −0.01 por paso, −15 al morir, +50 en la bandera.
- **Fin**: muerte (foso, goomba o tiempo), bandera, o truncado (150 pasos sin
  avanzar o 3000 pasos).

## 2. Pipeline de entrenamiento (P4.1)

`train.py` usa PPO de Stable-Baselines3 con varios entornos en paralelo
(`SubprocVecEnv`), checkpoints periódicos y registro en TensorBoard. `MlpPolicy`
para `grid` y `CnnPolicy` para `pixels`.

Prueba del pipeline en CartPole-v1 antes de gastar horas en Mario:

```bash
python train.py --cartpole --steps 100000
```

CartPole llega a ~500 de recompensa (tope del entorno), lo que confirma que el
bucle de PPO, la vectorización, los callbacks y el registro funcionan.

## 3. Métricas Mario en TensorBoard (P4.2)

`MarioMetricsCallback` (en `mario_training.py`) añade a TensorBoard, además de
las curvas estándar de PPO:

- `mario/progress_mean`: progreso medio hacia la bandera (media móvil de 100
  episodios).
- `mario/flag_rate`: fracción de episodios terminados en la bandera.
- `mario/death_pit`, `mario/death_goomba`, `mario/death_time`: reparto de causas
  de muerte.

Lee las claves del `info` del contrato (`progress`, `flag_get`, `death_reason`),
así que funciona con el entorno real y con un entorno falso (verificado en
`tests/test_training.py`).

```bash
tensorboard --logdir runs
```

## 4. Perfilado: elegir la configuración más rápida (P4.4)

`scripts/profile_training.py` mide los pasos/s de PPO barriendo hilos de PyTorch
y número de entornos. El plan avisa de que demasiados hilos ralentizan
(~800 pasos/s con 24 hilos; ~2500–3000 con 4).

```bash
python scripts/profile_training.py --obs grid --threads 1 2 4 8 --n-envs 4 8 16 --steps 30000
python scripts/profile_training.py --obs grid --threads 1 2 --n-envs 24 32 --steps 30000
```

Medido el 2026-11-08 en una CPU de 32 núcleos lógicos, sin GPU (torch 2.15.0+cpu),
`obs=grid`, `SubprocVecEnv`, 30 000 pasos por configuración:

| Entornos | 1 hilo | 2 hilos | 4 hilos | 8 hilos |
|---|---|---|---|---|
| 4 | 999 | 1 211 | 1 198 | 948 |
| 8 | 1 780 | 1 548 | 1 433 | 1 413 |
| 16 | **2 155** | 1 742 | 1 415 | 1 613 |
| 24 | 1 507 | 1 679 | — | — |
| 32 | 1 824 | 1 614 | — | — |

(pasos/s de PPO, recogida de experiencia + actualización)

Conclusión: usar `--n-envs 16 --torch-threads 1` en el entrenamiento principal
(~2 150 pasos/s, unos 15–20 min para 2 M de pasos). Como avisaba el plan, más
hilos de PyTorch no ayudan con una `MlpPolicy` pequeña; y pasar de 16 entornos
no compensa por el coste de los procesos. Las diferencias de ~10 % entre
configuraciones vecinas están dentro del ruido de una corrida corta.

## 5. Líneas base (P4.5)

`scripts/baselines.py` mide dos agentes que no aprenden, en el nivel 1 (medidas
reales, 30 episodios cada una):

| Línea base | Progreso medio | Tasa de bandera |
|---|---|---|
| Aleatorio | ~20 % | 0/30 (0 %) |
| Heurístico (bot que corre y salta) | ~100 % | 30/30 (100 %) |

El agente aleatorio apenas avanza: cualquier agente aprendido debería superarlo
con holgura. El bot heurístico pasa el nivel siempre y marca el techo de
referencia (el nivel es superable; lo valida también `mario_game/validate.py`).

## 6. Entrenamiento principal (P4.6)

```bash
python train.py --obs grid --steps 2000000 --name lvl1_grid_s0 --seed 0 --n-envs 16 --torch-threads 1
python evaluate.py --model models/lvl1_grid_s0/final.zip --episodes 20
```

Criterio de P4.6: el agente llega a la bandera desde el inicio. **Cumplido.**

Corrida del 2026-11-08 (CPU de 32 núcleos lógicos, sin GPU, configuración del §4):

- Pasos: 2 000 000 (2 007 040 por redondeo al tamaño del rollout).
- Tiempo de entrenamiento: 18 min 6 s (~1 850 pasos/s), muy por debajo del
  límite de 3 h del plan.
- Tasa de bandera al final (evaluación de 20 episodios): **20/20 (100 %)**,
  progreso medio 100 %, recompensa media 340,2, ~300 pasos por episodio.

Curva de aprendizaje (`mario/flag_rate` y `mario/progress_mean`, media móvil de
100 episodios de entrenamiento):

| Pasos | Tasa de bandera | Progreso medio |
|---|---|---|
| 200 000 | 3 % | 42 % |
| 400 000 | 77 % | 88 % |
| 600 000 | 92 % | 96 % |
| 800 000 | 97 % | 98 % |
| 1 000 000 – 1 800 000 | 98 % | 98–99 % |
| 2 000 000 | 100 % | 100 % |

El agente supera el 90 % hacia los 600 000 pasos (~6 min) y se estabiliza a
partir del millón. Este resultado es sin aleatoriedad: la robustez y la
generalización se miden en el §8.

## 7. Experimentos (P4.7)

`scripts/run_experiments.py` entrena la rejilla de experimentos y evalúa cada
modelo. Variables: `grid` frente a `pixels`, con y sin inicio aleatorio, 3
semillas.

```bash
python scripts/run_experiments.py --steps 2000000 --seeds 0 1 2 --random-start
```

| Observación | Inicio aleatorio | Semilla | Tasa de bandera (20 ep.) | Progreso medio |
|---|---|---|---|---|
| grid | no | 0/1/2 | (pendiente) | (pendiente) |
| grid | sí | 0/1/2 | (pendiente) | (pendiente) |
| pixels (4 frames) | no | 0/1/2 | (pendiente) | (pendiente) |
| pixels (4 frames) | sí | 0/1/2 | (pendiente) | (pendiente) |

Las curvas comparativas salen de TensorBoard (`mario/flag_rate` y
`mario/progress_mean` por experimento).

## 8. Robustez y generalización (P4.8)

Cada modelo se evalúa 20 episodios por condición. Las tres primeras prueban la
robustez (aleatoriedad del contrato, P3.9); la última, la generalización a un
nivel no visto (P3.10).

```bash
python evaluate.py --model models/lvl1_grid_s0/final.zip --episodes 20
python evaluate.py --model models/lvl1_grid_s0/final.zip --episodes 20 --sticky-prob 0.1
python evaluate.py --model models/lvl1_grid_s0/final.zip --episodes 20 \
    --goomba-speed-min 0.3 --goomba-speed-max 1.5
python evaluate.py --model models/lvl1_grid_s0/final.zip --episodes 20 --level 2
```

Resultados del modelo `lvl1_grid_s0` (§6, entrenado sin aleatoriedad), medidos
el 2026-11-08. La última fila combina las dos fuentes de aleatoriedad, que es lo
que pide el criterio de éxito del §1:

| Condición | Tasa de bandera (20 ep.) | Progreso medio | Muertes |
|---|---|---|---|
| Sin aleatoriedad (nivel 1) | 20/20 (100 %) | 100 % | — |
| Acciones pegajosas (0.1) | 20/20 (100 %) | 100 % | — |
| Velocidad de goomba 0.3–1.5 | 19/20 (95 %) | 96,9 % | 1 goomba |
| Pegajosas 0.1 + goomba 0.3–1.5 | **17/20 (85 %)** | 89,3 % | 3 goomba |
| Generalización (nivel 2) | 16/20 (80 %) | 88,8 % | 4 goomba |

Lectura:

- Las acciones pegajosas no le afectan: el agente no depende de una secuencia
  exacta de pulsaciones.
- Los goombas con velocidad variable son su punto débil. Todas las muertes son
  por goomba, casi todas en el mismo punto (~39 % del nivel 1, ~44 % del
  nivel 2): el agente aprendió dónde suelen estar los goombas a velocidad fija,
  no a reaccionar a dónde están.
- Con las dos fuentes de aleatoriedad a la vez se queda en el 85 %, **por debajo
  del 90 % del criterio de éxito**. En el nivel 2, que nunca ha visto, llega al
  80 %: generaliza bastante (mismas piezas, otro orden), pero no del todo.
- Siguiente paso: entrenar con aleatoriedad activada (`--goomba-speed-min 0.3
  --goomba-speed-max 1.5 --sticky-prob 0.1`) y con inicio aleatorio
  (`--random-start`), que es la rejilla del §7, y volver a medir esta tabla.

## 9. Vídeo de demostración (P4.9)

```bash
python scripts/record_video.py --model models/lvl1_grid_s0/final.zip \
    --episodes 3 --out docs/captures/agente_nivel1.mp4
```

Graba el agente completando el nivel usando el render con HUD del entorno. Si no
hay ffmpeg, cae a GIF automáticamente. Acepta las mismas opciones de
aleatoriedad que `evaluate.py` (`--sticky-prob`, `--goomba-speed-min/max`) y
además `--random-start`:

```bash
python scripts/record_video.py --model models/lvl1_grid_s0/final.zip --deterministic \
    --fps 15 --goomba-speed-min 0.3 --goomba-speed-max 1.5 --sticky-prob 0.1 \
    --episodes 3 --out docs/captures/agente_aleatorio.mp4
```

Vídeos grabados el 2026-11-08 con `lvl1_grid_s0`, a velocidad real (`--fps 15`:
un frame por paso del agente, que son 4 frames de juego a 60 FPS) y escalados ×3
sin suavizado (768×720) para proyectarlos:

| Vídeo | Contenido | Resultado |
|---|---|---|
| `agente_nivel1.mp4` | Nivel 1, sin aleatoriedad | Bandera |
| `agente_nivel2.mp4` | Nivel 2 (no visto), sin aleatoriedad | Bandera |
| `agente_aleatorio.mp4` | Nivel 1, 3 episodios con pegajosas 0.1 y goombas a 1,06, 0,91 y 0,61 | 3 banderas |

## 10. Reproducir todo

```bash
# 1. Entorno (Python 3.12)
py -3.12 -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Pipeline sano (CartPole)
python train.py --cartpole --steps 100000

# 3. Líneas base
python scripts/baselines.py --episodes 30

# 4. Perfilar y elegir configuración
python scripts/profile_training.py --obs grid

# 5. Experimentos + robustez + generalización (largo)
python scripts/run_experiments.py --steps 2000000 --seeds 0 1 2 --random-start

# 6. Vídeo del mejor modelo
python scripts/record_video.py --model models/<mejor>/final.zip --episodes 3
```

Los resultados numéricos finales quedan en
[`docs/resultados.md`](resultados.md), que genera `scripts/run_experiments.py`.

## 11. Estado y pendientes

- Pipeline (P4.1), métricas (P4.2), evaluación (P4.3), perfilado (P4.4), líneas
  base (P4.5), experimentos y robustez (P4.7, P4.8) y vídeo (P4.9): **código
  listo y probado** (suite en verde, incluidos los tests de P4).
- Entrenamiento largo de 2 M de pasos (P4.6) y tablas de resultados: **a correr**
  en el portátil del equipo; este informe se completa con esos números.
