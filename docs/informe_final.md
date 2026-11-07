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
python scripts/profile_training.py --obs grid --threads 1 2 4 8 --n-envs 8 16
```

| Entornos | Hilos PyTorch | Pasos/s | Notas |
|---|---|---|---|
| (pendiente de la corrida en el portátil del equipo) | | | |

Conclusión (a rellenar): usar `--n-envs N --torch-threads T` en el entrenamiento
principal.

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
python train.py --obs grid --steps 2000000 --name lvl1_grid_s0 --seed 0
```

Criterio de P4.6: el agente llega a la bandera desde el inicio.

- Pasos: 2 000 000.
- Tiempo de entrenamiento: (pendiente de la corrida).
- Tasa de bandera al final (evaluación de 20 episodios): (pendiente).

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

| Condición | Tasa de bandera (20 ep.) | Progreso medio |
|---|---|---|
| Sin aleatoriedad (nivel 1) | (pendiente) | (pendiente) |
| Acciones pegajosas (0.1) | (pendiente) | (pendiente) |
| Velocidad de goomba 0.3–1.5 | (pendiente) | (pendiente) |
| Generalización (nivel 2) | (pendiente) | (pendiente) |

Lectura esperada: si el agente memoriza el nivel determinista, caerá con
aleatoriedad y en el nivel 2; entrenar con inicio aleatorio y evaluar en ambos
niveles mide si de verdad aprendió a jugar.

## 9. Vídeo de demostración (P4.9)

```bash
python scripts/record_video.py --model models/lvl1_grid_s0/final.zip \
    --episodes 3 --out docs/captures/agente_nivel1.mp4
```

Graba el agente completando el nivel usando el render con HUD del entorno. Si no
hay ffmpeg, cae a GIF automáticamente.

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
