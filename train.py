import os
import gymnasium as gym
from stable_baselines3 import SAC, A2C
from stable_baselines3.common.callbacks import EvalCallback
from entorno_def import EntornoDef
from stable_baselines3.common.monitor import Monitor

def main():
    PHASE = "vectores/fase1.0.1"
    TRAINING_STEPS = 500000

    os.makedirs("./logs/tensorboard", exist_ok = True)
    os.makedirs(f"./modelos/{PHASE}", exist_ok = True)

    env = EntornoDef(render_mode = None)
    env_monitorized = Monitor(env)

    eval_env = EntornoDef(render_mode = None)
    eval_env_monitorized = Monitor(eval_env)

    ALGORITHM = "A2C" # Modificable a "A2C"
    print(f"Cargando: {ALGORITHM}")

    # Comprueba el rendimiento del modelo e irá guardando el mejor
    eval_callback = EvalCallback(
        eval_env_monitorized,
        best_model_save_path = f"./modelos/{ALGORITHM}_{PHASE}",
        log_path = f"./logs/{ALGORITHM}_{PHASE}",
        eval_freq = 5000,
        deterministic = True,
        render = False
    )

    # Hay que especificarle al algoritmo que usará parámetros de estado (vectores). Se define una política (MlpPolicy)
    if ALGORITHM == "SAC":
        # model = SAC("MlpPolicy", env_monitorized, buffer_size = 50000, verbose = 1, tensorboard_log = "./logs/tensorboard/")
        model = SAC.load("./modelos/vectores/fase1.0.1/best_model", env = env_monitorized, tensorboard_logs = "./logs/tensorboard")
    elif ALGORITHM == "A2C":
        model = A2C("MlpPolicy", env_monitorized, verbose = 1, learning_rate = 0.0005, ent_coef = 0.01, gamma = 0.99, tensorboard_log = "./logs/tensorboard")
    else:
        raise ValueError("Algoritmo no soportado.")
    
    # Entrenar
    model.learn(total_timesteps = TRAINING_STEPS, callback = eval_callback, tb_log_name = f"{ALGORITHM}{PHASE}", reset_num_timesteps = False)

    # Guardar resultados
    file_path = f"model_{ALGORITHM}_{PHASE}"
    model.save(file_path)

    print(f"Entrenamiento finalizado. Modelo guardado correctametne como {file_path}.zip")

    env.close()

if __name__ == "__main__":
    main()