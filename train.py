import gymnasium as gym
from stable_baselines3 import SAC, A2C
from stable_baselines3.common.callbacks import EvalCallback
from entorno_def import EntornoDef
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

def main():
    env_original = EntornoDef(render_mode = None)

    env_monitorized = Monitor(env_original)

    ALGORITHM = "SAC" # Modificable a "A2C"
    print(f"Cargando: {ALGORITHM}")

    # APILAR 4 IMAGENES
    vec_env = DummyVecEnv([lambda: env_monitorized])

    env_ap = VecFrameStack(vec_env, n_stack = 4)

    eval_env = EntornoDef(render_mode = None)
    eval_env_monitorized = Monitor(eval_env)
    eval_vec = DummyVecEnv([lambda: eval_env_monitorized])
    eval_vec_ap = VecFrameStack(eval_vec, n_stack = 4)
    eval_vec_ap = VecTransposeImage(eval_vec_ap)

    # Comprueba el rendimiento del modelo e irá guardando el mejor
    eval_callback = EvalCallback(
        eval_vec_ap,
        best_model_save_path = "./modelos/",
        log_path = "./logs/",
        eval_freq = 5000,
        deterministic = True,
        render = False
    )

    # Hay que especificarle al algoritmo que usará imagenes. Se define una política (CNNPolicy)
    if ALGORITHM == "SAC":
        model = SAC("CnnPolicy", env_ap, buffer_size = 50000, verbose = 1)
    elif ALGORITHM == "A2C":
        model = A2C("CnnPolicy", env_ap, verbose = 1)
    else:
        raise ValueError("Algoritmo no soportado.")
    
    # Entrenar
    training_steps = 200000
    model.learn(total_timesteps = training_steps, callback = eval_callback)

    # Guardar resultados
    file_path = f"model_{ALGORITHM}_test3_200ksteps"
    model.save(file_path)

    print(f"Entrenamiento finalizado. Modelo guardado correctametne como {file_path}.zip")

    env_ap.close()
    eval_vec_ap.close()

if __name__ == "__main__":
    main()