import gymnasium as gym
from stable_baselines3 import SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from entorno_def import EntornoDef

def main():
    print("Iniciando entorno...")
    env_orig = EntornoDef(render_mode = "human")

    # Hay que recrear el mismo empaquetado que en entrenamiento
    vec_env = DummyVecEnv([lambda: env_orig])
    env_ap = VecFrameStack(vec_env, n_stack = 4)

    model_path = "./modelos/fase2/best_model"
    print(f"Cargando modelo desde: {model_path}.zip")

    try:
        model = SAC.load(model_path)
    except FileNotFoundError:
        print("ERROR: Archivo no encontrado.")

    obs = env_ap.reset()

    print(f"Arrancando entorno...")

    while True:
        action, _states = model.predict(obs, deterministic = True) # deterministic evita que el modelo intente inventar y se centr en lo que sabe

        obs, rewards, dones, infos = env_ap.step(action)

if __name__ == "__main__":
    main()