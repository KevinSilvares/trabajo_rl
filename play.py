import gymnasium as gym
from stable_baselines3 import SAC, A2C
from entorno_def import EntornoDef

def main():
    PHASE = "vectores/fase1.0.2_continuacion"

    print("Iniciando entorno...")
    env = EntornoDef(render_mode = "human")

    model_path = f"modelos/{PHASE}/best_model"
    print(f"Cargando modelo desde: {model_path}.zip")

    try:
        model = SAC.load(model_path)
    except FileNotFoundError:
        print("ERROR: Archivo no encontrado.")

    obs, info = env.reset()

    print(f"Arrancando entorno...")

    while True:
        action, _states = model.predict(obs, deterministic = True) # deterministic evita que el modelo intente inventar y se centr en lo que sabe

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    main()