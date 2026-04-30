import argparse
import time
import pygame
import gymnasium as gym
from stable_baselines3 import SAC, A2C
from entorno_def import EntornoDef

def main():
    parser = argparse.ArgumentParser(description = "Visualización del algoritmo")
    parser.add_argument("--model", type = str, required = True, help = "Ruta del modelo en el equipo.")
    parser.add_argument("--track", type = str, default = "procedural", help = "Tipo del circuito para la visualización.")
    args = parser.parse_args()

    print("Iniciando entorno...")
    env = EntornoDef(render_mode = "human", track_type = args.track)

    model_path = args.model
    print(f"Cargando modelo desde: {model_path}.zip")

    try:
        model = SAC.load(model_path)
    except ValueError:
        model = A2C.load(model_path)
    except FileNotFoundError:
        print("ERROR: Archivo no encontrado.")

    obs, info = env.reset()

    print(f"Arrancando entorno...")

    running = True

    while running:
        if pygame.display.get_init(): # accede al pygame que crea EntornoDef
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            except pygame.error:
                running = False
        
        action, _states = model.predict(obs, deterministic = True) # deterministic evita que el modelo intente inventar y se centre en lo que sabe

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            time.sleep(0.3)

if __name__ == "__main__":
    main()