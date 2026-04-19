import gymnasium as gym
from stable_baselines3 import SAC, A2C
from entorno_def import EntornoDef

def main():
    env = EntornoDef(render_mode = None)

    ALGORITHM = "SAC" # Modificable a "A2C"
    print(f"Cargando: {ALGORITHM}")

    # Hay que especificarle al algoritmo que usará imagenes. Se define una política (CNNPolicy)
    if ALGORITHM == "SAC":
        model = SAC("CnnPolicy", env, verbose = 1)
    elif ALGORITHM == "A2C":
        model = A2C("CnnPolicy", env, verbose = 1)
    else:
        raise ValueError("Algoritmo no soportado.")
    
    # Entrenar
    training_steps = 50000
    model.learn(total_timesteps = training_steps)

    # Guardar resultados
    file_path = f"model_{ALGORITHM}_test"
    model.save(file_path)

    print(f"Entrenamiento finalizado. Modelo guardado correctametne como {file_path}.zip")

    env.close()

if __name__ == "__main__":
    main()