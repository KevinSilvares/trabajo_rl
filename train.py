import os
import argparse
import gymnasium as gym
from stable_baselines3 import SAC, A2C
from stable_baselines3.common.callbacks import EvalCallback
from entorno_def import EntornoDef
from stable_baselines3.common.monitor import Monitor

def main():
    print("Parseando argumentos.")
    # Argumentos comunes
    parser = argparse.ArgumentParser(description = "Entrenamiento de algoritmo")
    parser.add_argument("--name", type = str, required = True, help = "Nombre de archivo de salida.")
    parser.add_argument("--algorithm", type = str, required = True, help = "Algoritmo a utilizar (SAC o A2C).")
    parser.add_argument("--track", type = str, default = "oval", help = "Circuito para entrenar (oval o procedural).")
    parser.add_argument("--steps", type = int, default = 100000, help = "Pasos del entrenamiento.")
    parser.add_argument("--lr", type = float, default = 0.0003, help = "Tasa de aprendizaje.")

    # Argumentos SAC
    parser.add_argument("--buffer", types = int, default = 50000, help = "Tamaño del buffer de memoria a corto plazo de SAC.")

    # Argumentos A2C
    parser.add_argument("--ent_coef", type = float, default = 0.01)
    parser.add_argument("--gamma", type = float, default = 0.90)

    args = parser.parse_args()

    print(f"Argumentos:\n {args}")
    
    os.makedirs("./logs/tensorboard", exist_ok = True)
    os.makedirs(f"./modelos/{args.algorithm}", exist_ok = True)

    print("Creando entorno.")
    env = EntornoDef(render_mode = None, track_type = args.track)
    env_monitorized = Monitor(env)

    eval_env = EntornoDef(render_mode = None, track_type = args.track)
    eval_env_monitorized = Monitor(eval_env)

    print(f"Cargando algoritmo: {args.algorithm}")
    if args.algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env_monitorized,
            learning_rate = args.lr,
            buffer_size = args.buffer,
            tensorboard_log = "./logs/tensorboard"
        )
    elif args.algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            env_monitorized,
            learning_rate = args.lr,
            ent_coef = args.ent_coef,
            gamma = args.gamma,
            tensorboard_log = "./logs/tensorboard"
        )
    else:
        raise ValueError("Algoritmo no soportado. Sólo SAC o A2C.")

    # Comprueba el rendimiento del modelo e irá guardando el mejor
    eval_callback = EvalCallback(
        eval_env_monitorized,
        best_model_save_path = f"./modelos/{args.algorithm}_{args.name}",
        log_path = f"./logs/{args.algorithm}_{args.name}",
        eval_freq = 5000,
        deterministic = True,
        render = False
    )
    
    # Entrenar
    model.learn(total_timesteps = args.steps, callback = eval_callback, tb_log_name = f"{args.algorithm}_{args.name}", reset_num_timesteps = False)

    # Guardar resultados
    file_path = f"./modelos/final_{args.algorithm}_{args.name}"
    model.save(file_path)

    print(f"Entrenamiento finalizado. Modelo guardado correctametne como {file_path}.zip")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()