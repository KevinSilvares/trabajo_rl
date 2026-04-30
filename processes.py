import os
import subprocess
import sys

TEMP_MODELS_PATH = "./temp_modelos/"

def train_SAC(name, track = "Óvalo", steps = 200000, learning_rate = 0.0003, buffer_size = 50000, loaded_model_path = None):
    track_type = "oval" if track == "Óvalo" else "procedural" # más fácil de leer y escribir en código

    command = [
        sys.executable, "train.py",
        "--name", name,
        "--algorithm", "SAC",
        "--track", track_type,
        "--steps", str(steps),
        "--lr", str(learning_rate),
        "--buffer", str(buffer_size)
    ]

    if loaded_model_path:
        command.extend(["--load", loaded_model_path])

    print(f"Lanzando proceso en segundo plano: {command}")
    subprocess.Popen(command)
    return True

def train_A2C(name, track = "Óvalo", steps = 500000, learning_rate = 0.0005, ent_coef = 0.1, gamma = 0.80, loaded_model_path = None):
    track_type = "oval" if track == "Óvalo" else "procedural" # más fácil de leer y escribir en código

    command = [
        sys.executable, "train.py",
        "--name", name,
        "--algorithm", "A2C",
        "--track", track_type,
        "--steps", str(steps),
        "--lr", str(learning_rate),
        "--ent_coef", str(ent_coef),
        "--gamma", str(gamma)
    ]

    if loaded_model_path:
        command.extend(["--load", loaded_model_path])

    print(f"Lanzando proceso en segundo plano: {command}")
    subprocess.Popen(command)
    return True
    
def handle_visualization(model, visualization_track):
    vis_track = "oval" if visualization_track == "Óvalo" else "procedural"
    model_path = os.path.join(TEMP_MODELS_PATH, model)

    command = [
        sys.executable, "play.py",
        "--model", model_path,
        "--track", vis_track
    ]

    print("Lanzando ventana de visualización")
    subprocess.Popen(command)
    return True

def load_algorithm_to_ui(model):
    os.makedirs(TEMP_MODELS_PATH, exist_ok = True)

    file_path = os.path.join(TEMP_MODELS_PATH, model.name)

    try:
        with open(file_path, "wb") as f:
            f.write(model.getbuffer())
        return True
    except FileNotFoundError:
        return False

def get_loaded_algorithm_path(model):
    return os.path.join(TEMP_MODELS_PATH, model.name)