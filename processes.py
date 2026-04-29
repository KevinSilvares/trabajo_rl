import os
import subprocess
import sys

def train_SAC(name, track = "Óvalo", steps = 200000, learning_rate = 0.0003, buffer_size = 50000):
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

    print(f"Lanzando proceso en segundo plano: {command}")
    subprocess.Popen(command)
    return True

def train_A2C(name, track = "Óvalo", steps = 500000, learning_rate = 0.0005, ent_coef = 0.1, gamma = 0.80):
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

    print(f"Lanzando proceso en segundo plano: {command}")
    subprocess.Popen(command)
    return True
    
def handle_visualization(model):
    pass

def load_algorithm_to_ui(model):
    temp_models_path = "./temp_modelos/"
    os.makedirs(temp_models_path, exist_ok = True)

    file_path = os.path.join(temp_models_path, model.name)

    try:
        with open(file_path, "wb") as f:
            f.write(model.getbuffer())
        return True
    except FileNotFoundError:
        return False