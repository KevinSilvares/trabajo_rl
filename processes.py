def train_SAC(track = "Óvalo", steps = 200000, learning_rate = 0.0003, buffer_size = 50000):
    print(f"""Se comienza a entrenar SAC con los parámetros:\
        Circuito: {track}
        Pasos: {steps}
        Learning rate: {learning_rate}
        Buffer size: {buffer_size}
        """)

def train_A2C(track = "Óvalo", steps = 500000, learning_rate = 0.0005, ent_coef = 0.1, gamma = 0.80):
    print(f"""Se comienza a entrenar A2C con los parámetros:\
        Circuito: {track}
        Pasos: {steps}
        Learning rate: {learning_rate}
        Entropía: {ent_coef}
        Gamma: {gamma}
        """)
    
def handle_visualization(model):
    pass

def load_algorithm_to_ui(model_path):
    try:
        with open(model_path) as f:
            f.write(model_path.getbuffer())
        return True
    except FileNotFoundError:
        return False