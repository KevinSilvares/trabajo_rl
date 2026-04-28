import os
import streamlit as st
import processes as p

# SIDEBAR
st.sidebar.title("Coche con Reinforcement Learning")

section = st.sidebar.radio("Ir a:", ["Entrenamiento", "Visualización"])

# BODY
if section == "Entrenamiento":
    st.title("Entrenamiento")

    col1, col2 = st.columns([3, 1])

    with col1:
        algorithm = st.selectbox("Algoritmo", ["SAC", "A2C"], help = "Algoritmo para entrenar.")
        model_name = st.text_input(label = "Nombre para el algoritmo", max_chars = 30, help = "Al finalizar el entrenamiento se guardará un archivo con el nombre del algoritmo proporcionado.")
        st.markdown(f"[Para saber más sobre {algorithm}](docs/{algorithm})")
    with col2:
        models_path = "./modelos/"
        os.makedirs(models_path, exist_ok = True)

        # st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html = True) # Pequeño espacio y "regla" para alinear el botón con el selectbox
        loaded_model = st.file_uploader("Cargar Modelo", help = "Sube un modelo ya entrenado para seguir entrenando o visualizarlo.", type = [".zip"])

        if loaded_model is not None:
            file_path = os.path.join(models_path, loaded_model.name)

            if p.load_algorithm_to_ui(file_path):
                st.success(f"Modelo {loaded_model.name} cargado.")

    st.markdown("---")

    st.markdown("### Hiperparámetros")

    track = st.radio("Circuito:", ["Óvalo", "Procedimental"], help = "Circuito en el que se entrenará el algoritmo.\n- El óvalo es más sencillo y siempre estático. Es bueno para entrenamientos cortos iniciales.\n- Los circuitos procedimentales son aleatorios y nunca son iguales. Mucho mejores para la generalización del modelo, pero puede tardar en dar resultados en entrenamientos iniciales.")

    if algorithm == "SAC":
        training_steps = st.number_input(
            "Training Steps (Pasos de Entrenamiento)",
            value = 200000,
            step = 1000,
            help = "Cuantas iteraciones va a realizar el modelo para entrenarse. Para SAC 200.000 es un buen punto de partida."
        )

        learning_rate = st.number_input(
            "Learning Rate (Tasa de Aprendizaje)", 
            value = 0.0003,
            step = 0.00001,
            help = "Velocidad a la que el modelo actualiza sus pesos. 0.0003 es el valor por defecto para SAC.",
            format = "%.5f"
        )
        
        buffer_size = st.selectbox(
            "Buffer Size (Tamaño de Buffer)", 
            [50000, 100000, 1000000],
            help = "Tamaño de la memoria a corto plazo. + grande = + recuerdo = + RAM necesaria."
        )
    elif algorithm == "A2C":
        training_steps = st.number_input(
            "Training Steps (Pasos de Entrenamiento)",
            value = 500000,
            step = 1000,
            help = "Cuantas iteraciones va a realizar el modelo para entrenarse. Para A2C 500.000 es un buen punto de partida."
        )

        learning_rate = st.number_input(
            "Learning Rate (Tasa de Aprendizaje)", 
            value = 0.0005,
            step = 0.00001,
            help = "Velocidad a la que el modelo actualiza sus pesos. 0.0005 es el valor por defecto para A2C.",
            format = "%.5f"
        )
            
        ent_coef = st.slider(
                "Entropía (Exploración)", 
                min_value = 0.0,
                max_value = 0.1,
                value = 0.01,
                help = "Relación entre el uso de conocimiento obtenido y exploración. Cuanto menor sea, más se centrará en su propio conocimiento, pero menos aprenderá estrategias nuevas y viceversa."
        )
            
        gamma = st.slider("Gamma (Visión a Futuro)",
                min_value = 0.0,
                max_value = 1.0,
                value = 0.90,
                help = "Controla cuanto piensa el algoritmo en recompensas futuras frente a las recompensas inmediatas. Un valor bajo hará que el agente se centre en la recompensa inmediata. Un gamma alta hará que se centre en conseguir la mayor recompensa en el futuro."
        )

    if st.button("Comenzar entrenamiento"):
        if algorithm == "SAC":
            p.train_SAC(track, training_steps, learning_rate, buffer_size)
        elif algorithm == "A2C":
            p.train_A2C(track, training_steps, learning_rate, ent_coef, gamma)

    performance_graph = st.empty()
elif section == "Visualización":
    st.title("Visualización")