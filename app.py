import os
import streamlit as st
import processes as p
import pandas as pd
import time

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
        # st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html = True) # Pequeño espacio y "regla" para alinear el botón con el selectbox
        loaded_model = st.file_uploader("Cargar Modelo", help = "Sube un modelo ya entrenado para seguir entrenando o visualizarlo.", type = [".zip"])

        if loaded_model is not None:
            if p.load_algorithm_to_ui(loaded_model):
                st.success(f"Modelo {loaded_model.name} cargado.")

    st.markdown("---")

    st.markdown("### Hiperparámetros")

    track = st.radio("Circuito:", ["Óvalo", "Procedimental"], help = "Circuito en el que se entrenará el algoritmo.\n- El óvalo es más sencillo y siempre estático. Es bueno para entrenamientos cortos iniciales.\n- Los circuitos procedimentales son aleatorios y nunca son iguales. Mucho mejores para la generalización del modelo, pero puede tardar en dar resultados en entrenamientos iniciales.")

    if algorithm == "SAC":
        training_steps = st.number_input(
            "Training Steps (Pasos de Entrenamiento)",
            min_value = 5000,
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
            min_value = 5000,
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
        if not model_name:
            st.error("El nombre del modelo no puede estar vacío.")
        else:
            if algorithm == "SAC":
                p.train_SAC(model_name, track,  training_steps, learning_rate, buffer_size)
            elif algorithm == "A2C":
                p.train_A2C(model_name, track, training_steps, learning_rate, ent_coef, gamma)
        
            st.success("Entrenamiento lanzado en segundo plano.")

            # Monitorización
            st.markdown("---")
            st.markdown("Rendimiento en Tiempo Real")
            col_rew, col_ent, col_steps = st.columns(3)

            with col_rew:
                st.markdown("#### Reward (recompensa)")
                plot_reward = st.empty()

            with col_ent:
                st.markdown("#### Entropía")
                plot_entropy = st.empty()
            
            with col_steps:
                st.markdown("#### Pasos por episodio")
                plot_steps = st.empty()

            csv_path = f"./streamlit/logs/{algorithm}_{model_name}/progress.csv"

            while True:
                try:
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)

                        if "rollout/ep_rew_mean" in df.columns:
                            plot_reward.line_chart(df["rollout/ep_rew_mean"].dropna())
                        if "rollout/ep_len_mean" in df.columns:
                            plot_steps.line_chart(df["rollout/ep_len_mean"].dropna())
                        
                        # En SAC es train/ent_coef. En A2C es train/entropy_loss
                        col_ent = "train/ent_coef" if algorithm == "SAC" else "train/entropy_loss"
                        if col_ent in df.columns:
                            plot_entropy.line_chart(df[col_ent].dropna()) 
                    
                    time.sleep(5)
                except Exception as e:
                    # Evita que se escriba y se lea al mismo tiempo. Hacía que petase
                    time.sleep(1)

elif section == "Visualización":
    st.title("Visualización")