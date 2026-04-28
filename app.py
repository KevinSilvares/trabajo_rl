import streamlit as st

# SIDEBAR
st.sidebar.title("Coche con Reinforcement Learning")
algorithm = st.sidebar.selectbox("Algoritmo", ["SAC", "A2C"])

st.sidebar.markdown(f"[Para saber más sobre {algorithm}](docs/SAC)")
st.sidebar.markdown("Hiperparámetros")

if algorithm == "SAC":
    learning_rate = st.sidebar.number_input(
        "Learning Rate (Tasa de Aprendizaje)", 
        value = 0.0003,
        help = "Velocidad a la que el modelo actualiza sus pesos. 0.0003 es el valor por defecto para SAC.",
        format = "%.5f"
        )
    
    buffer = st.sidebar.selectbox(
        "Buffer Size (Tamaño de Buffer)", 
        [50000, 100000, 1000000],
        help = "Tamaño de la memoria a corto plazo. + grande = + recuerdo = + RAM necesaria."
        )
    
elif algorithm == "A2C":
    learning_rate = st.sidebar.number_input(
        "Learning Rate (Tasa de Aprendizaje)", 
        value = 0.0005,
        help = "Velocidad a la que el modelo actualiza sus pesos. 0.0005 es el valor por defecto para A2C.",
        format = "%.5f"
        )
    
    ent_coef = st.sidebar.slider("Entropía (Exploración)", 
        min_value = 0.0,
        max_value = 0.1,
        value = 0.01,
        help = "Relación entre el uso de conocimiento obtenido y exploración. Cuanto menor sea, más se centrará en su propio conocimiento, pero menos aprenderá estrategias nuevas y viceversa."
        )
    
    gamma = st.sidebar.slider("Gamma (Visión a Futuro)",
        min_value = 0.0,
        max_value = 1.0,
        value = 0.90,
        help = "Controla cuanto piensa el algoritmo en recompensas futuras frente a las recompensas inmediatas. Un valor bajo hará que el agente se centre en la recompensa inmediata. Un gamma alta hará que se centre en conseguir la mayor recompensa en el futuro."
        )
    
st.sidebar.button("Iniciar entrenamiento")

# BODY
st.title("Entrenamiento")

graph = st.empty()