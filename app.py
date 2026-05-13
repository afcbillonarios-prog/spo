import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuración de la interfaz
st.set_page_config(page_title="Predicciones Deportivas Pro", layout="wide")
st.title("⚽ Análisis Predictivo: ML & Redes Neuronales")

# 1. Carga y Limpieza de Datos
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("historical_matches.csv") 
    except FileNotFoundError:
        # Datos simulados para demostración si no hay archivo
        np.random.seed(42)
        n = 500
        data = pd.DataFrame({
            'home_rank': np.random.randint(1, 21, n),
            'away_rank': np.random.randint(1, 21, n),
            'avg_goals_last_5': np.random.uniform(0, 3, n),
            'shot_efficiency': np.random.uniform(0.05, 0.25, n),
            'result_encoded': np.random.choice([0, 1, 2], n) # 0: Home, 1: Draw, 2: Away
        })
    return data

df = load_data()

# 2. Preprocesamiento para la Red Neuronal
def prepare_model(df):
    X = df[['home_rank', 'away_rank', 'avg_goals_last_5', 'shot_efficiency']]
    y = df['result_encoded'] # 0: Home, 1: Draw, 2: Away
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# 3. Arquitectura de la Red Neuronal
def build_neural_network(input_shape):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_shape,)),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax') # Salida para Win/Draw/Loss
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Lógica de la App ---
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

if st.sidebar.button("Entrenar Modelo"):
    X, y, scaler = prepare_model(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = build_neural_network(X.shape[1])
    with st.spinner('Entrenando neuronas...'):
        model.fit(X_train, y_train, epochs=50, verbose=0)
    
    st.success("Modelo entrenado con éxito.")
    
    # Guardar en session state para que persista
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.model_trained = True
    
    # Evaluación
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.sidebar.metric("Precisión (Accuracy)", f"{accuracy*100:.2f}%")

# 4. Predicción en tiempo real
st.subheader("Predicción de Próximo Encuentro")

if not st.session_state.model_trained:
    st.warning("⚠️ Primero debes entrenar el modelo desde el menú lateral.")
else:
    col1, col2 = st.columns(2)
    with col1:
        h_rank = st.number_input("Ranking Local", 1, 20, 5)
        h_goals = st.slider("Goles Promedio (Últimos 5)", 0.0, 5.0, 1.5)
    with col2:
        a_rank = st.number_input("Ranking Visitante", 1, 20, 10)
        a_eff = st.slider("Eficiencia de Tiro", 0.0, 1.0, 0.15)

    if st.button("Calcular Probabilidades", type="primary"):
        # Preparar los datos de entrada
        input_data = pd.DataFrame({
            'home_rank': [h_rank],
            'away_rank': [a_rank],
            'avg_goals_last_5': [h_goals],
            'shot_efficiency': [a_eff]
        })
        
        # Escalar usando el scaler entrenado
        input_scaled = st.session_state.scaler.transform(input_data)
        
        # Predecir
        prediccion = st.session_state.model.predict(input_scaled)[0]
        
        # Mostrar resultados (0: Local, 1: Empate, 2: Visitante)
        st.write("### Resultados de la Red Neuronal:")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Victoria Local", f"{prediccion[0]*100:.1f}%")
        res_col2.metric("Empate", f"{prediccion[1]*100:.1f}%")
        res_col3.metric("Victoria Visitante", f"{prediccion[2]*100:.1f}%")
