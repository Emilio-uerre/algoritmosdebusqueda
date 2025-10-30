import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# ------------------ Configuración de la app ------------------
st.set_page_config(page_title="K-Means con PCA y Comparativa", layout="wide")
st.title("🎯 Clustering Interactivo con K-Means y PCA (Comparación Antes/Después)")
st.write("""
Sube tus datos, aplica **K-Means**, y observa cómo el algoritmo agrupa los puntos en un espacio reducido con **PCA (2D o 3D)**.  
También puedes comparar la distribución **antes y después** del clustering.
""")

# ------------------ Carga de datos ------------------
st.sidebar.header("📂 Subir datos")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if uploaded_file is None:
    st.info("👈 Carga un archivo CSV en la barra lateral para comenzar.")
    st.write("""
    **Ejemplo de formato:**
    | Ingreso_Anual | Gasto_Tienda | Edad |
    |---------------|--------------|------|
    | 45000 | 350 | 28 |
    | 72000 | 680 | 35 |
    | 28000 | 210 | 22 |
    """)
else:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Archivo cargado correctamente.")
    st.write("### Vista previa de los datos:")
    st.dataframe(data.head())

    # Solo columnas numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("⚠️ El archivo debe contener al menos **dos** columnas numéricas.")
        st.stop()

    # ------------------ Controles de modelo ------------------
    st.sidebar.header("⚙️ Configuración del modelo")

    # Columnas a usar
    selected_cols = st.sidebar.multiselect(
        "Selecciona las columnas numéricas para el clustering:",
        numeric_cols,
        default=numeric_cols
    )
    if len(selected_cols) < 2:
        st.warning("⚠️ Selecciona al menos **dos** columnas para aplicar PCA.")
        st.stop()

    # k y visualización
    k = st.sidebar.slider("Número de clusters (k):", 1, 10, 3)
    n_components = st.sidebar.radio("Visualización PCA:", [2, 3], index=0)

    # 🔽 Parámetros que pide la actividad
    init = st.sidebar.selectbox(
        "init (método de inicialización)",
        options=["k-means++", "random"],  # los principales en scikit-learn
        index=0,
        help="k-means++ acelera la convergencia; random usa centroides iniciales aleatorios."
    )
    max_iter = st.sidebar.number_input(
        "max_iter (máximo de iteraciones)",
        min_value=10, max_value=5000, value=300, step=10
    )
    # Desde scikit-learn 1.4 el default recomendado es 'auto'; lo exponemos también
    n_init_choice = st.sidebar.selectbox(
        "n_init (repeticiones con semillas distintas)",
        options=["auto", "valor entero"],
        index=0
    )
    if n_init_choice == "valor entero":
        n_init = st.sidebar.number_input("n_init =", min_value=1, max_value=100, value=10, step=1)
    else:
        n_init = "auto"

    use_random_state = st.sidebar.checkbox("Fijar random_state (para resultados reproducibles)", value=True)
    if use_random_state:
        random_state = st.sidebar.number_input("random_state =", min_value=0, max_value=10_000, value=0, step=1)
    else:
        random_state = None  # scikit-learn tomará None (=aleatorio cada vez)

    # ------------------ Entrenamiento ------------------
    X = data[selected_cols].copy()

    # Construimos el modelo con los parámetros elegidos
    kmeans = KMeans(
        n_clusters=k,
        init=init,
        max_iter=int(max_iter),
        n_init=n_init,
        random_state=None if random_state is None else int(random_state)
    )
    kmeans.fit(X)
    data['Cluster'] = kmeans.labels_

    # ------------------ PCA ------------------
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_cols = [f'PCA{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_cols)
    pca_df['Cluster'] = data['Cluster'].astype(str)

    # ------------------ Visualización: antes ------------------
    st.subheader("📊 Distribución original (antes de K-Means)")
    if n_components == 2:
        fig_before = px.scatter(
            pd.DataFrame(X_pca, columns=pca_cols),
            x='PCA1', y='PCA2',
            title="Datos originales proyectados con PCA (sin agrupar)",
            color_discrete_sequence=["gray"]
        )
    else:
        fig_before = px.scatter_3d(
            pd.DataFrame(X_pca, columns=pca_cols),
            x='PCA1', y='PCA2', z='PCA3',
            title="Datos originales proyectados con PCA (sin agrupar)",
            color_discrete_sequence=["gray"]
        )
    st.plotly_chart(fig_before, use_container_width=True)

    # ------------------ Visualización: después ------------------
    st.subheader(f"🎯 Datos agrupados con K-Means (k = {k})")
    if n_components == 2:
        fig_after = px.scatter(
            pca_df, x='PCA1', y='PCA2', color='Cluster',
            title="Clusters visualizados en 2D con PCA",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
    else:
        fig_after = px.scatter_3d(
            pca_df, x='PCA1', y='PCA2', z='PCA3', color='Cluster',
            title="Clusters visualizados en 3D con PCA",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
    st.plotly_chart(fig_after, use_container_width=True)

    # ------------------ Centroides en PCA ------------------
    st.subheader("📍 Centroides de los clusters (en espacio PCA)")
    centroides_pca = pd.DataFrame(pca.transform(kmeans.cluster_centers_), columns=pca_cols)
    st.dataframe(centroides_pca)

    # ------------------ Método del Codo ------------------
    st.subheader("📈 Método del Codo (Elbow Method)")
    if st.button("Calcular número óptimo de clusters"):
        inertias = []
        K = range(1, 11)
        for i in K:
            km = KMeans(
                n_clusters=i,
                init=init,
                max_iter=int(max_iter),
                n_init=n_init,
                random_state=None if random_state is None else int(random_state)
            )
            km.fit(X)
            inertias.append(km.inertia_)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(list(K), inertias, 'bo-')
        ax2.set_title('Método del Codo')
        ax2.set_xlabel('Número de Clusters (k)')
        ax2.set_ylabel('Inercia (SSE)')
        ax2.grid(True)
        st.pyplot(fig2)

    # ------------------ Descarga ------------------
    st.subheader("💾 Descargar datos con clusters asignados")
    buffer = BytesIO()
    data.to_csv(buffer, index=False, encoding="utf-8")
    buffer.seek(0)
    st.download_button(
        label="⬇️ Descargar CSV con Clusters",
        data=buffer,
        file_name="datos_clusterizados.csv",
        mime="text/csv"
    )
