import sys
from pathlib import Path
import streamlit as st

# Agregar 'src' al path para que el módulo 'credit_risk' sea visible en Streamlit Cloud
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm

# Imports de los módulos del proyecto
from credit_risk.model import split_dataset, fit_logistic_model, calculate_ks_statistic
from credit_risk.monte_carlo import simulate_losses

# --- Configuración de página ---
st.set_page_config(
    page_title="Credit Risk Monte Carlo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Colores Institucionales ---
COLORS = {
    "primary": "#1e40af",     # azul
    "danger": "#dc2626",      # rojo
    "success": "#16a34a",     # verde
    "neutral": "#6b7280",     # gris
    "warning": "#d97706",     # naranja
}

# --- Funciones Auxiliares ---
def _generate_portfolio(mean_pd: float, pd_std: float, mean_ead: float, ead_std: float, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    
    # Calcular parámetros alpha y beta para la distribución Beta de PDs
    var = pd_std**2
    max_var = mean_pd * (1 - mean_pd)
    if var >= max_var:
        var = max_var * 0.99
    
    alpha = mean_pd * ((mean_pd * (1 - mean_pd) / var) - 1)
    beta = (1 - mean_pd) * ((mean_pd * (1 - mean_pd) / var) - 1)
    
    pd_values = rng.beta(alpha, beta, n)
    
    # Generar targets binarios
    loan_status = rng.binomial(1, pd_values)
    
    # Generar EAD (Montos de préstamo)
    loan_amnt = rng.normal(mean_ead, ead_std, n).clip(min=1000)
    
    # Generar features correlacionadas artificialmente con PD para que el modelo funcione bien
    person_income = (120000 - 150000 * pd_values + rng.normal(0, 10000, n)).clip(min=15000)
    loan_int_rate = (5 + 20 * pd_values + rng.normal(0, 2, n)).clip(min=1.0, max=25.0)
    cb_person_cred_hist_length = (15 - 15 * pd_values + rng.normal(0, 2, n)).clip(min=0)
    
    df = pd.DataFrame({
        "loan_status": loan_status,
        "loan_amnt": loan_amnt,
        "person_income": person_income,
        "loan_int_rate": loan_int_rate,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    })
    return df

def _train_pd_model(df: pd.DataFrame, seed: int):
    # Validar que ambas clases existan antes de modelar
    if df["loan_status"].nunique() < 2:
        raise ValueError("La muestra generada no contiene defaults suficientes o no tiene 'buenos' suficientes. Ajusta la PD media.")
    
    df_train, df_test = split_dataset(df, random_state=seed)
    
    # Ajustar el modelo usando la función del módulo base
    model = fit_logistic_model(df_train)
    
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    
    # Predicciones en test para métricas
    X_test = sm.add_constant(df_test[features], has_constant='add')
    y_test = df_test["loan_status"]
    
    # sm.add_constant a veces falla si test set no tiene varianza, usar reindex
    # para asegurar que 'const' existe si add_constant no lo agrega bien
    if 'const' not in X_test.columns:
        X_test.insert(0, 'const', 1.0)
        
    y_scores = model.predict(X_test).values
    
    auc_roc = roc_auc_score(y_test, y_scores)
    ks_stat, _ = calculate_ks_statistic(y_test, y_scores)
    
    # Predicciones para todo el dataset (para Monte Carlo)
    X_all = sm.add_constant(df[features], has_constant='add')
    if 'const' not in X_all.columns:
        X_all.insert(0, 'const', 1.0)
    df["pd_hat"] = model.predict(X_all).values
    
    # Importancia de variables
    params = model.params.drop("const", errors='ignore')
    feature_importance = pd.DataFrame({
        "feature": params.index,
        "abs_importance": np.abs(params.values)
    }).sort_values("abs_importance", ascending=False)
    
    return model, auc_roc, ks_stat, y_test, y_scores, df, feature_importance

# --- Estado Inicial ---
if "results" not in st.session_state:
    st.session_state["results"] = None
    st.session_state["params"] = None
    st.session_state["simulation_run"] = False

# --- Interfaz Principal ---
st.title("📊 Credit Risk Monte Carlo Dashboard")
st.caption("Pipeline end-to-end de análisis de riesgo crediticio: estimación de PD con ML · simulación Monte Carlo vectorizada · VaR · Expected Shortfall")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Parámetros")
    
    st.subheader("Portafolio")
    n_simulations = st.slider("Número de simulaciones", min_value=1000, max_value=50000, value=10000, step=1000)
    n_exposures = st.slider("Número de deudores", min_value=100, max_value=5000, value=500, step=100)
    seed = st.number_input("Semilla (reproducibilidad)", value=42, step=1)
    
    st.subheader("Probabilidad de Default (PD)")
    mean_pd = st.slider("PD media del portafolio", min_value=0.01, max_value=0.30, value=0.05, step=0.01, format="%.2f", help="Porcentaje del portafolio que se espera que defaultee. Un portafolio retail típico tiene PD media de 3-8%.")
    pd_std = st.slider("Dispersión de PD entre deudores", min_value=0.01, max_value=0.10, value=0.02, step=0.01)
    
    st.subheader("Loss Given Default (LGD)")
    mean_lgd = st.slider("LGD media", min_value=0.20, max_value=0.80, value=0.45, step=0.05, format="%.2f")
    st.info("Nota: LGD estocástica deshabilitada temporalmente por simplificación.")
    
    st.subheader("Exposure at Default (EAD)")
    mean_ead = st.number_input("EAD media por deudor ($)", value=50000, min_value=1000, step=5000)
    ead_std = st.number_input("Desvío estándar EAD ($)", value=20000, min_value=0, step=5000)
    
    st.subheader("Umbrales de Riesgo")
    confidence_level_str = st.radio("Nivel de confianza para VaR/ES", ["95.0%", "99.0%", "99.9%"])
    confidence_level = float(confidence_level_str.strip("%")) / 100.0
    
    run_button = st.button("🚀 Correr simulación", type="primary", use_container_width=True)

# --- Lógica de Ejecución ---
if run_button:
    with st.spinner("Corriendo simulación Monte Carlo..."):
        try:
            # 1. Generar Portafolio
            df_portfolio = _generate_portfolio(mean_pd, pd_std, mean_ead, ead_std, n_exposures, seed)
            
            # 2. Entrenar Modelo de PD
            model, auc_roc, ks_stat, y_test, y_scores, df_scored, feature_importance = _train_pd_model(df_portfolio, seed)
            
            # 3. Monte Carlo
            losses, _ = simulate_losses(
                pd_values=df_scored["pd_hat"].values,
                ead_values=df_scored["loan_amnt"].values,
                lgd=mean_lgd,
                n_simulations=n_simulations,
                seed=seed
            )
            
            # 4. Métricas
            expected_loss = losses.mean()
            var_value = np.percentile(losses, confidence_level * 100)
            expected_shortfall = losses[losses >= var_value].mean() if len(losses[losses >= var_value]) > 0 else var_value
            unexpected_loss = var_value - expected_loss
            
            st.session_state["results"] = {
                "losses": losses,
                "expected_loss": expected_loss,
                "var_value": var_value,
                "expected_shortfall": expected_shortfall,
                "unexpected_loss": unexpected_loss,
                "auc_roc": auc_roc,
                "ks_stat": ks_stat,
                "y_test": y_test,
                "y_scores": y_scores,
                "feature_importance": feature_importance,
                "df_scored": df_scored
            }
            st.session_state["params"] = {
                "confidence_level": confidence_level
            }
            st.session_state["simulation_run"] = True
            
        except Exception as e:
            st.error(f"❌ Error en la simulación: {e}")
            st.stop()

# --- Renderización de Resultados ---
if not st.session_state["simulation_run"]:
    st.info("⚙️ **Configurá los parámetros en el panel izquierdo y presioná 'Correr simulación'.**\n\nLa simulación tarda menos de 2 segundos para los valores por defecto.")
    st.markdown("""
    ### ¿Qué hace este pipeline?
    
    1. **Genera** un portafolio crediticio sintético con los parámetros configurados
    2. **Entrena** un modelo de Logistic Regression para estimar la Probabilidad de Default (PD) de cada deudor
    3. **Simula** 10.000+ escenarios de pérdidas con Monte Carlo vectorizado
    4. **Calcula** VaR, Expected Shortfall y Expected Loss del portafolio
    5. **Muestra** diagnósticos del modelo (AUC-ROC, KS Statistic) y la distribución de pérdidas
    """)
else:
    res = st.session_state["results"]
    prm = st.session_state["params"]
    
    # 5.1 KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label="Expected Loss (EL)",
        value=f"${res['expected_loss']:,.0f}",
        help="Pérdida promedio esperada del portafolio. Es el costo 'normal' del negocio crediticio."
    )
    col2.metric(
        label=f"VaR {prm['confidence_level']*100:.1f}%",
        value=f"${res['var_value']:,.0f}",
        delta=f"${res['unexpected_loss']:,.0f} sobre EL",
        delta_color="inverse",
        help="Value at Risk: la pérdida máxima al nivel de confianza seleccionado."
    )
    col3.metric(
        label=f"Expected Shortfall {prm['confidence_level']*100:.1f}%",
        value=f"${res['expected_shortfall']:,.0f}",
        help="CVaR / Expected Shortfall: pérdida promedio en los peores escenarios (los que superan el VaR). Captura mejor el riesgo de cola."
    )
    col4.metric(
        label="AUC-ROC del modelo PD",
        value=f"{res['auc_roc']:.3f}",
        delta=f"KS: {res['ks_stat']:.3f}",
        help="AUC-ROC: capacidad discriminante del modelo. KS: separación entre buenos y malos pagadores."
    )
    
    st.markdown("---")
    
    # Gráficos
    col_a, col_b = st.columns(2)
    
    # 5.2 Gráfico 1 - Distribución de pérdidas
    with col_a:
        fig_losses = go.Figure()
        fig_losses.add_trace(go.Histogram(
            x=res["losses"],
            nbinsx=80,
            name="Distribución de pérdidas",
            marker_color=COLORS["primary"],
            opacity=0.75,
        ))
        fig_losses.add_vline(
            x=res["expected_loss"],
            line_dash="dash",
            line_color=COLORS["success"],
            annotation_text=f"EL: ${res['expected_loss']:,.0f}",
            annotation_position="top right",
        )
        fig_losses.add_vline(
            x=res["var_value"],
            line_dash="solid",
            line_color=COLORS["danger"],
            annotation_text=f"VaR {prm['confidence_level']*100:.1f}%: ${res['var_value']:,.0f}",
            annotation_position="top right",
        )
        fig_losses.add_vrect(
            x0=res["var_value"], x1=res["losses"].max(),
            fillcolor=COLORS["danger"],
            opacity=0.08,
            annotation_text="Zona de pérdidas extremas",
        )
        fig_losses.update_layout(
            title="Distribución de pérdidas del portafolio (Monte Carlo)",
            xaxis_title="Pérdida total del portafolio ($)",
            yaxis_title="Frecuencia (# simulaciones)",
            showlegend=False,
        )
        st.plotly_chart(fig_losses, use_container_width=True)
        
    # 5.3 Gráfico 2 - Curva ROC
    with col_b:
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_scores"])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"Modelo PD (AUC = {res['auc_roc']:.3f})",
            line=dict(color=COLORS["primary"], width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random (AUC = 0.50)",
            line=dict(color=COLORS["neutral"], width=1, dash="dash"),
        ))
        fig_roc.update_layout(
            title="Curva ROC — Modelo de Scoring PD",
            xaxis_title="Tasa de Falsos Positivos",
            yaxis_title="Tasa de Verdaderos Positivos",
            legend=dict(x=0.6, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    col_c, col_d = st.columns(2)
    
    # 5.4 Gráfico 3 - Distribución de scores (KS plot)
    with col_c:
        buenos = res["y_scores"][res["y_test"].values == 0]
        malos = res["y_scores"][res["y_test"].values == 1]
        
        fig_ks = go.Figure()
        fig_ks.add_trace(go.Histogram(
            x=buenos, name="Buenos pagadores",
            marker_color=COLORS["success"], opacity=0.6, nbinsx=40,
            histnorm="probability density",
        ))
        fig_ks.add_trace(go.Histogram(
            x=malos, name="Malos pagadores (default)",
            marker_color=COLORS["danger"], opacity=0.6, nbinsx=40,
            histnorm="probability density",
        ))
        fig_ks.update_layout(
            barmode="overlay",
            title=f"Distribución de scores PD — KS Statistic: {res['ks_stat']:.3f}",
            xaxis_title="Score de probabilidad de default",
            yaxis_title="Densidad",
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_ks, use_container_width=True)

    # 5.5 Gráfico 4 - Feature Importance
    with col_d:
        fi = res["feature_importance"].head(10)
        fig_fi = go.Figure(go.Bar(
            x=fi["abs_importance"],
            y=fi["feature"],
            orientation="h",
            marker_color=COLORS["primary"],
        ))
        fig_fi.update_layout(
            title="Importancia de variables — Modelo PD",
            xaxis_title="Importancia (|coeficiente|)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    
    # 5.7 Tabla resumen
    st.subheader("Resumen de Riesgo por Segmento")
    df_scored = res["df_scored"]
    
    # Crear segmentos por cuartiles de PD
    p25 = df_scored["pd_hat"].quantile(0.25)
    p75 = df_scored["pd_hat"].quantile(0.75)
    
    conditions = [
        df_scored["pd_hat"] < p25,
        df_scored["pd_hat"] > p75
    ]
    choices = ["Bajo riesgo (PD < p25)", "Alto riesgo (PD > p75)"]
    df_scored["Segmento"] = np.select(conditions, choices, default="Riesgo medio (p25-p75)")
    
    df_resumen = df_scored.groupby("Segmento").agg(
        n_deudores=("loan_status", "count"),
        pd_media=("pd_hat", "mean"),
        ead_media=("loan_amnt", "mean"),
    ).reset_index()
    
    # Calcular EL por segmento (aproximación EL = PD * EAD * LGD)
    df_resumen["Pérdida esperada ($)"] = df_resumen["pd_media"] * df_resumen["ead_media"] * mean_lgd * df_resumen["n_deudores"]
    df_resumen["% del portafolio"] = (df_resumen["n_deudores"] / df_resumen["n_deudores"].sum()) * 100
    
    # Formatear tabla
    df_resumen["pd_media"] = df_resumen["pd_media"].apply(lambda x: f"{x*100:.2f}%")
    df_resumen["ead_media"] = df_resumen["ead_media"].apply(lambda x: f"${x:,.0f}")
    df_resumen["Pérdida esperada ($)"] = df_resumen["Pérdida esperada ($)"].apply(lambda x: f"${x:,.0f}")
    df_resumen["% del portafolio"] = df_resumen["% del portafolio"].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(df_resumen, use_container_width=True, hide_index=True)
