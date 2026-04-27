import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# === CONFIGURAÇÃO DA PÁGINA ===
st.set_page_config(
    page_title="Diabetes Risk Classifier",
    page_icon="🏥",
    layout="wide"
)

# === CSS CUSTOMIZADO ===
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.4rem; font-weight: bold;
        margin: 1rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.4rem; font-weight: bold;
        margin: 1rem 0;
    }
    .metric-card {
        background: white; border-radius: 10px; padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #2d3436; }
    .metric-label { font-size: 0.85rem; color: #636e72; margin-top: 0.2rem; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #2d3436;
        margin-bottom: 1rem; padding-bottom: 0.5rem;
        border-bottom: 2px solid #dfe6e9;
    }
    .disclaimer {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        font-size: 0.85rem; color: #856404; margin-top: 1.5rem;
    }
    .input-section {
        background: white; border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# === CARREGAMENTO DOS MODELOS ===
@st.cache_resource
def carregar_modelos():
    modelo  = joblib.load("diabetes_model.pkl")
    scaler  = joblib.load("diabetes_scaler.pkl")
    smoking = joblib.load("smoking_mapping.pkl")
    return modelo, scaler, smoking

modelo, scaler, smoking_mapping = carregar_modelos()

# === HEADER ===
st.markdown("# 🏥 Diabetes Risk Classifier")
st.markdown("Estimativa de risco de diabetes baseada em dados clínicos — modelo Random Forest treinado com 100.000 pacientes.")
st.divider()

# === LAYOUT PRINCIPAL ===
col_inputs, col_results = st.columns([1, 1], gap="large")

with col_inputs:
    st.markdown('<div class="section-header">📋 Dados do paciente</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown("**Informações gerais**")
        gender = st.selectbox("Gênero", ["Female", "Male"],
                              format_func=lambda x: "Feminino" if x == "Female" else "Masculino")
        age    = st.slider("Idade", min_value=1, max_value=100, value=40)

        col_a, col_b = st.columns(2)
        with col_a:
            hypertension  = st.selectbox("Hipertensão", ["Não", "Sim"])
        with col_b:
            heart_disease = st.selectbox("Doença cardíaca", ["Não", "Sim"])

        smoking_history = st.selectbox("Histórico de fumo", list(smoking_mapping.keys()),
            format_func=lambda x: {
                "never": "Nunca fumou", "No Info": "Sem informação",
                "current": "Fumante atual", "former": "Ex-fumante",
                "ever": "Já fumou", "not current": "Não fuma atualmente"
            }.get(x, x))

    st.markdown("---")
    st.markdown("**Indicadores clínicos**")

    bmi = st.slider("IMC (Body Mass Index)", min_value=10.0,
                    max_value=60.0, value=25.0, step=0.1,
                    help="Índice de Massa Corporal — peso(kg) / altura²(m)")

    hba1c = st.slider("Nível de HbA1c (%)", min_value=3.5,
                      max_value=9.0, value=5.5, step=0.1,
                      help="Hemoglobina glicada — principal marcador de diabetes")

    glucose = st.slider("Glicose no sangue (mg/dL)", min_value=80,
                        max_value=300, value=130,
                        help="Nível de glicose em jejum")

    st.markdown("")
    calcular = st.button("🔍 Analisar risco", use_container_width=True, type="primary")

# === RESULTADOS ===
with col_results:
    st.markdown('<div class="section-header">📊 Resultado da análise</div>', unsafe_allow_html=True)

    if not calcular:
        st.info("Preencha os dados ao lado e clique em **Analisar risco** para ver o resultado.")

    else:
        # Preparando entrada
        gender_enc  = 0 if gender == "Female" else 1
        hypert_enc  = 1 if hypertension == "Sim" else 0
        heart_enc   = 1 if heart_disease == "Sim" else 0
        smoking_enc = smoking_mapping[smoking_history]

        entrada = np.array([[
            gender_enc, age, hypert_enc, heart_enc,
            smoking_enc, bmi, hba1c, glucose
        ]])
        entrada_scaled = scaler.transform(entrada)

        # Predição
        probabilidade = modelo.predict_proba(entrada_scaled)[0][1]
        risco_alto    = probabilidade >= 0.30

        # Card de resultado
        if risco_alto:
            st.markdown(f'<div class="risk-high">⚠️ ALTO RISCO DE DIABETES<br><span style="font-size:1rem;opacity:0.9">Probabilidade estimada: {probabilidade*100:.1f}%</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✅ BAIXO RISCO DE DIABETES<br><span style="font-size:1rem;opacity:0.9">Probabilidade estimada: {probabilidade*100:.1f}%</span></div>', unsafe_allow_html=True)

        # Barra de probabilidade
        st.progress(float(probabilidade))

        # Métricas rápidas
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{probabilidade*100:.1f}%</div><div class="metric-label">Probabilidade</div></div>', unsafe_allow_html=True)
        with m2:
            imc_cat = "Normal" if bmi < 25 else "Sobrepeso" if bmi < 30 else "Obesidade"
            st.markdown(f'<div class="metric-card"><div class="metric-value">{bmi:.1f}</div><div class="metric-label">IMC — {imc_cat}</div></div>', unsafe_allow_html=True)
        with m3:
            hba1c_cat = "Normal" if hba1c < 5.7 else "Pré-diab." if hba1c < 6.5 else "Diabético"
            st.markdown(f'<div class="metric-card"><div class="metric-value">{hba1c:.1f}%</div><div class="metric-label">HbA1c — {hba1c_cat}</div></div>', unsafe_allow_html=True)

        # === SHAP — Explicação individual ===
        st.markdown("---")
        st.markdown("**🔎 Por que o modelo retornou esse resultado?**")
        st.caption("O gráfico abaixo mostra quanto cada variável contribuiu para esta predição específica.")

        with st.spinner("Calculando explicação..."):
            explainer   = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(entrada_scaled)

            feature_names_pt = {
                "gender_encoded": "Gênero",
                "age": "Idade",
                "hypertension": "Hipertensão",
                "heart_disease": "Doença cardíaca",
                "smoking_encoded": "Histórico de fumo",
                "bmi": "IMC",
                "HbA1c_level": "HbA1c",
                "blood_glucose_level": "Glicose"
            }

            features = ["gender_encoded", "age", "hypertension", "heart_disease",
                        "smoking_encoded", "bmi", "HbA1c_level", "blood_glucose_level"]
            nomes_pt  = [feature_names_pt[f] for f in features]

            # Valores SHAP para classe positiva (diabetes)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0] if shap_values.ndim == 3 else shap_values[0]

            # Ordenando por valor absoluto
            # Garantindo arrays 1D limpos
            sv_flat  = np.array(sv).flatten()[:8]
            idx_flat = np.argsort(np.abs(sv_flat))

            sv_ord = sv_flat[idx_flat]
            nm_ord = [nomes_pt[int(i)] for i in idx_flat]

            fig, ax = plt.subplots(figsize=(7, 4))
            cores = ["#e74c3c" if v > 0 else "#3498db" for v in sv_ord]
            bars  = ax.barh(nm_ord, sv_ord, color=cores, edgecolor="white", height=0.6)

            ax.axvline(x=0, color="#2d3436", linewidth=0.8)
            ax.set_xlabel("Contribuição Para o Risco", fontsize=10)
            ax.set_title("Fatores que Influenciaram essa Predição:", fontsize=11, fontweight="bold")

            for bar, val in zip(bars, sv_ord):
                offset = 0.002 if val >= 0 else -0.002
                ha     = "left" if val >= 0 else "right"
                ax.text(val + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}",
                        va="center",
                        ha=ha,
                        fontsize=8,
                        color="#2d3436")

            ax.set_xlim(
                min(sv_ord) - abs(min(sv_ord)) * 0.35,
                max(sv_ord) + abs(max(sv_ord)) * 0.35
            )
            ax.tick_params(axis='y', labelsize=9)
            ax.set_ylabel("")

            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("""
        <div style="font-size:0.85rem; color:#636e72; margin-top:0.5rem;">
        🔴 <b>Barras Vermelhas</b> — Variáveis que <b>aumentaram</b> o risco estimado<br>
        🔵 <b>Barras Azuis</b> — Variáveis que <b>reduziram</b> o risco estimado
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Aviso:</b> Este app é um projeto educacional para evolução acadêmica e
        <b>não substitui avaliação médica profissional</b>. Consulte um médico para diagnóstico!
        </div>
        """, unsafe_allow_html=True)