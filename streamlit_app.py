import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# Configuração da página
# =====================================
st.set_page_config(page_title="Simulador SRAG — Antivirais", layout="centered")
st.title("🧬 Simulador de Tratamento — SRAG/INFLUD")
st.markdown("Simule pacientes e descubra qual antiviral seria recomendado segundo o modelo treinado.")

# =====================================
# Carregar modelo
# =====================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("modelo_antiviral.pkl")
        st.success("✅ Modelo carregado com sucesso!")
        return model
    except Exception as e:
        st.error("❌ Erro ao carregar o modelo. Certifique-se de ter 'modelo_antiviral.pkl' no diretório.")
        st.stop()

model = load_model()

# =====================================
# Função auxiliar
# =====================================
def binarize(val):
    return 1 if val == "Sim" else 0

# =====================================
# Simulação de paciente
# =====================================
st.header("👩‍⚕️ Dados do Paciente")

col1, col2 = st.columns(2)

with col1:
    idade = st.slider("Idade (anos)", 0, 100, 45)
    sexo = st.selectbox("Sexo", ["Masculino", "Feminino"])
    febre = st.selectbox("Febre", ["Sim", "Não"])
    tosse = st.selectbox("Tosse", ["Sim", "Não"])
    dispneia = st.selectbox("Dispneia", ["Sim", "Não"])
    saturacao = st.selectbox("Saturação <95%", ["Sim", "Não"])
    diarreia = st.selectbox("Diarreia", ["Sim", "Não"])
    vomito = st.selectbox("Vômito", ["Sim", "Não"])
    dor_abd = st.selectbox("Dor abdominal", ["Sim", "Não"])

with col2:
    diabetes = st.selectbox("Diabetes", ["Sim", "Não"])
    cardiopatia = st.selectbox("Cardiopatia", ["Sim", "Não"])
    asma = st.selectbox("Asma", ["Sim", "Não"])
    renal = st.selectbox("Doença renal crônica", ["Sim", "Não"])
    obesidade = st.selectbox("Obesidade", ["Sim", "Não"])
    imunodepre = st.selectbox("Imunodepressão", ["Sim", "Não"])
    neurologic = st.selectbox("Doença neurológica", ["Sim", "Não"])
    hepatica = st.selectbox("Doença hepática", ["Sim", "Não"])
    pneumopati = st.selectbox("Pneumopatia", ["Sim", "Não"])

# =====================================
# Montagem do vetor de entrada compatível
# =====================================
X_new = pd.DataFrame([{
    "age": idade,
    "sex_m": 1 if sexo == "Masculino" else 0,
    "febre_bin": binarize(febre),
    "tosse_bin": binarize(tosse),
    "dispneia_bin": binarize(dispneia),
    "saturacao_bin": binarize(saturacao),
    "diarreia_bin": binarize(diarreia),
    "vomito_bin": binarize(vomito),
    "dor_abd_bin": binarize(dor_abd),
    "diabetes_bin": binarize(diabetes),
    "cardiopati_bin": binarize(cardiopatia),
    "asma_bin": binarize(asma),
    "renal_bin": binarize(renal),
    "obesidade_bin": binarize(obesidade),
    "imunodepre_bin": binarize(imunodepre),
    "neurologic_bin": binarize(neurologic),
    "hepatica_bin": binarize(hepatica),
    "pneumopati_bin": binarize(pneumopati),
    # demais features ausentes preenchidas com 0
    **{col: 0 for col in [
        'out_morbi_bin', 'nosocomial_bin', 'ave_suino_bin', 'garganta_bin', 'desc_resp_bin',
        'perd_olft_bin', 'perd_pala_bin', 'fadiga_bin', 'fator_risc_bin', 'puerpera_bin',
        'hematologi_bin', 'sind_down_bin', 'obes_imc_bin', 'tabag_bin', 'dt_interna_bin',
        'dt_evoluca_bin', 'evolucao_bin', 'antiviral_bin', 'nu_idade_n_bin', 'idade_bin',
        'cs_sexo_bin', 'cs_gestant_bin', 'vacina_cov_bin', 'vacina_bin',
        'out_antiv_bin', 'hospital_bin', 'uti_bin', 'dt_entuti_bin', 'dt_saiduti_bin',
        'suport_ven_bin', 'raiox_res_bin', 'tomo_res_bin', 'tp_amostra_bin', 'pos_an_flu_bin',
        'tp_flu_an_bin', 'co_detec_bin', 'classi_fin_bin'
    ]}
}])

# =====================================
# Corrigir ordem das colunas para coincidir com o modelo
# =====================================
feature_order = [
 'age', 'sex_m', 'out_morbi_bin', 'diabetes_bin', 'cardiopati_bin', 'nosocomial_bin',
 'asma_bin', 'ave_suino_bin', 'febre_bin', 'tosse_bin', 'dispneia_bin', 'garganta_bin',
 'desc_resp_bin', 'saturacao_bin', 'diarreia_bin', 'vomito_bin', 'dor_abd_bin',
 'perd_olft_bin', 'perd_pala_bin', 'fadiga_bin', 'fator_risc_bin', 'puerpera_bin',
 'hematologi_bin', 'sind_down_bin', 'hepatica_bin', 'renal_bin', 'imunodepre_bin',
 'neurologic_bin', 'pneumopati_bin', 'obesidade_bin', 'obes_imc_bin', 'tabag_bin',
 'dt_interna_bin', 'dt_evoluca_bin', 'evolucao_bin', 'antiviral_bin', 'nu_idade_n_bin',
 'idade_bin', 'cs_sexo_bin', 'cs_gestant_bin', 'vacina_cov_bin', 'vacina_bin',
 'out_antiv_bin', 'hospital_bin', 'uti_bin', 'dt_entuti_bin', 'dt_saiduti_bin',
 'suport_ven_bin', 'raiox_res_bin', 'tomo_res_bin', 'tp_amostra_bin', 'pos_an_flu_bin',
 'tp_flu_an_bin', 'co_detec_bin', 'classi_fin_bin'
]

X_new = X_new.reindex(columns=feature_order, fill_value=0)

st.write("### 🧾 Dados processados")
st.dataframe(X_new)

# =====================================
# Predição do antiviral recomendado
# =====================================
if st.button("🔮 Prever antiviral recomendado"):
    try:
        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new).max()

        antivirais = {
            0: "Nenhum antiviral recomendado",
            1: "🧪 Oseltamivir",
            2: "💊 Zanamivir",
            3: "Outro antiviral"
        }

        recomendacao = antivirais.get(int(pred), "Desconhecido")

        st.subheader("🏥 Resultado da predição:")
        st.success(f"**Recomendação:** {recomendacao}")
        st.caption(f"Confiança da predição: {prob:.2%}")

    except Exception as e:
        st.error(f"Erro ao realizar predição: {e}")
