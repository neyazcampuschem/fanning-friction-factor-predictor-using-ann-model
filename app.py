import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="ANN Fanning Friction Factor Predictor",
    page_icon="⚡",
    layout="wide",
)

# Custom CSS matching the screenshot UI
st.markdown(
    """
<style>
    /* Global font and background styling */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] * {
        color: #94A3B8;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    
    /* Sidebar Navigation Links */
    .nav-header {
        font-size: 20px;
        font-weight: bold;
        color: #FFFFFF !important;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Top Header Badge */
    .status-badge {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        float: right;
    }
    
    /* Feature Highlights Bar */
    .feature-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Container Cards */
    .section-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    
    /* Guidance Tooltip Boxes */
    .info-subtext {
        background-color: #EFF6FF;
        color: #1E40AF;
        font-size: 11px;
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 4px;
    }

    /* Result Metric Boxes */
    .result-box-blue {
        background-color: #EFF6FF;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #DBEAFE;
    }
    .result-box-purple {
        background-color: #F5F3FF;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #DDD6FE;
    }
    .result-box-green {
        background-color: #F0FDF4;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #DCFCE7;
    }
    
    /* Model Performance Cards */
    .metric-box-mae {
        background-color: #FEF2F2;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #FEE2E2;
    }
    .metric-box-mse {
        background-color: #EEF2FF;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #E0E7FF;
    }
    
    /* Footer */
    .footer-text {
        font-size: 12px;
        color: #64748B;
        text-align: center;
        margin-top: 30px;
        border-top: 1px solid #E2E8F0;
        padding-top: 15px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# MODEL TRAINING & CACHING
# ==========================================
@st.cache_resource
def load_and_train():
    df_lam = pd.read_excel("ANN_data.xlsx").dropna(subset=["Re", "F"])
    X_lam = df_lam[["Re"]].values
    y_lam = df_lam["F"].values

    df_tur = pd.read_excel("ANN_data_tur.xlsx").dropna(
        subset=["Reynold Number (Re)", "Frictional Factor F=0.0791/Re^0.25"]
    )
    X_tur = df_tur[["Reynold Number (Re)"]].values
    y_tur = df_tur["Frictional Factor F=0.0791/Re^0.25"].values

    try:
        df_haal = pd.read_excel("ANN_data_FF_HAAL.xlsx")
    except FileNotFoundError:
        df_haal = pd.read_excel("ANN_data_FF_HAAL_2.xlsx")

    df_haal = df_haal.dropna(subset=["Re", "k/D", "FF"])
    X_haal = np.column_stack(
        [np.log10(df_haal["Re"].values), df_haal["k/D"].values]
    )
    y_haal = df_haal["FF"].values

    s_X_lam, s_y_lam = StandardScaler(), StandardScaler()
    X_lam_sc = s_X_lam.fit_transform(X_lam)
    y_lam_sc = s_y_lam.fit_transform(y_lam.reshape(-1, 1))

    s_X_tur, s_y_tur = StandardScaler(), StandardScaler()
    X_tur_sc = s_X_tur.fit_transform(X_tur)
    y_tur_sc = s_y_tur.fit_transform(y_tur.reshape(-1, 1))

    s_X_haal, s_y_haal = StandardScaler(), StandardScaler()
    X_haal_sc = s_X_haal.fit_transform(X_haal)
    y_haal_sc = s_y_haal.fit_transform(y_haal.reshape(-1, 1))

    m_lam = Sequential(
        [
            Input(shape=(1,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    m_lam.compile(optimizer="adam", loss="mse")
    m_lam.fit(X_lam_sc, y_lam_sc, epochs=150, batch_size=32, verbose=0)

    m_tur = Sequential(
        [
            Input(shape=(1,)),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    m_tur.compile(optimizer="adam", loss="mse")
    m_tur.fit(X_tur_sc, y_tur_sc, epochs=150, batch_size=32, verbose=0)

    m_haal = Sequential(
        [
            Input(shape=(2,)),
            Dense(128, activation="swish"),
            Dense(128, activation="swish"),
            Dense(64, activation="swish"),
            Dense(32, activation="swish"),
            Dense(1),
        ]
    )
    m_haal.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse"
    )
    m_haal.fit(X_haal_sc, y_haal_sc, epochs=250, batch_size=64, verbose=0)

    def calc_m(model, X_sc, y_true, s_y):
        p_sc = model.predict(X_sc, verbose=0)
        y_pred = s_y.inverse_transform(p_sc).ravel()
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return mae, mse

    metrics = {
        "laminar": calc_m(m_lam, X_lam_sc, y_lam, s_y_lam),
        "turbulent": calc_m(m_tur, X_tur_sc, y_tur, s_y_tur),
        "rough": calc_m(m_haal, X_haal_sc, y_haal, s_y_haal),
    }

    models = {"laminar": m_lam, "turbulent": m_tur, "rough": m_haal}
    scalers = {
        "lam_X": s_X_lam,
        "lam_y": s_y_lam,
        "tur_X": s_X_tur,
        "tur_y": s_y_tur,
        "haal_X": s_X_haal,
        "haal_y": s_y_haal,
    }

    return models, scalers, metrics


models, scalers, metrics = load_and_train()

# ==========================================
# SIDEBAR DESIGN
# ==========================================
with st.sidebar:
    st.markdown("### 🚰 ANN Friction Factor Predictor")
    st.markdown("---")

    selected_nav = st.radio(
        "Navigation",
        options=["Predict", "About", "Theory", "Data & Model", "Contact"],
        label_visibility="collapsed",
    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown(
        """
        <div style="font-size: 11px; line-height: 1.6;">
            <p>🎓 <strong>Designed by</strong><br><span style="color:#FFFFFF;">Neyaz Reza</span> (M.Tech Student)</p>
            <p>👤 <strong>Under the Supervision of</strong><br><span style="color:#FFFFFF;">Mr. Mohammad Abdul Hakeem</span><br>(Associate Professor)</p>
            <p>🏛️ <strong>Department of Chemical Engineering</strong><br>Aligarh Muslim University</p>
        </div>
        <br>
        <p style="font-style: italic; font-size: 11px; color: #64748B;">"From Data to Fluid Dynamics"</p>
        """,
        unsafe_allow_html=True,
    )

# ==========================================
# MAIN INTERFACE
# ==========================================

# Top Title Header
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title("ANN Fanning Friction Factor Predictor")
    st.caption(
        "Predict fluid friction factors using trained Artificial Neural Networks across laminar, turbulent, and rough pipe flow regimes."
    )

with col_badge:
    st.markdown(
        '<div class="status-badge">📊 Deployed Model &nbsp;🟢 Online</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# 4 Feature Highlights
f1, f2, f3, f4 = st.columns(4)
with f1:
    st.markdown(
        '<div class="feature-card">⚡ <div><strong>Fast Prediction</strong><br><small style="color:#64748B;">Instant results</small></div></div>',
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        '<div class="feature-card">🎯 <div><strong>High Accuracy</strong><br><small style="color:#64748B;">Trained on reliable correlations</small></div></div>',
        unsafe_allow_html=True,
    )
with f3:
    st.markdown(
        '<div class="feature-card">🗄️ <div><strong>Multiple Regimes</strong><br><small style="color:#64748B;">Laminar, Turbulent, Rough</small></div></div>',
        unsafe_allow_html=True,
    )
with f4:
    st.markdown(
        '<div class="feature-card">⚙️ <div><strong>User Friendly</strong><br><small style="color:#64748B;">Simple & Intuitive Interface</small></div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Form Container
st.markdown("### 🎛️ Input Parameters")
st.caption("Set the flow conditions to predict the Fanning friction factor.")

p1, p2, p3, p4 = st.columns(4)

with p1:
    status = st.selectbox("Pipe Surface Condition 🛈", options=["Rough", "Smooth"])

if status == "Smooth":
    with p2:
        flow_type = st.selectbox(
            "Flow Regime 🛈", options=["Laminar", "Turbulent"]
        )

    if flow_type == "Laminar":
        with p3:
            re = st.number_input(
                "Reynolds Number (Re) 🛈",
                min_value=0.1,
                max_value=2100.0,
                value=1000.0,
            )
        with p4:
            kD = st.number_input(
                "Relative Roughness (k/D) 🛈",
                value=0.00000,
                format="%.5f",
                disabled=True,
            )

        st.markdown(
            """
        <div style="display:flex; gap:10px;">
            <div class="info-subtext" style="flex:1;">ℹ️ Select the internal surface condition of the pipe.</div>
            <div class="info-subtext" style="flex:1;">ℹ️ Automatically suggested based on Re and k/D.</div>
            <div class="info-subtext" style="flex:1;">ℹ️ Enter Reynolds number (typically 0.1 - 2100).</div>
            <div class="info-subtext" style="flex:1;">ℹ️ Smooth pipe regime (k/D = 0).</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        actual_f = 16.0 / re
        re_sc = scalers["lam_X"].transform([[re]])
        p_sc = models["laminar"].predict(re_sc, verbose=0)
        predicted_f = scalers["lam_y"].inverse_transform(p_sc)[0][0]
        cur_m = metrics["laminar"]
        regime_label = "Laminar Model"

    else:
        with p3:
            re = st.number_input(
                "Reynolds Number (Re) 🛈",
                min_value=2100.0,
                max_value=100000.0,
                value=10000.0,
            )
        with p4:
            kD = st.number_input(
                "Relative Roughness (k/D) 🛈",
                value=0.00000,
                format="%.5f",
                disabled=True,
            )

        st.markdown(
            """
        <div style="display:flex; gap:10px;">
            <div class="info-subtext" style="flex:1;">ℹ️ Select the internal surface condition of the pipe.</div>
            <div class="info-subtext" style="flex:1;">ℹ️ Automatically suggested based on Re and k/D.</div>
            <div class="info-subtext" style="flex:1;">ℹ️ Enter Reynolds number (typically 2100 - 10⁵).</div>
            <div class="info-subtext" style="flex:1;">ℹ️ Smooth pipe regime (k/D = 0).</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        actual_f = 0.0791 / (re**0.25)
        re_sc = scalers["tur_X"].transform([[re]])
        p_sc = models["turbulent"].predict(re_sc, verbose=0)
        predicted_f = scalers["tur_y"].inverse_transform(p_sc)[0][0]
        cur_m = metrics["turbulent"]
        regime_label = "Turbulent Model"

else:
    flow_type = "Turbulent (Rough)"
    with p2:
        st.selectbox("Flow Regime 🛈", options=["Turbulent (Rough)"])

    with p3:
        re = st.number_input(
            "Reynolds Number (Re) 🛈",
            min_value=1000.0,
            max_value=100000000.0,
            value=50000.0,
        )

    with p4:
        kD = st.number_input(
            "Relative Roughness (k/D) 🛈",
            min_value=0.0,
            max_value=0.05,
            value=0.00100,
            format="%.5f",
        )

    st.markdown(
        """
    <div style="display:flex; gap:10px; margin-top:5px;">
        <div class="info-subtext" style="flex:1;">ℹ️ Select the internal surface condition of the pipe.</div>
        <div class="info-subtext" style="flex:1;">ℹ️ Automatically suggested based on Re and k/D.</div>
        <div class="info-subtext" style="flex:1;">ℹ️ Enter Reynolds number (typically 10² – 10⁸).</div>
        <div class="info-subtext" style="flex:1;">ℹ️ Enter relative roughness (typically 10⁻⁶ – 10⁻¹).</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    term = (6.9 / re) if kD == 0 else ((kD / 3.7) ** 1.11) + (6.9 / re)
    f_darcy = (1.0 / (-1.8 * np.log10(term))) ** 2
    actual_f = f_darcy / 16.0

    inp_sc = scalers["haal_X"].transform([[np.log10(re), kD]])
    p_sc = models["rough"].predict(inp_sc, verbose=0)
    predicted_f = scalers["haal_y"].inverse_transform(p_sc)[0][0]
    cur_m = metrics["rough"]
    regime_label = "Rough Model"

st.markdown("<br>", unsafe_allow_html=True)

# Action Buttons
btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    predict_clicked = st.button(
        "▶ Predict Friction Factor", use_container_width=True, type="primary"
    )
with btn_col2:
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Output Calculations
error = abs(predicted_f - actual_f)
rel_error = (error / actual_f) * 100

# Results & Metrics Cards Side by Side
res_left, res_right = st.columns([1.2, 1])

with res_left:
    st.markdown("### 📊 Prediction Results")
    st.caption("Friction factor predictions and analysis")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(
            f"""
        <div class="result-box-blue">
            <small style="color:#1E40AF; font-weight:600;">ANN Predicted</small>
            <h2 style="color:#1E3A8A; margin: 8px 0;">{predicted_f:.6f}</h2>
            <small style="color:#64748B;">Fanning Friction Factor</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with r2:
        st.markdown(
            f"""
        <div class="result-box-purple">
            <small style="color:#5B21B6; font-weight:600;">Actual / Analytical</small>
            <h2 style="color:#4C1D95; margin: 8px 0;">{actual_f:.6f}</h2>
            <small style="color:#64748B;">Reference Value</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with r3:
        st.markdown(
            f"""
        <div class="result-box-green">
            <small style="color:#166534; font-weight:600;">Relative Error</small>
            <h2 style="color:#14532D; margin: 8px 0;">{rel_error:.2f}%</h2>
            <small style="color:#64748B;">Prediction Accuracy</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

with res_right:
    st.markdown(f"### 🏆 Model Performance ({regime_label})")
    st.caption("Performance metrics on test data")

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f"""
        <div class="metric-box-mae">
            <small style="color:#991B1B; font-weight:600;">Mean Absolute Error (MAE)</small>
            <h3 style="color:#7F1D1D; margin: 12px 0;">{cur_m[0]:.6e}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
        <div class="metric-box-mse">
            <small style="color:#3730A3; font-weight:600;">Mean Squared Error (MSE)</small>
            <h3 style="color:#312E81; margin: 12px 0;">{cur_m[1]:.6e}</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# Note Card
st.info(
    "ℹ️ **Note:** This tool uses trained Artificial Neural Networks (ANN) to predict the Fanning friction factor for different pipe flow regimes. Ensure input parameters are within the range of the training data for best results."
)

# Footer
st.markdown(
    """
<div class="footer-text">
    © 2025 | ANN Friction Factor Predictor &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; Designed by Neyaz Reza &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; Department of Chemical Engineering, AMU ❤️
</div>
""",
    unsafe_allow_html=True,
)