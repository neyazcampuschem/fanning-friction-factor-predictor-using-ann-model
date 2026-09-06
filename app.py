import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="ANN Fanning Friction Factor Predictor",
    page_icon="🧪",
    layout="wide",
)

# Custom CSS matching the academic web interface
st.markdown(
    """
<style>
    .stApp {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Academic Header Banner */
    .top-header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 30px;
        background-color: #FFFFFF;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 25px;
    }
    .header-title-text {
        text-align: center;
    }
    .header-title-text h1 {
        font-family: 'Georgia', serif;
        color: #0F172A;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    .header-title-text p {
        color: #475569;
        font-size: 14px;
        margin: 4px 0 0 0;
    }
    .header-sub-tags {
        text-align: right;
        font-size: 11px;
        color: #64748B;
        font-style: italic;
        line-height: 1.4;
    }

    /* Sidebar Academic Theme */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9;
        border-right: 1px solid #CBD5E1;
    }
    [data-testid="stSidebar"] * {
        color: #334155;
    }
    
    /* Form Input Container */
    .input-card {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Table Result Styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 14px;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        overflow: hidden;
    }
    .styled-table th {
        background-color: #F8FAFC;
        color: #1E293B;
        text-align: left;
        padding: 10px 15px;
        border-bottom: 1px solid #E2E8F0;
        font-weight: 600;
    }
    .styled-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #F1F5F9;
        color: #334155;
    }
    
    /* Note Card */
    .note-box {
        background-color: #F0F9FF;
        border: 1px solid #BAE6FD;
        border-left: 4px solid #0284C7;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #0369A1;
        margin-top: 20px;
    }
    
    .footer-text {
        text-align: center;
        font-size: 12px;
        color: #64748B;
        border-top: 1px solid #E2E8F0;
        padding-top: 15px;
        margin-top: 40px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# MODEL TRAINING & CACHING
# ==========================================
@st.cache_resource
def load_and_train_models():
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

    scaler_X_lam, scaler_y_lam = StandardScaler(), StandardScaler()
    X_lam_sc = scaler_X_lam.fit_transform(X_lam)
    y_lam_sc = scaler_y_lam.fit_transform(y_lam.reshape(-1, 1))

    scaler_X_tur, scaler_y_tur = StandardScaler(), StandardScaler()
    X_tur_sc = scaler_X_tur.fit_transform(X_tur)
    y_tur_sc = scaler_y_tur.fit_transform(y_tur.reshape(-1, 1))

    scaler_X_haal, scaler_y_haal = StandardScaler(), StandardScaler()
    X_haal_sc = scaler_X_haal.fit_transform(X_haal)
    y_haal_sc = scaler_y_haal.fit_transform(y_haal.reshape(-1, 1))

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

    def calc_metrics(model, X_sc, y_true, s_y):
        p_sc = model.predict(X_sc, verbose=0)
        y_pred = s_y.inverse_transform(p_sc).ravel()
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, r2

    metrics = {
        "laminar": calc_metrics(m_lam, X_lam_sc, y_lam, scaler_y_lam),
        "turbulent": calc_metrics(m_tur, X_tur_sc, y_tur, scaler_y_tur),
        "rough": calc_metrics(m_haal, X_haal_sc, y_haal, scaler_y_haal),
    }

    models = {"laminar": m_lam, "turbulent": m_tur, "rough": m_haal}
    scalers = {
        "lam_X": scaler_X_lam,
        "lam_y": scaler_y_lam,
        "tur_X": scaler_X_tur,
        "tur_y": scaler_y_tur,
        "haal_X": scaler_X_haal,
        "haal_y": scaler_y_haal,
    }

    return models, scalers, metrics


models, scalers, metrics = load_and_train_models()

# ==========================================
# INITIALIZE SESSION STATE FOR RESET / PREDICT
# ==========================================
if "status" not in st.session_state:
    st.session_state.status = "Rough"
if "re_val" not in st.session_state:
    st.session_state.re_val = 50000.0
if "kd_val" not in st.session_state:
    st.session_state.kd_val = 0.00100
if "predicted" not in st.session_state:
    st.session_state.predicted = False


def reset_inputs():
    st.session_state.status = "Rough"
    st.session_state.re_val = 50000.0
    st.session_state.kd_val = 0.00100
    st.session_state.predicted = False


# ==========================================
# TOP ACADEMIC BANNER
# ==========================================
st.markdown(
    """
<div class="top-header-container">
    <div>
        <strong style="color: #065F46; font-size: 16px;">Aligarh Muslim University</strong><br>
        <span style="font-size: 12px; color: #475569;">Department of Chemical Engineering<br>Aligarh, India</span>
    </div>
    <div class="header-title-text">
        <h1>ANN Fanning Friction Factor Predictor</h1>
        <p>Prediction of Fluid Friction Factor in Pipe Flow Using Artificial Neural Networks</p>
    </div>
    <div class="header-sub-tags">
        Process Modeling<br>
        Machine Learning<br>
        Transport Phenomena
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR METADATA & NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("### Navigation")
    nav_option = st.radio(
        "Go to",
        options=[
            "Prediction",
            "About",
            "Theory & Methodology",
            "Model Details",
            "References",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 12px; line-height: 1.6; color: #334155;">
            <p><strong>Designed by</strong><br><span style="font-weight: 600; color: #0F172A;">Neyaz Reza</span><br>(M.Tech Student)</p>
            <p><strong>Under the Supervision of</strong><br><span style="font-weight: 600; color: #0F172A;">Mr. Mohammad Abdul Hakeem</span><br>(Associate Professor)</p>
            <p><strong>Department of Chemical Engineering</strong><br>Aligarh Muslim University</p>
        </div>
        <br>
        <p style="font-style: italic; font-size: 12px; color: #64748B; text-align: center;">"From Data to Engineering Solutions"</p>
        """,
        unsafe_allow_html=True,
    )

# ==========================================
# PAGE 1: PREDICTION INTERFACE
# ==========================================
if nav_option == "Prediction":

    st.markdown(
        """
    <div class="input-card">
        <h3 style="margin-top:0; color:#0F172A; font-family:serif;">Input Parameters</h3>
        <p style="font-size:13px; color:#475569; margin-bottom:15px;">Provide the flow conditions to predict the Fanning friction factor.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Wrap inputs inside a form so press-Enter does not trigger computation
    with st.form(key="prediction_form"):
        p1, p2, p3, p4 = st.columns(4)

        with p1:
            status = st.selectbox(
                "Pipe Regime",
                options=["Rough", "Smooth"],
                key="status",
            )

        if status == "Smooth":
            with p2:
                flow_type = st.selectbox(
                    "Flow Regime",
                    options=["Laminar", "Turbulent"],
                    key="flow_type",
                )

            if flow_type == "Laminar":
                with p3:
                    re = st.number_input(
                        "Reynolds Number",
                        min_value=0.1,
                        max_value=2100.0,
                        key="re_val",
                    )
                with p4:
                    kD = st.number_input(
                        "Relative Roughness (k/D)",
                        value=0.00000,
                        format="%.5f",
                        disabled=True,
                    )
            else:
                with p3:
                    re = st.number_input(
                        "Reynolds Number",
                        min_value=2100.0,
                        max_value=100000.0,
                        key="re_val",
                    )
                with p4:
                    kD = st.number_input(
                        "Relative Roughness (k/D)",
                        value=0.00000,
                        format="%.5f",
                        disabled=True,
                    )
        else:
            with p2:
                st.text_input(
                    "Flow Regime", value="Turbulent (Rough)", disabled=True
                )

            with p3:
                re = st.number_input(
                    "Reynolds Number",
                    min_value=1000.0,
                    max_value=100000000.0,
                    key="re_val",
                )

            with p4:
                kD = st.number_input(
                    "Relative Roughness (k/D)",
                    min_value=0.0,
                    max_value=0.05,
                    format="%.5f",
                    key="kd_val",
                )

        btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 3])
        with btn_col1:
            predict_submitted = st.form_submit_button(
                "Predict", type="primary", use_container_width=True
            )

    # Reset button placed outside the form
    with btn_col2:
        st.button("Reset", on_click=reset_inputs, use_container_width=True)

    if predict_submitted:
        st.session_state.predicted = True

    st.markdown("<br>", unsafe_allow_html=True)

    # RESULTS DISPLAY SECTION (Shows ONLY after hitting "Predict")
    if st.session_state.predicted:
        if status == "Smooth":
            if flow_type == "Laminar":
                actual_f = 16.0 / re
                re_sc = scalers["lam_X"].transform([[re]])
                p_sc = models["laminar"].predict(re_sc, verbose=0)
                predicted_f = scalers["lam_y"].inverse_transform(p_sc)[0][0]
                cur_metrics = metrics["laminar"]
                model_name = "Laminar Model"
            else:
                actual_f = 0.0791 / (re**0.25)
                re_sc = scalers["tur_X"].transform([[re]])
                p_sc = models["turbulent"].predict(re_sc, verbose=0)
                predicted_f = scalers["tur_y"].inverse_transform(p_sc)[0][0]
                cur_metrics = metrics["turbulent"]
                model_name = "Turbulent Model"
        else:
            term = (6.9 / re) if kD == 0 else ((kD / 3.7) ** 1.11) + (6.9 / re)
            f_darcy = (1.0 / (-1.8 * np.log10(term))) ** 2
            actual_f = f_darcy / 16.0

            inp_sc = scalers["haal_X"].transform([[np.log10(re), kD]])
            p_sc = models["rough"].predict(inp_sc, verbose=0)
            predicted_f = scalers["haal_y"].inverse_transform(p_sc)[0][0]
            cur_metrics = metrics["rough"]
            model_name = "Rough Model"

        abs_err = abs(predicted_f - actual_f)
        rel_err = (abs_err / actual_f) * 100

        col_res, col_perf = st.columns(2)

        with col_res:
            st.markdown(
                """
            <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 12px 16px; margin-bottom: 10px;">
                <h4 style="margin: 0; color: #166534; font-family: serif;">Prediction Results</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <table class="styled-table">
                <tr><th>Quantity</th><th>Value</th></tr>
                <tr><td>ANN Predicted Fanning Friction Factor (f)</td><td><strong>{predicted_f:.6f}</strong></td></tr>
                <tr><td>Actual / Analytical Value</td><td><strong>{actual_f:.6f}</strong></td></tr>
                <tr><td>Relative Error</td><td><strong>{rel_err:.2f} %</strong></td></tr>
            </table>
            """,
                unsafe_allow_html=True,
            )

        with col_perf:
            st.markdown(
                f"""
            <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; padding: 12px 16px; margin-bottom: 10px;">
                <h4 style="margin: 0; color: #1E40AF; font-family: serif;">Model Performance ({model_name})</h4>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <table class="styled-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Mean Absolute Error (MAE)</td><td><strong>{cur_metrics[0]:.6e}</strong></td></tr>
                <tr><td>Mean Squared Error (MSE)</td><td><strong>{cur_metrics[1]:.6e}</strong></td></tr>
                <tr><td>Coefficient of Determination (R²)</td><td><strong>{cur_metrics[2]:.4f}</strong></td></tr>
            </table>
            """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
        <div class="note-box">
            <strong>Note:</strong> This tool uses trained Artificial Neural Networks (ANN) to predict the Fanning friction factor for laminar, turbulent, and rough pipe flow regimes. Ensure input parameters are within the range of the training data for best results.
        </div>
        """,
            unsafe_allow_html=True,
        )

# ==========================================
# PAGE 2: ABOUT
# ==========================================
elif nav_option == "About":
    st.subheader("1. About the Fanning Friction Factor Predictor")
    st.write(
        "The **Fanning friction factor** ($f_F$) is a dimensionless parameter used to characterize the resistance offered by a pipe wall to fluid flow. It is an important parameter in the analysis and design of piping systems because it is directly related to wall shear stress and pressure loss during fluid flow."
    )

    st.markdown(
        "For fully developed flow through a circular pipe, the Fanning friction factor is defined as:"
    )
    st.latex(r"f_F = \frac{\tau_w}{\frac{1}{2}\rho V^2}")

    st.markdown(
        """
    * **$f_F$**: Fanning friction factor
    * **$\tau_w$**: Wall shear stress
    * **$\rho$**: Fluid density
    * **$V$**: Average fluid velocity
    """
    )

    st.write(
        "The friction factor depends primarily on the Reynolds number ($Re$) and, for turbulent flow in rough pipes, the relative roughness ($k/D$ or $\epsilon/D$)."
    )

    st.markdown("#### Flow Regimes")
    st.write("The predictor considers different pipe-flow regimes:")
    st.markdown(
        """
    * **Laminar flow:** Friction factor is primarily dependent on Reynolds number.
    * **Transitional flow:** Flow behavior changes between laminar and turbulent conditions.
    * **Turbulent flow in smooth pipes:** Friction factor depends mainly on Reynolds number.
    * **Turbulent flow in rough pipes:** Friction factor depends on both Reynolds number and relative roughness.
    """
    )

    st.write(
        "The Fanning friction factor is widely used in chemical engineering and transport-phenomena calculations. Care must be taken because the Fanning friction factor and Darcy friction factor differ by a factor of four ($f_D = 4f_F$)."
    )

    st.markdown("#### Purpose of This Application")
    st.write(
        "This application uses **Artificial Neural Networks (ANNs)** to predict the Fanning friction factor from relevant flow parameters. The predicted value can be compared with an analytical or correlation-based reference value, allowing the user to evaluate prediction accuracy."
    )

    st.warning(
        "**Important:** The ANN prediction should be used within the range of Reynolds number and relative roughness represented in the training dataset. Extrapolation beyond the training range may reduce prediction reliability."
    )

# ==========================================
# PAGE 3: THEORY & METHODOLOGY
# ==========================================
elif nav_option == "Theory & Methodology":
    st.subheader("2. Theory & Methodology")

    st.markdown("#### 2.1 Reynolds Number")
    st.write("The Reynolds number is defined as:")
    st.latex(r"Re = \frac{\rho V D}{\mu}")
    st.markdown(
        "where $\rho$ is fluid density, $V$ is average velocity, $D$ is pipe diameter, and $\mu$ is dynamic viscosity. It characterizes the flow regime."
    )

    st.markdown("#### 2.2 Fanning Friction Factor")
    st.write("For a circular pipe:")
    st.latex(r"f_F = \frac{\tau_w}{\frac{1}{2}\rho V^2}")
    st.write("For laminar flow, the theoretical relationship is:")
    st.latex(r"f_F = \frac{16}{Re}")
    st.write(
        "Standard fluid-mechanics references often use the **Darcy friction factor** ($f_D$), for which the corresponding laminar relationship is:"
    )
    st.latex(r"f_D = \frac{64}{Re}")
    st.write("The two friction factors are related by:")
    st.latex(r"f_D = 4 f_F")
    st.info(
        "This distinction is prominently highlighted because confusing Darcy and Fanning friction factors introduces a factor-of-four error."
    )

    st.markdown("#### 2.3 Pressure Drop Equation")
    st.write(
        "For the Fanning friction factor, the frictional pressure drop in a straight circular pipe is expressed as:"
    )
    st.latex(r"\Delta P_f = 4 f_F \frac{L}{D} \frac{\rho V^2}{2}")
    st.write("The equivalent Darcy-Weisbach form is:")
    st.latex(r"\Delta P_f = f_D \frac{L}{D} \frac{\rho V^2}{2} \quad \text{with} \quad f_D = 4f_F")

    st.markdown("---")
    st.markdown("#### 2.4 Moody Diagram and Friction Factor")
    st.write(
        "The **Moody diagram** is a graphical representation used to determine the Darcy friction factor as a function of Reynolds number and relative roughness ($k/D$)."
    )

    col_m1, col_m2 = st.columns([1.2, 1])
    with col_m1:
        st.markdown(
            """
        The diagram contains:
        * **Reynolds number** on the horizontal axis ($Re$).
        * **Darcy friction factor** on the vertical axis ($f_D$).
        * **Curves** corresponding to different relative roughness values ($k/D$).
        * **Laminar, transitional, and turbulent flow regions**.
        
        For an application predicting the Fanning friction factor, Darcy values obtained from a conventional Moody diagram must be converted using:
        """
        )
        st.latex(r"f_F = \frac{f_D}{4}")

    with col_m2:
        st.markdown(
            """
        <div style="background-color:#FFFFFF; border:1px solid #CBD5E1; padding:15px; border-radius:6px; text-align:center;">
            <svg width="280" height="180" viewBox="0 0 300 200">
                <line x1="40" y1="170" x2="280" y2="170" stroke="#334155" stroke-width="2"/>
                <line x1="40" y1="20" x2="40" y2="170" stroke="#334155" stroke-width="2"/>
                <path d="M 40 30 L 100 130" stroke="#DC2626" stroke-width="2" fill="none"/>
                <path d="M 120 120 Q 200 110 280 110" stroke="#2563EB" stroke-width="2" stroke-dasharray="4" fill="none"/>
                <path d="M 120 100 Q 200 80 280 70" stroke="#2563EB" stroke-width="2" fill="none"/>
                <path d="M 120 80 Q 200 50 280 40" stroke="#2563EB" stroke-width="2" fill="none"/>
                <text x="160" y="190" font-size="10" fill="#475569">Reynolds Number (Re)</text>
                <text x="10" y="100" font-size="10" fill="#475569" transform="rotate(-90 15,100)">Darcy Friction Factor (f_D)</text>
            </svg>
            <p style="font-size:11px; color:#64748B; margin-top:5px;">Figure 1. Schematic Moody diagram illustrating flow regimes.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ==========================================
# PAGE 4: MODEL DETAILS & CORRELATIONS
# ==========================================
elif nav_option == "Model Details":
    st.subheader("3. Analytical References & Model Architecture")

    st.markdown("#### Analytical Correlations")
    st.write(
        "ANN predictions are evaluated against established analytical reference correlations:"
    )

    st.markdown("**1. Blasius Formula (Smooth Turbulent Flow):**")
    st.latex(r"f_F = 0.0791 Re^{-0.25}")

    st.markdown("**2. Colebrook & Haaland Equations (Rough Pipe Flow):**")
    st.write(
        "For rough turbulent pipe flow, the implicit Colebrook equation defines the Darcy friction factor ($f_D$):"
    )
    st.latex(
        r"\frac{1}{\sqrt{f_D}} = -2.0 \log_{10} \left( \frac{k/D}{3.7} + \frac{2.51}{Re \sqrt{f_D}} \right)"
    )
    st.write(
        "The explicit Haaland approximation used for ground truth calculation is:"
    )
    st.latex(
        r"f_D = \left[ -1.8 \log_{10} \left( \left(\frac{k/D}{3.7}\right)^{1.11} + \frac{6.9}{Re} \right) \right]^{-2}"
    )
    st.latex(r"f_F = \frac{f_D}{4}")

    st.markdown("---")
    st.markdown("#### Artificial Neural Network Architecture")
    st.markdown(
        """
    * **Laminar & Smooth Models**: Dense deep architecture (Inputs: $Re$) $\rightarrow$ [64, 64, 32] Hidden ReLU layers $\rightarrow$ Single linear output ($f_F$).
    * **Rough Pipe Model**: Multi-input architecture (Inputs: $\log_{10}(Re)$, $k/D$) $\rightarrow$ [128, 128, 64, 32] Swish activated layers $\rightarrow$ Single linear output ($f_F$).
    * **Preprocessing**: `StandardScaler` feature normalization applied to all inputs and targets.
    """
    )

# ==========================================
# PAGE 5: REFERENCES
# ==========================================
elif nav_option == "References":
    st.subheader("4. Academic References")
    st.markdown(
        """
    1. **Bird, R. B., Stewart, W. E., & Lightfoot, E. N. (2002).** *Transport Phenomena* (2nd ed.). John Wiley & Sons, New York. ISBN: 0-471-41077-2.
    """
    )

# ==========================================
# FOOTER
# ==========================================
st.markdown(
    """
<div class="footer-text">
    © 2026 | ANN Fanning Friction Factor Predictor &nbsp;&nbsp;|&nbsp;&nbsp; Department of Chemical Engineering, Aligarh Muslim University
</div>
""",
    unsafe_allow_html=True,
)
