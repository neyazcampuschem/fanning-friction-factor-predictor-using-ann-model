# ==========================================
# PAGE 1: PREDICTION INTERFACE
# ==========================================
if nav_option == "Prediction":

    st.markdown(
        """
    <div class="input-card">
        <h3 style="margin-top:0; color:#0F169A; font-family:serif;">Input Parameters</h3>
        <p style="font-size:13px; color:#475579; margin-bottom:15px;">Provide the flow conditions to predict the Fanning friction factor.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        status = st.selectbox(
            "Pipe Regime",
            options=["Rough", "Smooth"],
            key="status",
            on_change=clear_prediction,
        )
        st.caption("📍 **Range:** Rough / Smooth")

    if status == "Smooth":
        with p2:
            flow_type = st.selectbox(
                "Flow Regime",
                options=["Laminar", "Turbulent"],
                key="flow_type",
                on_change=clear_prediction,
            )
            st.caption("📍 **Range:** Laminar / Turbulent")

        if flow_type == "Laminar":
            with p3:
                re = st.number_input(
                    "Reynolds Number",
                    min_value=0.1,
                    max_value=2100.0,
                    key="re_val",
                    on_change=clear_prediction,
                )
                st.caption("📍 **Range:** 0.1 to 2,100")

            with p4:
                kD = st.number_input(
                    "Relative Roughness (k/D)",
                    value=0.00000,
                    format="%.5f",
                    disabled=True,
                )
                st.caption("📍 **Range:** 0.00000 (Smooth)")

        else:
            with p3:
                re = st.number_input(
                    "Reynolds Number",
                    min_value=2100.0,
                    max_value=100000.0,
                    key="re_val",
                    on_change=clear_prediction,
                )
                st.caption("📍 **Range:** 2,100 to 100,000")

            with p4:
                kD = st.number_input(
                    "Relative Roughness (k/D)",
                    value=0.00000,
                    format="%.5f",
                    disabled=True,
                )
                st.caption("📍 **Range:** 0.00000 (Smooth)")

    else:
        with p2:
            st.text_input(
                "Flow Regime", value="Turbulent (Rough)", disabled=True
            )
            st.caption("📍 **Range:** Turbulent (Fixed)")

        with p3:
            re = st.number_input(
                "Reynolds Number",
                min_value=1000.0,
                max_value=100000000.0,
                key="re_val",
                on_change=clear_prediction,
            )
            st.caption("📍 **Range:** 1,000 to 100,000,000")

        with p4:
            kD = st.number_input(
                "Relative Roughness (k/D)",
                min_value=0.0,
                max_value=0.05,
                format="%.5f",
                key="kd_val",
                on_change=clear_prediction,
            )
            st.caption("📍 **Range:** 0.00000 to 0.05000")

    # Action Buttons
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 3])
    with btn_col1:
        if st.button(
            "Predict", type="primary", use_container_width=True
        ):
            st.session_state.predicted = True

    with btn_col2:
        st.button("Reset", on_click=reset_inputs, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
