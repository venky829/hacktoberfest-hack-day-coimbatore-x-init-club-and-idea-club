import streamlit as st
import requests

st.set_page_config(
    page_title="NutriLens - Food Integrity & Label Auditor",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 NutriLens — Autonomous Food Integrity Agent")
st.caption("Powered by Gemma Multimodal Agent Skill (Open Standard)")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Ingest Food Label")
    
    cloud_toggle = st.toggle("Force Cloud Inference", value=False)
    source_choice = st.radio("Input Source", ["Upload Image", "Use Webcam"], horizontal=True)

    img_file = None
    if source_choice == "Upload Image":
        img_file = st.file_uploader("Upload packaging/nutrition label photo", type=["jpg", "png", "jpeg"])
    else:
        img_file = st.camera_input("Capture live label")

    if img_file:
        try:
            st.image(img_file, caption="Target Input Image", width="stretch")
        except TypeError:
            st.image(img_file, caption="Target Input Image", use_container_width=True)

    button_disabled = img_file is None
    try:
        run_btn = st.button("Run Multimodal Audit", type="primary", width="stretch", disabled=button_disabled)
    except TypeError:
        run_btn = st.button("Run Multimodal Audit", type="primary", use_container_width=True, disabled=button_disabled)

    if run_btn and img_file:
        with st.spinner("Analyzing visual tokens and running deterministic skill audit..."):
            try:
                files = {"file": (img_file.name, img_file.getvalue(), "image/jpeg")}
                data = {"force_cloud": str(cloud_toggle).lower()}
                resp = requests.post("http://127.0.0.1:8000/api/audit", files=files, data=data, timeout=150)
                
                if resp.status_code == 200:
                    st.session_state["report"] = resp.json()
                    st.success("Analysis complete!")
                else:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend server: {e}")

with col2:
    st.subheader("2. Compliance & Verification Report")
    
    if "report" in st.session_state:
        rep = st.session_state["report"]
        execution_source = rep.get("execution_source", "Unknown")
        audit = rep.get("audit", {})
        ext = rep.get("extracted", {})

        # Execution Engine Indicator
        if "Edge" in execution_source or "RTX" in execution_source:
            st.success(f"⚡ Execution Engine: **{execution_source}**")
        else:
            st.info(f"☁️ Execution Engine: **{execution_source}**")

        # Claim Integrity Score
        score = audit.get("integrity_score", 0)
        st.metric(
            label="Claim Integrity Score",
            value=f"{score} / 100",
            delta=f"{score - 100} pts" if score < 100 else "Perfect"
        )

        st.divider()

        # Deceptive Ingredients Section (Handles both dicts and strings)
        deceptive = audit.get("deceptive_ingredients", [])
        if deceptive:
            flagged_items = []
            for item in deceptive:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("ingredient") or str(item)
                    gi = item.get("gi") or item.get("glycemic_index")
                    flagged_items.append(f"{name} (GI: {gi})" if gi else name)
                else:
                    flagged_items.append(str(item))

            st.error(f"⚠️ **Deceptive High-GI Fillers Detected:** {', '.join(flagged_items)}")
            
            # If detailed dictionary data is present, display it in a tidy table
            if any(isinstance(item, dict) for item in deceptive):
                with st.expander("Inspection Details on Flagged Additives", expanded=True):
                    st.json(deceptive)
        else:
            st.success("✅ **Clean Formulation:** No hidden glycemic spikes or maltodextrin detected.")

        st.divider()

        # Caloric Math Discrepancy (Atwater Factor)
        st.markdown("#### Caloric Verification (Atwater Factor)")
        c_a, c_b, c_c = st.columns(3)
        declared = audit.get("declared_calories", 0.0)
        calculated = audit.get("calculated_calories", 0.0)
        variance = audit.get("variance_percentage", 0.0)

        c_a.metric("Declared Calories", f"{declared} kcal")
        c_b.metric("Calculated Calories", f"{calculated} kcal")
        c_c.metric("Caloric Variance", f"{variance}%")

        # Local RTX 3050 Offline Analysis feedback if available
        if "local_analysis" in ext and ext["local_analysis"]:
            st.markdown("#### Edge Reasoning (RTX 3050 Gemma 2B)")
            st.info(ext["local_analysis"])

        # Raw Extracted Label Metadata
        with st.expander("View Extracted Label Metadata", expanded=False):
            st.json(ext)
    else:
        st.info("Upload or capture a photo of a food label on the left and click **Run Multimodal Audit** to view the live report.")
