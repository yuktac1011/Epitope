import streamlit as st
from PIL import Image
import os
import sys

# Ensure backend modules can be resolved
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Safely purge stale KYC dependency modules from Streamlit's process memory
for prefix in ("backend.app.kyc.decision", "backend.app.kyc.inference", "backend.app.kyc.fusion", "backend.app.kyc.signals", "backend.app.kyc.face", "backend.app.kyc.io"):
    for k in list(sys.modules.keys()):
        if k.startswith(prefix):
            sys.modules.pop(k, None)

from backend.app.kyc.inference import KYCDetector

st.set_page_config(page_title="FedShield - Real-World KYC Defense", page_icon="🛡️", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
    .badge-low { color: #155724; background-color: #d4edda; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .badge-unc { color: #856404; background-color: #fff3cd; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .badge-high { color: #721c24; background-color: #f8d7da; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .badge-retake { color: #383d41; background-color: #e2e3e5; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ FedShield KYC Screening & Deepfake Defense")
st.markdown("""
Production-grade multi-stage digital KYC pipeline implementing **Phase 1 Robust Screening**:
Safe ingestion $\\rightarrow$ YuNet Face Detection $\\rightarrow$ Quality Gate (Blur/Exposure) $\\rightarrow$ Canonical Crop $\\rightarrow$ Dual-Signal Scoring $\\rightarrow$ Calibrated Decision Routing.
""")

@st.cache_resource
def load_kyc_detector_v3():
    return KYCDetector()

detector = load_kyc_detector_v3()

col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("1. Photo Ingestion")
    uploaded_file = st.file_uploader("Upload an onboarding face photo (JPG/PNG/WebP)", type=["jpg", "jpeg", "png", "webp"])

    enable_gradcam = st.checkbox("Generate Grad-CAM Explainability Heatmap", value=True, help="Highlights facial regions influencing the synthetic risk score.")

    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            raw_pil = Image.open(uploaded_file).convert("RGB")
            st.image(raw_pil, caption="Uploaded Original Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            raw_pil = None
    else:
        raw_pil = None
        st.info("Upload a selfie, photo, or sample image to run automated KYC screening.")

if raw_pil is not None:
    with col_right:
        st.subheader("2. Multi-Stage Screening Results")
        with st.spinner("Executing KYC screening pipeline..."):
            try:
                result = detector.assess_photo(raw_pil, explain=enable_gradcam)
            except (TypeError, AttributeError):
                fresh_detector = KYCDetector()
                result = fresh_detector.assess_photo(raw_pil, explain=enable_gradcam)

        decision = result["decision"]
        route = result["route"]
        risk = result["risk"]
        quality = result["quality"]
        signals = result["signals"]
        reasons = result["reasons"]

        # Face Crop & Heatmap Preview
        if result.get("cropped_face"):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.image(result["cropped_face"], caption="Canonical Crop", use_container_width=True)
            with c2:
                if result.get("explanation_heatmap"):
                    st.image(result["explanation_heatmap"], caption="Grad-CAM Heatmap", use_container_width=True)
                else:
                    st.caption("No heatmap")
            with c3:
                st.write(f"**Faces:** `{quality.get('n_faces', 1)}`")
                st.write(f"**Latency:** `{result.get('latency_ms', 0)} ms`")
                st.write(f"**Version:** `{result.get('model_version')}`")

        st.markdown("---")

        # Decision & Action Banner
        if decision == "LOW_RISK":
            st.success(f"### Outcome: 🟢 LOW RISK ({risk:.1%})")
            st.markdown(f"**Recommended Action:** {result['action']}")
        elif decision == "UNCERTAIN":
            st.warning(f"### Outcome: 🟡 UNCERTAIN ({risk:.1%})")
            st.markdown(f"**Recommended Action:** {result['action']}")
        elif decision == "HIGH_RISK":
            st.error(f"### Outcome: 🔴 HIGH RISK ({risk:.1%})")
            st.markdown(f"**Recommended Action:** {result['action']}")
        elif decision == "RETAKE":
            st.info(f"### Outcome: ⚪ RETAKE REQUIRED")
            st.markdown(f"**Recommended Action:** {result['action']}")
        else: # CANNOT_ASSESS
            st.warning(f"### Outcome: 🟣 CANNOT ASSESS")
            st.markdown(f"**Recommended Action:** {result['action']}")

        st.markdown("---")

        # Quality Gate Metrics
        st.subheader("3. Quality Gate Inspection")
        metrics = quality.get("metrics", {})
        if quality.get("usable"):
            q1, q2, q3 = st.columns(3)
            q1.metric("Blur Variance", f"{metrics.get('blur_variance', 0):.1f}", "Pass (≥45)")
            q2.metric("Mean Luminance", f"{metrics.get('mean_luminance', 0):.1f}", "Pass (35-235)")
            q3.metric("Face Short Side", f"{metrics.get('face_short_side_px', 0)} px", "Pass (≥60px)")
        else:
            st.error(f"Quality Gate Failed: {', '.join(quality.get('retake_reasons', []))}")
            st.write("Defective metrics:", metrics)

        # Signals Breakdown (Phase 2 Multi-Modal Suite)
        st.subheader("4. Multi-Signal Breakdown")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            art_score = signals.get("artifact", {}).get("score")
            st.metric(
                "Spatial Artifact",
                f"{art_score:.1%}" if art_score is not None else "N/A",
                help="CNN spatial manipulation probability"
            )
        with s2:
            chan_score = signals.get("channel", {}).get("score")
            st.metric(
                "Capture Channel",
                f"{chan_score:.1%}" if chan_score is not None else "N/A",
                help="Screenshot / screen-replay indicator"
            )
        with s3:
            freq_score = signals.get("frequency", {}).get("score")
            st.metric(
                "FFT Frequency",
                f"{freq_score:.1%}" if freq_score is not None else "N/A",
                help="2D FFT checkerboard periodic upsampling spikes"
            )
        with s4:
            prov_score = signals.get("provenance", {}).get("score")
            st.metric(
                "Provenance Tags",
                f"{prov_score:.1%}" if prov_score is not None else "N/A",
                help="AI generator metadata or EXIF editing tags"
            )

        # Reason Codes
        if reasons:
            st.subheader("5. Active Reason Codes")
            for r in reasons:
                st.code(f"[{r.get('signal', 'system').upper()}] {r.get('code')}")

st.markdown("---")
st.caption("🛡️ FedShield Multi-Tenant Privacy-Preserving KYC Screening Architecture | Zero Raw Biometrics Persisted")
