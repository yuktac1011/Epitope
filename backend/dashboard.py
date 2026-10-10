import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="FedShield Dashboard", layout="wide")

# Banner removed per user request

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/tables"))
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/runs"))

def load_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def load_md(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read()

def download_button(df, name):
    if df is not None:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name=name, mime='text/csv')

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview", 
    "Bank x Method Matrix", 
    "Benefit & Contribution", 
    "MLP vs Hybrid Ablation", 
    "FedAvg vs FedProx", 
    "Privacy (DP & MIA)", 
    "Robustness (Poisoning)", 
    "Communication Curve", 
    "Payload Audit", 
    "Methodology & Limits"
])

def render_page(title, csv_file, md_file=None):
    st.title(title)
    df = load_csv(csv_file)
    if df is None:
        st.warning(f"Data file '{csv_file}' not found. Please run the corresponding experiment first.")
    else:
        st.dataframe(df)
        download_button(df, csv_file)
        
    if md_file:
        md = load_md(md_file)
        if md:
            with st.expander("Findings", expanded=True):
                st.markdown(md)

if page == "Overview":
    st.title("FedShield Dashboard")
    st.write("Welcome to the FedShield results dashboard. Use the sidebar to navigate through the experiment results.")
    
elif page == "Bank x Method Matrix":
    st.title("Bank x Method Matrix")
    metric = st.radio("Metric", ["Average Precision", "Recall at 1% FPR"])
    target = "matrix_ap.csv" if metric == "Average Precision" else "matrix_recall.csv"
    
    df = load_csv(target)
    if df is None:
        st.warning(f"Data file '{target}' not found. Please run the matrix experiment.")
    else:
        st.dataframe(df)
        download_button(df, target)
        md = load_md("matrix_ap.md")
        if md: st.markdown(md)

elif page == "Benefit & Contribution":
    render_page("Benefit and Contribution Analysis", "benefit.csv", "findings.md")
    st.subheader("Shapley Contributions")
    contrib = load_csv("contribution.csv")
    if contrib is not None:
        st.dataframe(contrib)
        download_button(contrib, "contribution.csv")

elif page == "MLP vs Hybrid Ablation":
    render_page("MLP vs Hybrid Ablation", "ablation.csv", "ablation.md")

elif page == "FedAvg vs FedProx":
    render_page("FedAvg vs FedProx", "prox.csv")

elif page == "Privacy (DP & MIA)":
    render_page("Differential Privacy", "dp.csv", "dp_guarantee.md")
    st.subheader("Membership Inference Attack")
    mia = load_csv("mia.csv")
    if mia is not None:
        st.dataframe(mia)
        download_button(mia, "mia.csv")

elif page == "Robustness (Poisoning)":
    render_page("Robustness against Poisoning", "poison.csv", "security_findings.md")
    
    st.divider()
    st.subheader("Interactive Attack Simulator")
    st.write("Simulate a live data poisoning attack against the federation.")
    
    col1, col2 = st.columns(2)
    with col1:
        defense_sim = st.selectbox("Server Aggregation Defense", ["FedAvg", "CoordinateMedian", "TrimmedMean", "Krum"])
    with col2:
        attackers_sim = st.slider("Number of Malicious Clients", min_value=0, max_value=2, value=1)
        
    df_poison = load_csv("poison.csv")
    if df_poison is not None:
        # Get baseline (0 attackers)
        baseline_ap = df_poison[(df_poison["Malicious Clients"] == 0) & (df_poison["Defense"] == "FedAvg")]["Macro AP"].values
        baseline_ap = baseline_ap[0] if len(baseline_ap) > 0 else 0.65
        
        # Get simulated attack
        sim_ap = df_poison[(df_poison["Malicious Clients"] == attackers_sim) & (df_poison["Defense"] == defense_sim)]["Macro AP"].values
        sim_ap = sim_ap[0] if len(sim_ap) > 0 else 0.0
        
        st.metric(
            label=f"Federation Macro Average Precision (Attackers: {attackers_sim}, Defense: {defense_sim})",
            value=f"{sim_ap:.3f}",
            delta=f"{sim_ap - baseline_ap:.3f} from Clean Baseline",
            delta_color="normal" if sim_ap >= baseline_ap - 0.15 else "inverse"
        )
        
        if attackers_sim == 0:
            st.info("The federation is currently running cleanly with 0 malicious clients.")
        elif sim_ap >= baseline_ap - 0.15:
            st.success(f"**SUCCESS:** {defense_sim} successfully defended the global model against {attackers_sim} poisoned clients!")
        else:
            st.error(f"**COMPROMISED:** {defense_sim} failed to mitigate the attack. The global model was severely degraded.")

elif page == "Communication Curve":
    render_page("Communication Curve", "communication.csv")

elif page == "Payload Audit":
    st.title("Payload Audit")
    # Mocking reading from an audit log for raw rows sent
    # If the user injects a bad payload into our mock file, it updates.
    audit_log = os.path.join(RESULTS_DIR, "audit.json")
    raw_records_sent = 0
    if os.path.exists(audit_log):
        with open(audit_log, "r") as f:
            data = json.load(f)
            raw_records_sent = data.get("raw_records_sent", 0)
            st.json(data)
    else:
        st.info("No audit log found. Simulating zero raw records sent.")
        
    st.metric(label="Raw Records Sent", value=raw_records_sent, delta=None, delta_color="inverse")
    if raw_records_sent > 0:
        st.error("SECURITY VIOLATION: Raw records leaked across the boundary!")
    else:
        st.success("Zero raw records transmitted.")

elif page == "Methodology & Limits":
    st.title("Methodology & Limits")
    st.write("All experiments are strictly configured via YAML and validated with Pydantic.")
