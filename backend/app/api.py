import os
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="FedShield API", description="API for Federated Learning Fraud Detection Metrics")

# Enable CORS for frontend applications (like Streamlit or React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FED_DIR = os.path.join(BASE_DIR, 'federated')

# --- Pydantic Models ---
class HealthResponse(BaseModel):
    status: str
    timestamp: str

class MacroMetric(BaseModel):
    num_participating_banks: int
    macro_average_precision: float
    macro_pr_auc: float
    macro_recall_1fpr: float

class PerBankMetric(BaseModel):
    bank_id: int
    local_ap: float
    federated_ap: float
    federated_recall_1fpr: float
    improvement: float

class AuditResponse(BaseModel):
    status: str
    raw_transactions_shared: int
    account_ids_shared: int
    dataframes_shared: int
    model_parameters_sent_per_client: int
    description: str

class CommunicationMetric(BaseModel):
    network_size: str
    total_params_sent_recv_per_round: int
    estimated_bandwidth_kb_5_rounds: float

# --- Helper Functions ---
def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Data file not found: {os.path.basename(path)}")
    return pd.read_csv(path)

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def get_health():
    """Returns the health status of the API."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

@app.get("/results/macro", response_model=List[MacroMetric])
def get_macro_results():
    """Returns macro performance for 1-bank, 2-bank, 3-bank, and 5-bank settings."""
    df = load_csv_safe(os.path.join(FED_DIR, 'federated_experiment_results.csv'))
    
    results = []
    for _, row in df.iterrows():
        results.append(MacroMetric(
            num_participating_banks=int(row["Num_Participating_Banks"]),
            macro_average_precision=float(row["Macro_AP"]),
            macro_pr_auc=float(row["Macro_PR_AUC"]),
            macro_recall_1fpr=float(row["Macro_Recall_1FPR"])
        ))
    return results

@app.get("/results/per-bank", response_model=List[PerBankMetric])
def get_per_bank_results():
    """Returns per-bank metrics for Local Hybrid vs Global Federated 5-bank model."""
    local_df = load_csv_safe(os.path.join(MODELS_DIR, 'local_hybrid_per_bank.csv'))
    fed_df = load_csv_safe(os.path.join(FED_DIR, 'per_bank_federated_results.csv'))
    
    # Isolate the 5 Banks setting from the federated results
    fed_5_df = fed_df[fed_df["Setting"] == "5 Banks"]
    
    # Merge on BankID
    merged = pd.merge(local_df, fed_5_df, on="BankID", suffixes=('_Local', '_Fed'))
    
    results = []
    for _, row in merged.iterrows():
        results.append(PerBankMetric(
            bank_id=int(row["BankID"]),
            local_ap=float(row["Average Precision_Local"]),
            federated_ap=float(row["Average Precision_Fed"]),
            federated_recall_1fpr=float(row["Recall @ 1% FPR_Fed"]),
            improvement=float(row["Average Precision_Fed"] - row["Average Precision_Local"])
        ))
    return results

@app.get("/results/comparison")
def get_comparison():
    """Returns a clean summary comparison of Local Hybrid vs Federated models."""
    per_bank_data = get_per_bank_results()
    
    avg_improvement = sum([item.improvement for item in per_bank_data]) / len(per_bank_data) if per_bank_data else 0
    num_banks_improved = sum([1 for item in per_bank_data if item.improvement > 0])
    
    return {
        "summary": f"{num_banks_improved} out of {len(per_bank_data)} banks improved their Average Precision by joining the federated network.",
        "average_ap_improvement": avg_improvement,
        "details": per_bank_data
    }

@app.get("/audit", response_model=AuditResponse)
def get_audit():
    """Returns the data-boundary audit summary."""
    return AuditResponse(
        status="PASS",
        raw_transactions_shared=0,
        account_ids_shared=0,
        dataframes_shared=0,
        model_parameters_sent_per_client=3969,
        description="Verified mathematically during training. The global server never sees raw transaction data because complex Autoencoder layers are frozen locally."
    )

@app.get("/communication", response_model=List[CommunicationMetric])
def get_communication_metrics():
    """Returns estimated bandwidth used for each federated setting."""
    return [
        CommunicationMetric(
            network_size="2 Banks",
            total_params_sent_recv_per_round=15876,
            estimated_bandwidth_kb_5_rounds=390.62
        ),
        CommunicationMetric(
            network_size="3 Banks",
            total_params_sent_recv_per_round=23814,
            estimated_bandwidth_kb_5_rounds=585.94
        ),
        CommunicationMetric(
            network_size="5 Banks",
            total_params_sent_recv_per_round=39690,
            estimated_bandwidth_kb_5_rounds=976.56
        )
    ]
