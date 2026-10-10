import os
import sys
import time
import json
import numpy as np

# Ensure root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from backend.app.kyc.data import load_kyc_data
from backend.app.kyc.inference import KYCDetector

def run_evaluation_benchmark(num_samples: int = 100, save_report: bool = True):
    """
    Executes Section 3.7 Phase 1 evaluation harness.
    Runs validation images through full pipeline and reports per-condition metrics.
    """
    print(f"Loading {num_samples} validation samples for benchmark...")
    dataset = load_kyc_data(split="validation", max_samples=num_samples)
    detector = KYCDetector()

    results = []
    latencies = []

    print("Running screening pipeline on evaluation set...")
    for idx, item in enumerate(dataset):
        img = item["image"]
        ground_truth = item["label"] # 0 = Fake, 1 = Real

        t0 = time.time()
        res = detector.assess_photo(img)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        results.append({
            "index": idx,
            "ground_truth": "REAL" if ground_truth == 1 else "FAKE",
            "decision": res["decision"],
            "route": res["route"],
            "risk": res["risk"],
            "quality_usable": res["quality"].get("usable", False),
            "latency_ms": lat
        })

    # Metric calculations
    total = len(results)
    decisions = {}
    for r in results:
        dec = r["decision"]
        decisions[dec] = decisions.get(dec, 0) + 1

    # Breakdown by class
    fakes = [r for r in results if r["ground_truth"] == "FAKE"]
    reals = [r for r in results if r["ground_truth"] == "REAL"]

    # Fraud catch rate: fakes routed to UNCERTAIN or HIGH_RISK
    fakes_caught = sum(1 for r in fakes if r["decision"] in ("UNCERTAIN", "HIGH_RISK"))
    fakes_auto_passed = sum(1 for r in fakes if r["decision"] == "LOW_RISK")

    # Real auto-pass rate: reals routed to LOW_RISK
    reals_auto_passed = sum(1 for r in reals if r["decision"] == "LOW_RISK")
    reals_stepped_up = sum(1 for r in reals if r["decision"] in ("UNCERTAIN", "HIGH_RISK"))

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": detector.version,
        "total_evaluated": total,
        "latency_stats": {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2)
        },
        "decisions_distribution": decisions,
        "security_metrics": {
            "total_fakes": len(fakes),
            "fakes_caught_rate": round(fakes_caught / len(fakes), 4) if fakes else 0.0,
            "fakes_auto_passed_leakage": round(fakes_auto_passed / len(fakes), 4) if fakes else 0.0,
            "total_reals": len(reals),
            "reals_auto_pass_rate": round(reals_auto_passed / len(reals), 4) if reals else 0.0,
            "reals_step_up_friction": round(reals_stepped_up / len(reals), 4) if reals else 0.0
        }
    }

    print("\n================ EVALUATION BENCHMARK REPORT ================")
    print(json.dumps(report, indent=4))

    if save_report:
        report_dir = "backend/app/kyc/reports"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "kyc_eval_baseline.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"\nReport successfully saved to: {report_path}")

    return report

if __name__ == "__main__":
    run_evaluation_benchmark(num_samples=50)
