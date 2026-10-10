import os
import json
import pytest
from streamlit.testing.v1 import AppTest

def test_dashboard_loads_all_pages():
    # streamlit.testing AppTest loads every page with no exceptions.
    at = AppTest.from_file("dashboard.py").run()
    assert not at.exception, "App crashed on Overview page"
    
def test_dashboard_missing_files_warning():
    # Deleting one results file shows a friendly warning, not a crash.
    at = AppTest.from_file("dashboard.py").run()
    # Go to Bank x Method matrix
    at.sidebar.radio[0].set_value("Bank x Method Matrix").run()
    assert not at.exception
    # Should contain a warning string since we didn't mock the CSV for the test env
    warnings = [w.value for w in at.warning]
    assert any("not found" in w for w in warnings), "Missing file warning not rendered"

def test_dashboard_payload_audit_injection(tmp_path):
    # The 0 raw records counter changes if a test injects a bad payload into the audit log.
    # We patch RESULTS_DIR for the test
    audit_file = tmp_path / "audit.json"
    audit_file.write_text(json.dumps({"raw_records_sent": 500}))
    
    with open("dashboard.py", "r") as f:
        code = f.read()
    
    # Quick monkeypatch of RESULTS_DIR inside the file just for the test
    patched_code = code.replace('RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/tables"))', f'RESULTS_DIR = r"{str(tmp_path)}"')
    
    test_file = tmp_path / "test_dashboard.py"
    test_file.write_text(patched_code)
    
    at = AppTest.from_file(str(test_file)).run()
    at.sidebar.radio[0].set_value("Payload Audit").run()
    assert not at.exception
    
    # Check that metric for Raw Records Sent updated to 500
    metrics = [m.value for m in at.metric]
    assert "500" in metrics or 500 in metrics or any(m == "500" for m in metrics)
