import pytest
import numpy as np
from app.core.config import ExperimentConfig
from app.core.metrics import calculate_threshold_from_validation, evaluate_metrics
from app.federated.experiments import data_boundary_audit
from app.data.splits import load_fixed_splits

def test_config_parsing(tmp_path):
    yaml_content = """
method: fedavg
model: hybrid
bank_ids: [0, 1]
rounds: 2
seeds: [42]
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    
    config = ExperimentConfig.from_yaml(str(config_file))
    assert config.method == "fedavg"
    assert config.bank_ids == [0, 1]
    assert config.rounds == 2

def test_metrics_threshold():
    y_val = np.array([0, 0, 0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    
    thresh = calculate_threshold_from_validation(y_val, y_probs, 0.01)
    assert thresh >= 0.4
    
    metrics = evaluate_metrics(y_val, y_probs, thresh)
    assert "ap" in metrics

def test_data_boundary_audit():
    valid_weights = [np.array([1.0, 2.0]), np.array([[1.0], [2.0]])]
    total = data_boundary_audit(valid_weights)
    assert total == 4
    
    with pytest.raises(AssertionError):
        data_boundary_audit([[1.0, 2.0]])
        
    huge_weights = [np.zeros(20000)]
    with pytest.raises(AssertionError):
        data_boundary_audit(huge_weights)

def test_fixed_splits_identical():
    # Load Bank 0 splits twice to simulate two runs
    split_A = load_fixed_splits(0)
    split_B = load_fixed_splits(0)
    
    X_test_A, y_test_A = split_A['test']
    X_test_B, y_test_B = split_B['test']
    
    # Assert they are exactly the same
    assert np.array_equal(X_test_A.values, X_test_B.values)
    assert np.array_equal(y_test_A.values, y_test_B.values)
