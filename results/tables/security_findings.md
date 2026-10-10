# Security Testing Findings

## Membership Inference Attacks
Loss-threshold attacks against standard FedAvg models yield higher AUCs. Applying DP-SGD strictly bounds this capability. Refer to `mia.csv`.

## Poisoning & Robust Aggregation
When subject to malicious Label Flipping and Scaled Model Replacements:
- **CoordinateMedian**: Successfully recovered the score (Macro AP: 0.550).
- **TrimmedMean**: Failed to fully recover the score against 2 attackers (Macro AP: 0.400).
- **Krum**: Successfully recovered the score (Macro AP: 0.590).
