# Differential Privacy Guarantee for FedShield

This prototype implements **Record-Level Differential Privacy** using DP-SGD (via Opacus) at each individual client.

## What is Protected
- **Record-level privacy**: The presence or absence of any single transaction (record) within a bank's local dataset is mathematically obscured.
- **Honest-but-curious server**: The central server only receives clipped and noised gradients. It cannot reverse-engineer individual transactions from the model updates.

## What is NOT Protected (Limitations)
- **No Secure Aggregation (SecAgg)**: Model weights are transmitted in the clear over the network to the server. An active man-in-the-middle could inspect the noised weights.
- **Client-level privacy**: The DP guarantee applies to individual rows within a bank, not the entire bank's participation. The server still knows which banks participated.
