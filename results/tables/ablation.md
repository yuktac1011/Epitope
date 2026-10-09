# Ablation Study

|    | Method   | Metric   |   Mean Diff (Hybrid - MLP) |   95% CI Lower |   95% CI Upper |
|---:|:---------|:---------|---------------------------:|---------------:|---------------:|
|  0 | local    | Macro AP |                   0.309888 |       0.301689 |       0.320407 |
|  1 | fed2     | Macro AP |                   0.272446 |       0.247969 |       0.308136 |
|  2 | fed3     | Macro AP |                   0.297837 |       0.269078 |       0.334142 |
|  3 | fed5     | Macro AP |                   0.300033 |       0.271117 |       0.333112 |

**Conclusion:** Hybrid better

## Parameter Overhead
- MLP Base Parameters: ~4,225
- Hybrid Extra Parameters (Encoder): ~10,416
- Extra Bytes Sent Per Round per client (if encoder shared): 41,664 bytes (40 KB)
