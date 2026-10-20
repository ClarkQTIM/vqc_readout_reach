# Reference: the original `cnn_vqc_expressivity` code and results

Kept unchanged for comparison. Do not build on this code; `vqcx/` replaces it.

`original_results/performance_runs_reparsed.csv` re-parses all 1,080 training runs (180 configs × 3 runs, Iris and RetinaMNIST) from the run folders on the server, fixing the folder-name parsing bug below.

## Known issues in the original code

1. `sweep_results_4_qubits/` (used by the final SHAP analysis) was generated with random real input states, not |0000⟩. KL (0.106–0.127) and entropy (0.83–1.02) barely vary across circuits; random real states alone reproduce them (KL 0.119, entropy 0.86 nats).
2. `generate_roc_auc_results.py` splits folder names on `_`, so `ising_xx_ring` parses as entangler `ising`, topology `xx`. Ising and "none" rows then drop out of every merge (96 of 180 configs survive).
3. In the KL and gradient sweeps, `entanglement_op=ent_op if ent_op else "cnot"` turns "no entanglement" into CNOT all-to-all. The entanglement sweep is unaffected.
4. Parameters are drawn from U[0, π], not U[0, 2π] (Sim et al.), which creates the single-qubit "depth helps" effect.
5. Only qubit 0 is measured. Which input features can affect that readout, and whether only even functions of them can, drives much of the Iris AUC variation (see `vqcx/reach.py`).
6. Entropies are in nats (PennyLane `vn_entropy` default). Gradient "norms" are L2 norms over all parameters, so they grow with parameter count.
7. The retina results cannot come from the uploaded `main.py` (`encoder` is undefined in that branch); they came from an earlier version.
8. `np.trapz` no longer exists in NumPy 2.x.
