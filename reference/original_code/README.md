# Quantum Expressivity Analysis for VQC Architectures

This project investigates how the **expressivity** of Variational Quantum Circuit (VQC) architectures relates to their **downstream performance** on medical classification tasks.

We focus on:
- Designing and evaluating **modular VQC architectures**
- Measuring **expressivity** via **fidelity distributions** of randomly initialized circuits
- Correlating those expressivity scores with **test performance** (e.g., AUROC, accuracy) after training

---

## 🏗️ Architectural Search Space

Each VQC model is defined by the following configurable parameters:

- **Single-qubit rotation gate**: `RY`, `RX`, `RZ`, `U3`
- **Entanglement gate**: `CNOT`, `CRX`, `CRY`, `CZ`, `IsingXX`, `IsingYY`, `IsingZZ`
- **Connectivity topology**: `ring` or `all-to-all`
- **Depth**: number of layers
- **Encoder**: amplitude encoding with or without learnable projection

This modular design enables a combinatorial sweep of different quantum circuit configurations.

---

## 📏 Expressivity Measurement

We evaluate the expressivity of each architecture using the **fidelity distribution method**:

1. Generate a large number (e.g., 100–500) of output states from **randomly initialized parameters**.
2. Compute pairwise fidelities:
   \[ F(|\psi_i\rangle, |\psi_j\rangle) = |\langle \psi_i | \psi_j \rangle|^2 \]
3. Analyze the **distribution** of these fidelities:
   - Narrow distribution → low expressivity
   - Broad/flat distribution → high expressivity
4. Summarize expressivity using:
   - **Standard deviation** or **entropy** of the histogram
   - **KL divergence** from Haar-random fidelity distribution

🧠 **Important**: These fidelity-based expressivity scores are computed on **randomly initialized models**, not trained ones. This isolates the *architectural capacity* from optimization artifacts.

### 📚 Key References
- Sim et al., *"Expressibility and Entangling Capability of Parameterized Quantum Circuits"*, Adv. Quantum Technol. (2019)
- Abbas et al., *"The Power of Quantum Neural Networks"*, Nat. Comput. Sci. (2021)
- Holmes et al., *"Connecting Barren Plateaus to Representational Capacity"*, PRX Quantum (2022)

---

## 🏥 Downstream Evaluation

After quantifying expressivity:

1. Train each VQC architecture on a medical classification task (e.g., retinal or cervical cancer screening)
2. Keep training protocol constant across architectures
3. Record downstream metrics (AUROC, AUPRC, etc.)

We then plot and analyze:
- **Expressivity score** vs. **Test performance**
- Identify sweet spots where architectures are expressive enough to learn well but not so expressive that they overfit or suffer from barren plateaus

---

## 📈 Future Directions
- Add noise models back in to analyze NISQ robustness
- Study generalization gap as a function of expressivity
- Explore learned state diversity (post-training) as a secondary metric

---

This README summarizes the goals, methods, and design choices for linking **quantum circuit expressivity** to **real-world task performance**.