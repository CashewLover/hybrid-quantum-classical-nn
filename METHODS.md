# Methods and Protocol

## 1. Circuit and Model Definitions

Every quantum run uses exact, noiseless 64-bit statevector simulation (`float64` / `complex128`) starting from $|0\rangle^{\otimes n}$, without shot sampling. For $n$ simulated qubits ($n = 4$ on two moons, $n = 2$ on parity) and $L = 3$ layers, each layer $l \in \{1, \dots, L\}$ applies three operations in order:

1. Data re-uploading via $R_Y(s \cdot x_q) = \exp(-i \frac{s x_q}{2} Y)$ on each wire $q \in \{0, \dots, n-1\}$. Two moons uses $s = \pi$ because incoming activations come from $\tanh$ in $(-1, 1)$; parity uses $s = \pi/2$.
2. A trainable single-qubit rotation $\mathrm{Rot}(\phi_{l,q}, \theta_{l,q}, \omega_{l,q}) = R_Z(\omega_{l,q}) R_Y(\theta_{l,q}) R_Z(\phi_{l,q})$ on each wire, initialized uniformly on $[-\pi, \pi]$ ($3 n L$ trainable angles in total).
3. An entangling ring of $\mathrm{CNOT}_{q \to (q+1)\bmod n}$ gates for $q = 0, \dots, n-1$ (on $n = 2$, $\mathrm{CNOT}_{0\to 1}$ followed by $\mathrm{CNOT}_{1\to 0}$). When the entanglement switch is off, this step is replaced by the identity, so the state remains a product $\bigotimes_{q=0}^{n-1} |\psi_q(x_q)\rangle$ and qubit $q$ depends only on input $x_q$.

The circuit returns the vector of Pauli-$Z$ expectations:
$$z_q(\mathbf{x}) = \langle \psi(\mathbf{x}) | Z^{(q)} | \psi(\mathbf{x}) \rangle \in [-1, 1], \qquad q = 0, \dots, n-1.$$
We write the circuit gate by gate in PennyLane (`default.qubit`) and wrap it as a PyTorch layer (`TorchLayer`) differentiated by backpropagation through the simulator. At the end of each hybrid run, we evaluate single-qubit entanglement on a fixed 500-point test sample by averaging the base-2 von Neumann entropy of each reduced single-qubit state $\rho_q = \mathrm{Tr}_{\neq q}(|\psi\rangle\langle\psi|)$:
$$\bar{S} = \frac{1}{n} \sum_{q=0}^{n-1} S(\rho_q), \qquad S(\rho_q) = -\mathrm{Tr}(\rho_q \log_2 \rho_q).$$

### Two-moons variants

All four models share the outer layers $\mathrm{Linear}(2 \to 4) \to \tanh \to \mathrm{Middle}(4 \to 4) \to \mathrm{Linear}(4 \to 2)$ and differ only in the middle block:

| Variant | Middle block | Middle params | Total params |
|---|---|---:|---:|
| `hybrid_entangled` | $n=4, L=3$ quantum circuit, CNOT ring on | 36 | 58 |
| `hybrid_no_entanglement` | $n=4, L=3$ quantum circuit, CNOT ring off | 36 | 58 |
| `classical_mlp` | $\mathrm{Linear}(4 \to 4) \to \tanh \to \mathrm{Linear}(4 \to 4)$ | 40 | 62 |
| `no_middle` | Identity | 0 | 22 |

Middle parameter counts for the quantum and MLP blocks are close (`36` vs. `40`) rather than identical. For any seed $s$, the outer linear layers are initialized from a separate RNG stream before constructing the middle block so all four variants start from identical input and readout weights.

### Parity / XOR variants

To prevent classical layers from solving XOR before the quantum circuit, each input feature $x_q$ ($q \in \{0, 1\}$) passes only through its own scalar scale and shift:
$$u_q = \alpha_q x_q + \beta_q, \qquad \frac{\partial u_q}{\partial x_r} = 0 \quad (q \neq r),$$
initialized at $\alpha_q = 1, \beta_q = 0$. Those two values feed directly into the $n = 2, L = 3$ quantum block (`18` middle parameters) followed by $\mathrm{Linear}(2 \to 2)$ (`28` total parameters). We set $s = \pi/2$ here because $s = \pi$ maps $+1$ and $-1$ to states that differ only by a global phase ($e^{\pm i\pi/2}$), collapsing opposite clusters onto the same physical state. Alongside `parity_entangled` and `parity_no_entanglement`, we run the standard `classical_mlp` (`62` parameters) as an unconstrained reference showing that classical networks solve the task easily once feature mixing is allowed.

## 2. Datasets

Both datasets are generated in code in `float64` and standardized per seed using `StandardScaler` fit on the training split only:

- **Two moons:** `sklearn.datasets.make_moons` with noise `0.20`. Each seed $s$ draws `300` fresh balanced training points (seed `100000 + s`); all seeds are evaluated on one fixed `5,000`-point balanced test split (seed `900001`).
- **Four-cluster parity:** Four isotropic Gaussians centered at $(\pm 1, \pm 1)$ with standard deviation `0.25`. Clusters at $(+1, +1)$ and $(-1, -1)$ have label `0`; $(+1, -1)$ and $(-1, +1)$ have label `1`. Each seed draws `75` training points per cluster (`300` total, seed `150000 + s`) against a fixed `5,000`-point test split (`1,250` per cluster, seed `900002`).

## 3. Training Schedule and Pilot Check

Models minimize cross-entropy loss using Adam with learning rate `0.05` and mini-batches of `32` (keeping the final smaller batch), reshuffled each epoch from a dedicated seeded generator (`400000 + seed`). No weight decay, learning-rate schedule, early stopping, or checkpoint selection is used; test metrics are taken from the final epoch (`40` epochs on two moons, `60` on parity), while per-epoch curves serve only as diagnostics.

Before running confirmatory seeds (`0` to $N - 1$), we checked training-loss convergence once on separate pilot seeds (`1000` to `1002`). Every variant's training loss plateaued within `40` epochs on moons and `60` epochs on parity, so those epoch counts were kept unchanged without looking at test comparisons.

During seed-0 runs, 2D decision-grid probabilities (`100 x 100` in `standard`, `40 x 40` in `quick` and `smoke`) are saved at epochs `1, 6, 11, 16, 21, 26, 31, 36`, and the final epoch so boundary plots and animations can be rebuilt from raw files without retraining.

## 4. Statistics and Decision Rules

The unit of analysis is the paired seed $s \in \{0, \dots, N-1\}$ ($N = 20$ in `standard`), where variants share training data, outer weights, and batch order. On final-epoch test accuracy, we compute paired differences $d_s = a_s - b_s$, their mean $\bar{d}$, and a 95% percentile bootstrap interval over `10,000` resamples (fixed bootstrap seed `20260605`), reporting effect sizes and intervals directly rather than p-values.

With equivalence margin $\delta = 0.02$, every hypothesis uses the same decision rule:
- 95% CI strictly inside $(-\delta, +\delta)$: **Practically equivalent**
- 95% CI strictly above $0$: **A higher** (marked `> margin` if the lower bound exceeds $+\delta$)
- 95% CI strictly below $0$: **B higher**
- Otherwise: **Inconclusive**

Advance predictions tested against this rule:
- **H1:** `hybrid_entangled` minus `classical_mlp` on moons -> Practically equivalent.
- **H2:** `hybrid_entangled` minus `hybrid_no_entanglement` on moons -> Practically equivalent.
- **H3:** `parity_entangled` minus `parity_no_entanglement` on parity -> Entangled higher by more than $+\delta$ (requires CI lower bound $> +0.02$).

Because an additive classifier of the form $g_0(x_0) + g_1(x_1)$ can separate at most three of the four balanced XOR quadrants ($\approx 0.75$ accuracy), H3 includes a leak check: if `parity_no_entanglement` averages above $0.75 + 0.05 = 0.80$, H3 is marked invalid. Runs outside unmodified `--preset standard` are labeled exploratory, and runs with $N < 10$ seeds label verdicts preliminary.

## 5. Validation Checks (`hybrid-qml doctor`)

1. **Reference simulator:** An independent dense Kronecker-product `complex128` statevector simulator written directly from gate matrices matches `QuantumBlock` within `1e-9` across $n \in \{2, 4\}$, $L \in \{1, 3\}$, and entanglement on/off, with state norms within `1e-12` of `1`.
2. **Entanglement switch:** With entanglement off, changing one input changes other qubit outputs by at most `1e-12` and keeps mean single-qubit entropy $\le 10^{-10}$; with entanglement on, cross-qubit sensitivity and entropy are strictly positive.
3. **Gradients:** Backpropagation gradients match exact two-term parameter-shift gradients $\frac{1}{2}[f(\theta + \frac{\pi}{2}) - f(\theta - \frac{\pi}{2})]$ within `1e-6` relative $L_2$ error.
4. **Parity feature isolation:** Off-diagonal derivatives $\partial u_i / \partial x_j$ ($i \neq j$) in the parity input layer equal `0.0` identically.
5. **Data and pairing:** Asserts split sizes (`300`/`5,000`), class balance, zero train-test duplicates, standardized train mean/std, parity cluster label signs, identical seed-0 outer weights and batch permutations across variants, and parameter counts.
6. **Negative control:** Training `hybrid_entangled` on shuffled moons labels across `3` seeds (`8000`–`8002`) keeps mean test accuracy within `0.07` of `0.50`.

## 6. Threats to Validity

Results reflect noiseless statevector simulation on 2 to 4 simulated qubits and two simple 2D synthetic tasks under one fixed Adam learning rate (`0.05`). They do not establish behavior under hardware noise, shot sampling, deeper circuits, or higher-dimensional datasets. Each HTML report also checks six run-level conditions automatically: seed count $< 10$, exploratory flags, shortened `smoke`/`quick` schedules, CUDA device usage, dirty or missing git commit state, and the `0.80` parity additive ceiling.

## 7. References

1. Bowles, J., Ahmed, S., & Schuld, M. (2024). *Better than classical? The subtle art of benchmarking quantum machine learning models.* arXiv:2403.07059.
2. Bergholm, V., Izaac, J., Schuld, M., et al. (2018). *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* arXiv:1811.04968.
3. Paszke, A., Gross, S., Massa, F., et al. (2019). *PyTorch: An imperative style, high-performance deep learning library.* NeurIPS 2019 (arXiv:1912.01703).
4. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J. I. (2020). *Data re-uploading for a universal quantum classifier.* Quantum, 4, 226 (arXiv:1907.02085).
5. Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization.* arXiv:1412.6980.
6. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). *Scikit-learn: Machine learning in Python.* Journal of Machine Learning Research, 12, 2825-2830 (arXiv:1201.0490).
