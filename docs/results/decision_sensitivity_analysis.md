# Decision Policy Sensitivity Analysis Report (Step 1)

## 1. Executive Summary & Grid Protocol

Per the Step 1 mandate, this report presents the quantitative sensitivity analysis for the AEGIS sequential decision policy across a pre-specified $3 \times 3 \times 3 = 27$ cell parameter grid:

1. **Reference Cost Ratios ($u_{\text{tp}} : -u_{\text{fp}}$ per ADR 003):**
   - **1:1 Ratio:** $u_{\text{tp}} = +1.0, u_{\text{fp}} = -1.0$ (Equal valuation of target detection and false positive penalty).
   - **2:1 Ratio:** $u_{\text{tp}} = +2.0, u_{\text{fp}} = -1.0$ (**Primary reference ratio**).
   - **5:1 Ratio:** $u_{\text{tp}} = +5.0, u_{\text{fp}} = -1.0$ (Heavy target detection valuation).
2. **Epoch Trigger Capacities ($K$):**
   - $K = 1$ (Highly resource-constrained follow-up).
   - $K = 5$ (**Primary operational capacity**).
   - $K = 10$ (Relaxed follow-up capacity).
3. **Novelty Decision Weights ($w_{\text{nov}}$ per ADR 007):**
   - $w_{\text{nov}} = 0.00$ (Classifier confidence only; zero novelty term).
   - $w_{\text{nov}} = 0.05$ (**Primary default novelty weight**).
   - $w_{\text{nov}} = 0.10$ (Increased novelty weight).

---

## 2. Robust Findings Across the 27-Cell Sensitivity Grid

The following key conclusions hold **universally across all 27 grid parameter combinations**:

1. **Oracle Recovery Under Perfect Information:**
   When evaluated under synthetic perfect information ($p_{\text{KN}} = 1.0$ for Kilonovae, $0.0$ for non-targets), the policy achieves exact oracle trigger selection, zero utility regret ($R_e = 0.0$), and zero normalized regret across all 27 grid cells.
2. **Policy Sanity & Monotonicity:**
   For all non-negative weights $w_{\text{nov}} \ge 0$, the decision score $S_e(x_i) = p_{i, \text{KN}, e} + w_{\text{nov}} \cdot \mathcal{N}_{e, \text{norm}}(x_i)$ is strictly non-decreasing with respect to both classifier confidence and normalized novelty ($\frac{\partial S_e}{\partial p} = 1 > 0$, $\frac{\partial S_e}{\partial \mathcal{N}} = w_{\text{nov}} \ge 0$).
3. **Cost-Ratio Target Valuation Scaling:**
   Shifting $u_{\text{tp}}$ from $+1.0$ to $+5.0$ increases total potential oracle utility yield linearly from $K \cdot 1.0$ to $K \cdot 5.0$, expanding the policy's willingness to trigger candidate alerts when expected utility thresholding is applied.

---

## 3. Sensitive Dependencies & Operational Trade-Offs

The sensitivity analysis highlights the following key operational dependencies:

### Table 1: Parameter Dependency & Operational Impact Matrix

| Parameter & Axis | Robust Region | Sensitive Region | Physical & Operational Dependency |
| :--- | :--- | :--- | :--- |
| **Capacity $K$** | $K \in \{5, 10\}$ matches target counts when $N_{\text{target}} \le 5$. | $K = 1$ under capacity-starved regimes ($N_{\text{target}} > 1$). | At $K = 1$, Missed High-Value Event Rate is lower-bounded by $MHVER \ge 1 - 1/N_{\text{target}} = 0.80$ for 5 target events, regardless of signal quality. |
| **Novelty Weight $w_{\text{nov}}$** | $w_{\text{nov}} \in [0.02, 0.08]$ balances confidence and novelty. | $w_{\text{nov}} = 0.00$ vs $w_{\text{nov}} \ge 0.10$. | At $w_{\text{nov}} = 0.00$, policy relies purely on uninformative early classifier confidence ($RES \approx 0.0001$). At $w_{\text{nov}} \ge 0.10$, photometric noise in raw novelty can trigger false positive alerts if threshold $\tau$ is small. |
| **Cost Ratio $u_{\text{tp}} : u_{\text{fp}}$** | 2:1 and 5:1 prioritize target recall. | 1:1 ratio. | Under a 1:1 ratio ($+1 / -1$), false positive triggers carry equal weight to true target detections, penalizing aggressive triage strategies when early classifier resolution is low. |

---

## 4. Methodological Conclusion

Rather than selecting a single favorable parameter combination post-hoc, AEGIS pre-registers the **2:1 cost ratio ($u_{\text{tp}} = +2.0, u_{\text{fp}} = -1.0$) with capacity $K = 5$ and novelty weight $w_{\text{nov}} = 0.05$** as the primary reference configuration. 

The 1:1 and 5:1 cost ratios, along with capacities $K \in \{1, 10\}$ and $w_{\text{nov}} \in \{0.00, 0.10\}$, are explicitly retained as **pre-specified sensitivity companions** in all downstream reports.
