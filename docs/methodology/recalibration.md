# Selection-Aware Recalibration Methodology

This document details the mathematical framework, causal and identifiability assumptions, diagnostics, and residual error decomposition for the **Selection-Aware Recalibration** methodology in AEGIS, closing Phase 3.

---

## 1. Mathematical Framework & IPW Formulation

When a probabilistic classifier is trained on a spectroscopically selected sample ($S=1$) subject to selection bias $P(S=1 \mid X)$, its uncorrected probability predictions $\hat{P}(Y=c \mid X)$ reflect both model overconfidence and the skewed base rates of the selected sample.

To adjust for selection bias without altering the underlying classification features, AEGIS uses **Inverse Probability Weighting (IPW)** based on the empirical selection proxy model $p_{\text{spec}}(z)$ established in ADR 004:

$$p_{\text{spec}}(z_{\text{phot}}) = p_{\text{floor}} + \frac{p_{\text{bright}} - p_{\text{floor}}}{1 + \exp\left(\frac{z_{\text{phot}} - z_{50}}{w_z}\right)}$$

where $p_{\text{bright}} = 0.8$, $p_{\text{floor}} = 0.1$, $z_{50} = 0.5$, and $w_z = 0.15$.

### Importance Weights
Each labeled training object $i \in \{1, \dots, N_{S=1}\}$ is assigned an importance weight $w_i$:

$$w_i = \frac{1}{p_{\text{spec}}(z_{\text{phot}, i})}$$

Because $p_{\text{spec}}(z) \in [0.10, 0.80]$, importance weights are bounded strictly on $[1.25, 10.0]$, preventing extreme weight blow-up.

### Multinomial Platt Recalibrator
All recalibration parameters are estimated **exclusively on the labeled $S=1$ population** using weighted multinomial logistic regression (Platt scaling):

$$\min_{\mathbf{W}, \mathbf{b}} \sum_{i \in S=1} w_i \sum_{c=1}^K - y_{ic} \log \sigma_c(\mathbf{W} \mathbf{\ell}_i + \mathbf{b})$$

where $\mathbf{\ell}_i = \log \hat{\mathbf{p}}_i$ are the baseline classifier log-probabilities and $\sigma_c(\cdot)$ is the softmax function over study classes $c \in \{64, 90, 95\}$.

> [!IMPORTANT]
> **Data Scoping Rule**  
> All recalibrator parameters $(\mathbf{W}, \mathbf{b})$ are fit strictly on $S=1$ training data with sample weights $w_i$. The TRUE population ($S=0$) is never touched during fitting and is used solely for evaluation.

---

## 2. Weight & Covariate Balance Diagnostics

### Weight Distribution & Effective Sample Size ($ESS$)
Importance weights modify the effective statistical capacity of the sample. To detect weight instability, AEGIS computes:

1. **Weight Summary Statistics:** Minimum ($w_{\text{min}}$), median ($w_{\text{med}}$), 95th percentile ($w_{\text{p95}}$), and maximum ($w_{\text{max}}$).
2. **Coefficient of Variation ($CV$):**
   $$CV = \frac{\sigma_w}{\mu_w}$$
3. **Kish's Effective Sample Size ($ESS$):**
   $$ESS = \frac{\left( \sum_{i=1}^{N_{S=1}} w_i \right)^2}{\sum_{i=1}^{N_{S=1}} w_i^2}$$

### Covariate Balance Diagnostics (SMDs)
To confirm that importance weighting aligns the $S=1$ sample covariate distribution with the TRUE population, we compute **Standardized Mean Differences ($SMD$)** before and after weighting:

$$SMD_{\text{unweighted}} = \frac{\mu_{S=1} - \mu_{\text{TRUE}}}{\sigma_{\text{TRUE}}}$$

$$SMD_{\text{weighted}} = \frac{\sum_{i \in S=1} w_i X_i / \sum w_i - \mu_{\text{TRUE}}}{\sigma_{\text{TRUE}}}$$

A feature is considered balanced when $|SMD_{\text{weighted}}| < 0.10$.

---

## 3. Positivity & Overlap Diagnostics

ADR 001 and ADR 004 mandate that selection corrections must not extrapolate into covariate regions with near-zero labeled support.

### Positivity Violation Boundary
We define the positivity violation region as high redshift ($z > 1.50$), where inclusion probability approaches $p_{\text{floor}} = 0.10$ and $S=1$ sample density drops below 10 objects per redshift sub-bin.

For any object $i$ falling into this region ($z_i > 1.50$):
- The object is flagged as `uncorrectable` (positivity violation).
- Extrapolation is explicitly masked: the model returns unextrapolated baseline probabilities $\hat{\mathbf{p}}_i$ rather than extrapolating Platt scaling parameters fitted on low-redshift data.

---

## 4. Identifiability & Causal Assumptions

Selection-aware recalibration relies on four explicit identifiability assumptions:

| Assumption | Definition | Empirically Testable in AEGIS? | Empirical Findings in AEGIS |
| :--- | :--- | :---: | :--- |
| **1. Positivity** | $P(S=1 \mid X) > 0$ for all $X$ in evaluation population. | **Yes** | Violated for high redshift $z > 1.50$ ($5.47\%$ of TRUE population). Masked from extrapolation. |
| **2. Consistency** | Observed outcome under $S=1$ equals counterfactual outcome. | **Yes** | Holds by design in PLAsTiCC simulation (class definitions are invariant to $S$). |
| **3. Conditional Exchangeability (MAR)** | $Y \perp S \mid Z_{\text{phot}}$ (Selection depends only on observed $Z_{\text{phot}}$). | **No** (Untestable) | Partially violated in operational surveys where selection depends on brightness, host surface brightness, and human prioritization. |
| **4. Selection Model Specification** | Empirical $p_{\text{spec}}(z)$ matches true selection probabilities. | **Yes** | Exact match in AEGIS by construction (logistic proxy ADR 004). |

---

## 5. Residual Error Decomposition

Residual calibration error on the TRUE population is decomposed into three distinct physical and statistical mechanisms:

$$\text{Miscalibration Error} = \text{Selection Bias} + \text{Positivity Violations} + \text{Intrinsic Information Deficit}$$

1. **Selection Bias (Correctable in Principle):** Shift in base rates and covariate distribution between $S=1$ and $S=0$. IPW weighting reduces $SMD$ to $-0.0039$.
2. **Positivity Violations (Uncorrectable by Weighting):** Regions with $P(S=1 \mid X) \approx 0$ ($z > 1.50$).
3. **Intrinsic Information Deficit (Uncorrectable by Recalibration):** At early decision epochs ($e \le 2$d), light curves lack discriminative physical features ($RES \approx 0.0001$). Recalibration cannot create resolution where early photometry provides no signal.
