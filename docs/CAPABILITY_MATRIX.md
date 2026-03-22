# IVE Capability Matrix & Support Policy

## Executive Summary

The **Invisible Variables Engine (IVE)** is purpose-built for structured tabular datasets where a numeric target column exhibits meaningful predictive residual structure.  IVE works by training base models (Linear, XGBoost), extracting out-of-fold residuals, and then searching those residuals for systematic subgroup patterns that suggest the existence of an unmeasured latent variable.

This mechanism depends on two conditions:

1. **A numeric target column** that at least partially reflects the influence of the hidden construct.
2. **Residual structure** — if a standard model already explains the target perfectly, or if the signal is dominated by irreducible noise, no hidden structure will be found.

This document describes what IVE currently supports, what is partially supported or experimental, and what lies outside its design envelope.  It is intended for engineering teams, data scientists, and business stakeholders evaluating the tool for a specific use case.

---

## Capability Matrix

| Dataset Type | Support Level | Recommended Mode | Notes |
|---|---|---|---|
| **Numeric regression** | ✅ Supported | Production | Primary use case. Linear and XGBoost residuals are reliable; subgroup detection is well-calibrated. |
| **Binary classification** | ⚠️ Partial | Demo | Target must be numeric (0/1). Residuals from regression models on binary targets are noisier; use Demo mode for exploratory work. Classification-native residuals are not yet implemented. |
| **Multiclass classification** | 🔬 Experimental | Demo | Requires binarising or encoding the target externally. Not an official workflow; treat findings as hypothesis-generating only. |
| **Mixed numeric/categorical CSVs** | ✅ Supported | Production | Categorical columns are dropped before modelling unless pre-encoded. IVE operates on numeric features only. Encode categoricals upstream for best results. |
| **High-cardinality categorical datasets** | ⚠️ Partial | Demo | Columns with many unique string values are excluded automatically. Encoding (target, ordinal, or embedding) is required before ingestion. |
| **Text-heavy datasets** | ❌ Not Recommended | — | Free-text columns are ignored by the pipeline. If the bulk of signal lives in text, IVE will not detect it. |
| **Datetime-heavy datasets** | ⚠️ Partial | Demo | Datetime columns are excluded; numeric features must be derived externally (e.g. day-of-week, elapsed time). Time-series correlation is not modelled. |
| **All-categorical datasets** | ❌ Not Recommended | — | IVE cannot model or detect patterns without numeric features. Pre-encode all columns before use. |
| **All-text datasets** | ❌ Not Recommended | — | Completely outside the current design envelope. Use NLP-specific tooling. |
| **Missing-value-heavy datasets** (>30% NaN per column) | ⚠️ Partial | Demo | IVE imputes NaNs with column medians. High missingness may distort residuals and produce false or weak patterns. Impute upstream for reliable results. |
| **Small datasets** (<100 rows) | ❌ Not Recommended | — | Cross-validation and bootstrap validation are statistically underpowered. Findings should not be acted upon. |
| **Medium datasets** (100–100 k rows) | ✅ Supported | Production | Optimal operating range. All pipeline stages are stable and statistically meaningful. |
| **Large datasets** (>100 k rows) | ⚠️ Partial | Production | Supported but may increase runtime significantly. Subgroup discovery scales linearly; HDBSCAN clustering may require tuning. Consider sampling for exploratory runs. |
| **Synthetic hidden-variable datasets** | ✅ Supported | Demo | Designed to validate IVE's detection capability. Demo mode thresholds are tuned for high recall on controlled datasets. |
| **Noisy real-world operational datasets** | ✅ Supported | Production | Production mode applies stricter thresholds to reduce false positives. Expect lower recall; findings carry higher confidence. |

---

## Best Practices

### Target Column

- The target column **must be numeric** (integer or float).  Binary 0/1 columns are accepted but treated as regression targets.
- Avoid targets that are entirely deterministic (e.g. a primary key derived column) — residuals will be near-zero and no patterns will be found.
- Targets with very long tails (e.g. revenue, page views) should be log-transformed upstream to improve model fit and residual interpretability.

### Row Count

| Rows | Recommendation |
|---|---|
| < 100 | Do not use IVE |
| 100–999 | Demo mode only; treat results as exploratory |
| 1 000–100 000 | Full support; both modes viable |
| > 100 000 | Supported; consider sub-sampling for speed |

### Feature Balance

- Aim for **5–50 numeric feature columns**.  Too few features give models nothing to condition on.  Very high-dimensional datasets (hundreds of features) increase the likelihood of spurious subgroup correlations.
- Remove or encode non-numeric columns **before upload**.  IVE silently drops columns it cannot use.
- Colinear feature blocks do not cause errors but may dilute detection power.  Light feature selection upstream improves signal clarity.

### Choosing Demo vs. Production Mode

| | Demo Mode | Production Mode |
|---|---|---|
| **Purpose** | Exploration, prototyping, demonstrations | Actionable findings |
| **Bootstrap threshold** | Relaxed (≥ 20% presence rate) | Strict (≥ 40% presence rate) |
| **False positive risk** | Higher | Lower |
| **Recall** | Higher | Lower |
| **When to use** | New datasets, feasibility studies, verifying IVE works on your data | Downstream decisions, stakeholder reporting |

### "No Latent Variables Found" Is a Valid Outcome

IVE is a **detection tool, not a guarantee**.  If no validated latent variables are returned:

- The dataset may already be well-explained by the provided features.
- The target signal may be dominated by irreducible noise.
- The hidden construct, if present, may not project strongly enough onto the residual space to exceed detection thresholds.

Absence of findings should be interpreted as "no strong evidence of hidden structure" — not as a system error.

---

## Known Limitations

### Text-Rich Datasets
IVE has no text-processing capability.  Free-text columns (comments, descriptions, notes) are silently excluded.  If the hypothesis involves a latent construct that manifests primarily through language (e.g. customer sentiment), IVE is not the right tool without upstream text embedding.

### Unsupported Data Modalities
IVE does not support:
- **Image or video data**
- **Audio or time-series signals**
- **PDF or document corpora**
- **Graph-structured data**

All such data must be reduced to a numeric tabular representation before use.

### High-Dimensional Sparse Problems
Datasets with many features relative to rows (e.g. genomics, text BoW matrices) are prone to noise-driven subgroup findings.  IVE does not apply regularisation to the detection stage.  Dimensionality reduction (PCA, UMAP) upstream is strongly recommended.

### Weak or Absent Signal
A latent variable is only detectable if its effect:
1. Influences the numeric target.
2. Creates systematic residual error patterns in at least one feature subgroup.
3. Is stable enough to survive bootstrap resampling.

If the hidden construct's effect size is small relative to noise, IVE will correctly return no findings.  This is expected behaviour, not a defect.

### No Causal Inference
IVE identifies **correlational subgroup structure** in model residuals.  It does not establish causality.  All output should be treated as hypothesis-generating evidence requiring domain validation.

---

## Practical Guidance

### When to Use IVE

- You have a tabular dataset with a numeric outcome and you suspect there is an unmeasured factor driving unexplained variance.
- Your models plateau at a performance ceiling despite adding features.
- Domain experts believe a latent construct (e.g. customer frustration, operational stress) is influencing outcomes but is not captured in the data schema.
- You are conducting an exploratory analysis to surface data collection gaps.

### When _Not_ to Use IVE

- Your dataset is entirely categorical or text-based without numeric features.
- Your dataset has fewer than 100 rows.
- You need causal attribution or counterfactual reasoning — IVE provides correlational evidence only.
- Your target column is not numeric (multiclass string labels, ordinal categories) without external preprocessing.
- You require real-time or streaming inference — IVE is a batch analytical tool.

### Interpreting Output Responsibly

1. **Validated ≠ Confirmed.**  Bootstrap validation increases confidence but does not replace domain expert review.  Present findings as hypotheses.
2. **Effect sizes matter.**  A validated latent variable with a low stability score or small effect size warrants less confidence than one with high scores.
3. **Reproduce on held-out data.**  If your dataset is large enough, validate IVE findings by checking whether the identified subgroup structure replicates in a portion of data not used during analysis.
4. **Document the analysis mode.**  All reports should specify whether Demo or Production mode was used, as thresholds differ materially.
5. **Communicate uncertainty.**  IVE surfaces statistical patterns.  Business decisions based on IVE output should include domain validation and, where appropriate, controlled experiments.

---

*Last updated: March 2026.  For questions or to report unexpected behaviour, open an issue in the project repository.*
