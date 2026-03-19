# Viva Q&A — Invisible Variables Engine

25 likely questions from professor and industry review panels, with crisp model answers.

---

## Core Concept Questions

---

**Q1. What exactly is a latent variable in the context of IVE?**

A latent variable is an unmeasured condition that is not present in the feature set but has a systematic, statistically significant effect on the target variable. In IVE, latent variables are not assumed to exist — they are discovered by analyzing where and how a trained model consistently fails. A variable qualifies as "latent" if its absence creates a predictable pattern in residuals that survives statistical testing and bootstrap validation.

---

**Q2. Why analyze residuals instead of the raw predictions or features?**

Residuals — the difference between predicted and actual values — represent exactly what the model doesn't know. If a hidden variable influences outcomes, the model's errors will cluster in a way that correlates with that variable's values. Raw features have already been used by the model, so they don't reveal what's missing. Predictions average out signal and noise. Residuals isolate the gap between understanding and reality, which is precisely where hidden variables show up.

---

**Q3. Why do you use both Linear Regression and XGBoost?**

The two models have fundamentally different biases. Linear regression misses non-linear interactions but produces residuals that are simple and interpretable. XGBoost captures complex patterns but may overfit. If a pattern appears in the residuals of *both* models, it is robust — it's not an artifact of one model's specific bias. If it only appears in one, we treat it with lower confidence. Using both gives us a more reliable signal on what's genuinely missing from the data.

---

**Q4. What are out-of-fold predictions and why do they matter?**

In K-fold cross-validation, each sample is predicted exactly once by a model that was trained without seeing that sample. These are called out-of-fold predictions. The residuals from these predictions are unbiased — the model never saw the sample, so the error isn't inflated by memorization. If we used training residuals instead, patterns would reflect what the model failed to overfit, which is uninformative. Out-of-fold residuals reflect genuine model blind spots, which is what we want to scan for hidden structure.

---

**Q5. What is a KS test and why do you use it for subgroup discovery?**

The Kolmogorov-Smirnov (KS) test measures whether two distributions are statistically different. For each feature bin, we compare the distribution of residuals inside the bin to the distribution outside the bin. A large KS statistic with a small p-value means the model behaves significantly differently in that bin — a strong signal that something unrecorded is driving outcomes for that subgroup. It's non-parametric, so it makes no assumption about the distribution shape, which is appropriate for residuals.

---

**Q6. Why is Bonferroni correction needed?**

When you test many hypotheses simultaneously — one KS test per feature bin, across many features — the probability of getting at least one spurious significant result grows rapidly. Bonferroni correction divides the significance threshold by the number of tests, ensuring the experiment-wide false positive rate stays at the intended level (e.g., 5%). Without it, IVE would flag many noise patterns as significant. With it, only genuinely anomalous bins survive the filter.

---

**Q7. Why HDBSCAN instead of k-means or DBSCAN?**

k-means requires specifying the number of clusters in advance, which is not possible when you don't know how many hidden conditions exist. DBSCAN requires a fixed epsilon (neighborhood radius) that needs tuning per dataset. HDBSCAN — Hierarchical DBSCAN — is density-adaptive: it finds clusters of varying density and sizes, handles noise natively, and does not require the number of clusters to be specified. For our use case, where we don't know the shape, count, or density of hidden variable clusters, HDBSCAN is the most appropriate choice.

---

**Q8. What is bootstrap validation and why does IVE use it?**

Bootstrap validation tests whether a detected pattern is reproducible. We draw 50 random resamples of the original data (with replacement) and re-apply the candidate variable's construction rule to each resample. We then check three survival gates: variance of the score distribution, range of scores, and the fraction of rows where the pattern fires (support rate). A pattern that passes all gates on most resamples is considered stable. One that collapses under resampling is rejected as likely noise. This eliminates variables that were detected due to the specific composition of the original sample rather than a genuine underlying signal.

---

**Q9. What are the three survival gates in bootstrap validation?**

1. **Variance gate** — the score distribution must not be near-constant. A genuine hidden variable creates variation in scores.
2. **Range gate** — the maximum minus minimum score must exceed a floor. This prevents trivially flat distributions from passing.
3. **Support gate** — the fraction of rows where the variable fires must be within bounds. Too sparse (< 1%) suggests it's a one-off; too broad (> 95%) suggests it's not discriminating.

All three gates must pass on a majority of resamples for the variable to be validated.

---

**Q10. How is this different from standard feature engineering?**

Feature engineering requires a human hypothesis: "I think temperature might matter, let me add it." IVE requires no prior hypothesis. It discovers candidates purely from statistical patterns in model errors. Additionally, feature engineering produces features for a model to use directly. IVE produces *evidence* that a measurement gap exists — pointing to what should be collected or investigated. The two are complementary: IVE identifies the gap, feature engineering fills it.

---

**Q11. How is this different from SHAP or feature importance?**

SHAP explains which existing features the model weighs most heavily in its predictions. It requires the feature to be in the dataset to analyze it. IVE analyzes features that are *not* in the dataset — specifically, features that are missing and causing systematic errors. SHAP tells you "distance matters a lot." IVE tells you "there's something about certain distances that your model doesn't understand, and here's where it's failing." They answer different questions.

---

## Architecture and Design Questions

---

**Q12. Why FastAPI for the API layer?**

FastAPI is built on ASGI (async), which allows the API to handle many concurrent requests without blocking — important because IVE experiments are long-running. It uses Pydantic v2 for request validation and response serialization, which provides type safety and clear error messages. The automatic OpenAPI documentation generation at `/docs` is a practical engineering benefit. It is also one of the fastest Python web frameworks by benchmark.

---

**Q13. Why Celery for background processing?**

ML analysis is not a sub-second operation. Training models, computing residuals, and running bootstrap validation can take 30 seconds to several minutes depending on dataset size. Doing this in an HTTP request would timeout. Celery decouples the request from the work: the API queues the task and returns immediately with a task ID, and the client polls for progress. This enables concurrent experiments, retry logic, and visibility into worker utilization via Flower.

---

**Q14. Why PostgreSQL instead of a simpler database?**

IVE's data model has relational structure: datasets have many experiments, experiments have many patterns and latent variables, patterns relate to specific columns. PostgreSQL handles this naturally with foreign keys, cascades, and transactions. It also scales well. We use SQLAlchemy 2.0 with async support for non-blocking database operations, and Alembic for reproducible schema migrations. A simpler key-value store would require application-level joins and lack schema enforcement.

---

**Q15. What does Alembic do in this project?**

Alembic manages database schema migrations. When the data model changes — a new column is added, a table is renamed — Alembic generates a versioned migration script. Running `alembic upgrade head` applies all pending migrations. This ensures the database schema stays in sync with the codebase across all environments (development, staging, production) without manual SQL.

---

**Q16. Why have both demo and production modes?**

Synthetic datasets used in demos have known strong effects, moderate support rates, and are designed to be detectable. Applying production-grade thresholds to them would be overly conservative and could reject signals that are clearly real in the controlled context. Demo mode relaxes thresholds specifically calibrated for synthetic data. Production mode is stricter — a higher stability threshold, tighter variance and range floors — because false positives in production environments carry real operational cost. Separating the modes prevents contaminating production confidence estimates with demo-tuned permissiveness.

---

## Methodology and Validity Questions

---

**Q17. What is Cohen's d and how is it used here?**

Cohen's d is a standardized measure of effect size: the difference between two group means divided by the pooled standard deviation. IVE uses it to measure whether the residuals inside a detected subgroup are meaningfully different from those outside — not just statistically significant (which even tiny differences can be with large datasets), but practically significant. We require a minimum Cohen's d of 0.15 (demo) or 0.20 (production) to ensure detected patterns have real-world impact, not just statistical artifacts.

---

**Q18. How do you prevent false positives in subgroup detection?**

Three layers: (1) Bonferroni correction reduces the false positive rate from the multiple testing problem. (2) Minimum effect size (Cohen's d) ensures detected patterns have practical significance. (3) Bootstrap validation is the final filter — only patterns that consistently reconstruct across resampled datasets are retained. Each layer independently reduces noise, and together they make it highly unlikely that a random fluctuation survives to become a validated latent variable.

---

**Q19. What happens when a pattern is detected but rejected in bootstrap?**

The candidate variable is marked "rejected" and receives a `rejection_reason` — one of: `low_presence_rate`, `low_variance`, `low_range`, `support_too_sparse`, or `support_too_broad`. The `bootstrap_diagnostics` dict records per-gate failure counts and aggregate statistics so the reason is transparent. The explanation generator produces a clear human-readable message: "this signal was not stable enough across resampled datasets." Rejected variables are stored in the database but excluded from recommendations.

---

**Q20. How do you know IVE found the *right* hidden variable on the demo datasets?**

Each demo dataset ships with a `metadata.json` file that specifies the exact hidden rule (e.g., `distance > 10 AND traffic > 7.5 → +25 min`). After running IVE, we check whether the validated latent variables correspond to the stated hidden rule. The platform includes a statistical test suite (`tests/statistical/`) that verifies IVE recovers the ground truth variable in each demo dataset with high confidence. This is the platform's empirical correctness validation.

---

## Business and Practical Questions

---

**Q21. How would a company actually use IVE in practice?**

A data team would connect IVE to their existing prediction pipeline. After a model is trained and deployed, they periodically run IVE on recent production data to analyze residuals. If IVE identifies a validated latent variable, the data engineering team investigates whether the corresponding real-world condition can be measured and added to the feature set. The next model version trains on enriched data. IVE can be run as part of a model monitoring workflow to continuously audit for new hidden variables as the data distribution evolves.

---

**Q22. What are the limitations of IVE?**

Several. First, IVE requires a baseline model and target variable — it does not work on unsupervised problems. Second, it identifies *where* the model fails and what kind of pattern explains the failure, but it cannot name the hidden variable. Naming requires domain knowledge. Third, the bootstrap validation makes IVE more conservative — strong effects will be found, but very subtle or very rare hidden variables may not survive all three survival gates. Fourth, IVE currently operates on static datasets; it does not natively handle streaming or time-series data. Finally, the system requires a working ML pipeline (API, Celery, PostgreSQL, Redis) which adds operational overhead compared to a notebook.

---

**Q23. What would you improve or add next?**

The highest-priority improvements are: (1) **time-series support** — detecting latent variables that are temporal (e.g., the hidden condition only appears certain hours of the day); (2) **interaction variable detection** — the current system handles single-feature subgroups, but some hidden variables are interactions between two features that don't show up marginally; (3) **automated data collection recommendations** — not just "investigate this pattern" but "here are the specific measurements you should add to your next data collection cycle"; (4) **closed-loop retraining** — after a hidden variable is validated, automatically generate a synthetic proxy feature and retrain the model to demonstrate quantitative impact.

---

**Q24. How does this scale to large datasets?**

The current implementation is CPU-bound on the worker. For large datasets, three levers are available: (1) SHAP sampling limits computation to a representative sample (configurable via `SHAP_SAMPLE_SIZE`); (2) Celery concurrency can be scaled horizontally by adding more worker containers; (3) for very large datasets, a stratified sample can be passed to Phase 3 detection while Phase 2 residuals are computed on the full dataset. The database layer is async and connection-pooled, so the API and storage layers scale independently of the compute-intensive analysis.

---

**Q25. How is this a contribution beyond combining existing tools?**

Each component in IVE — KS tests, HDBSCAN, bootstrap resampling — is an established technique. The contribution is in the **synthesis and the framing**. No existing platform combines: (1) out-of-fold residual analysis with (2) multi-method pattern detection with (3) rule-preserving candidate synthesis with (4) three-gate bootstrap validation with (5) business-ready explanation generation — as a unified, production-deployable system with a REST API and monitoring infrastructure.

The novelty is also in the operationalization: IVE is not a research method, it is a production service that can be integrated into enterprise ML workflows. Bridging that gap — from method to system — is the core engineering contribution.
