# Credit Scoring Business Understanding

## 1. Basel II and the Need for Interpretable Models

The Basel II Accord places strong emphasis on accurate risk measurement, transparency, and capital adequacy in financial institutions. Because banks must justify how much capital they hold against potential losses, credit risk models must be auditable, explainable, and stable over time. This directly increases the need for interpretable models such as scorecards or Logistic Regression with Weight of Evidence (WoE), because:

- Regulators must be able to trace how predictions are made
- Financial decisions (approve/reject loans) must be explainable to stakeholders
- Models must support validation, stress testing, and governance
- Black-box models without explanation are often difficult to approve in regulated environments

In short, Basel II does not only require accurate predictions—it requires understandable risk reasoning, not just output probabilities.

## 2. Why Proxy Variables Are Necessary Without Direct Default Labels

In many real-world credit datasets, a clean “default” label is not always available. For example:

- Customers may not explicitly default but become inactive
- Data may only show late payments, charge-offs, or account closures
- Historical datasets may not track formal default events

To address this, a proxy variable (e.g., “90+ days past due”, “write-off status”) is used as a substitute target.

Why proxies are needed:
- Enables supervised learning when true labels are missing
- Allows model training on historical behavioral patterns
- Provides a measurable definition of “bad risk”
- Business risks of using proxies:

However, proxy-based labeling introduces several risks:

- Label noise: Proxy may not perfectly represent true default behavior
- Misclassification bias: Good customers may be labeled as bad (or vice versa)
- Policy dependency: The model reflects past business rules, not true risk
- Distribution shift: Changes in policy or collection strategies can distort future predictions
- Ethical risk: Certain customer groups may be unfairly impacted due to imperfect labeling

Therefore, proxy selection must be carefully validated because it directly determines the model’s business behavior.

## 3. Trade-offs Between Interpretable Models and High-Performance Models

In regulated financial environments, there is a fundamental trade-off between interpretability and predictive performance.

### (A) Interpretable Models (e.g., Logistic Regression + WoE)

#### Advantages:

Highly interpretable and transparent
Easy to explain to regulators and business teams
Stable and robust over time
Naturally aligned with scorecard development
Supports monotonic relationships via binning

#### Disadvantages:

Limited ability to capture complex nonlinear patterns
May underperform on large, high-dimensional datasets
Requires manual feature engineering (WoE/IV)
### (B) High-Performance Models (e.g., Gradient Boosting, XGBoost)

#### Advantages:

Superior predictive accuracy
Captures nonlinear relationships and feature interactions
Handles large and complex datasets effectively
Requires less manual feature engineering

#### Disadvantages:

Low interpretability (black-box behavior)
Difficult to justify to regulators without explainability tools (SHAP, LIME)
Harder to convert into traditional scorecards
May introduce overfitting or instability if not carefully tuned

## 4. Business Trade-off Summary
Criteria	Interpretable Models	High-Performance Models
Explainability	High	Low
Predictive Power	Medium	High
Regulatory Acceptance	High	Conditional
Stability	High	Medium

## 5. Final Insight

In regulated credit scoring systems, the best model is often not the most accurate one, but the one that balances:

- Regulatory compliance (Basel II requirements)
- Business interpretability
- Predictive performance
- Fairness and stability

This is why many institutions use a hybrid approach, where:

- Logistic Regression scorecards are used for decisioning
- Machine learning models are used for risk insights and monitoring