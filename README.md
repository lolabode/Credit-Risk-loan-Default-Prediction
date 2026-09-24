# Credit Risk: Predicting Loan Default with Neural Networks

**A business analytics case study on flagging high-risk loan applicants while limiting the number of good customers turned away.**

*Python · TensorFlow/Keras · scikit-learn · pandas*

---

## At a glance

| | |
|---|---|
| **Business question** | Can a lender identify applicants likely to default before approving a loan? |
| **Data** | 10,000 simulated borrower records, 23 financial and behavioural attributes, 20.6% defaulters |
| **Approach** | Seven neural network configurations compared through controlled, one-change-at-a-time experiments |
| **Outcome** | The selected model caught **64% of defaulters** in unseen data (198 of 309) with an AUC of **0.74** |
| **Key insight** | The more complex model beat a simple baseline by only a small margin. Complexity bought very little, which matters when a lender must also explain its decisions |

![Business outcomes](images/business_outcomes.png)

---

## 1. The business problem

Lenders face two kinds of mistake, and they are not equally costly:

- **Approving a borrower who defaults** (a false negative) causes a direct loss of the unpaid loan balance.
- **Rejecting a borrower who would have repaid** (a false positive) loses the interest income and the customer relationship.

Missing a defaulter is usually far more expensive, so the project was designed to **maximise the share of defaulters caught (recall)** while keeping false alarms at a reasonable level. Because only about 1 in 5 applicants defaults, overall accuracy is misleading here: a model that approves everyone would be 79% "accurate" and useless. Success was therefore measured on **recall, F1-score and AUC**.

## 2. The data

The dataset contains 10,000 simulated loan applications with no missing values. Features cover:

- **Loan details:** amount, term, interest rate, purpose, credit sub-grade
- **Borrower profile:** annual income, employment length, home ownership, verification status, state
- **Credit behaviour:** FICO score, debt-to-income ratio, credit utilisation, recent enquiries, delinquencies and public records

The target is binary: `1` = defaulted, `0` = repaid (7,940 repaid vs 2,060 defaulted).

> **Note:** The data is simulated for learning purposes. Results show the method and reasoning, not real-world lending performance.

## 3. Approach

**Preparation.** Borrower IDs were dropped, the issue date was split into year and month to capture timing effects, `grade` was removed because `sub_grade` holds the same information in more detail, categorical fields were one-hot encoded, and numeric fields were standardised. Data was split **70% training / 15% validation / 15% test**, and **class weights** were applied so the model didn't simply learn to predict "no default".

**Experiment design.** Rather than trial and error, each configuration changed one thing at a time so its effect could be isolated:

| Config | Change tested |
|---|---|
| 1 | Baseline: simple network, standard gradient descent (SGD), no regularisation |
| 2 | + Momentum (faster training) |
| 3 | Adam optimiser, no regularisation |
| 4 | Adam + dropout + early stopping (overfitting controls) |
| 5 | **Deeper network (4 hidden layers) + dropout + early stopping** |
| 6 | Config 5 + L2 weight penalty |
| 7 | Config 5 with a lower learning rate |

The best model was chosen on the validation set and then confirmed on the untouched test set.

## 4. Results

![Model comparison](images/model_comparison.png)

| Config | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| 1 Baseline (SGD) | 0.697 | 0.364 | 0.634 | 0.463 | 0.722 |
| 2 SGD + momentum | 0.681 | 0.318 | 0.482 | 0.384 | 0.662 |
| 3 Adam | 0.716 | 0.324 | 0.350 | 0.336 | 0.645 |
| 4 Adam + dropout + early stopping | 0.704 | 0.365 | 0.592 | 0.452 | 0.716 |
| **5 Deeper + dropout + early stopping** | **0.703** | **0.372** | **0.641** | **0.471** | **0.740** |
| 6 Deeper + L2 | 0.687 | 0.351 | 0.608 | 0.445 | 0.731 |
| 7 Deeper, lower learning rate | 0.693 | 0.354 | 0.599 | 0.445 | 0.711 |

*Test set, 1,500 applicants, default threshold of 0.5.*

**What the selected model (Config 5) means in practice:**

- It caught **198 of 309 defaulters (64%)** and missed 111.
- It flagged **334 of 1,191 good applicants (28%)** as risky.
- Of everyone it flagged, roughly **1 in 3 actually defaulted** (precision 0.37).

In other words, it is useful as an **early-warning screen that routes applicants to further review**, not as an automatic approve/reject decision.

<details>
<summary>Supporting charts: ROC curve and training behaviour</summary>

![ROC curve](images/roc_curve_config5.png)
![Loss curve](images/loss_curve_config5.png)

The training and validation curves stay close together, indicating the model generalises rather than memorising the training data.
</details>

## 5. Key insights

1. **"Stronger" training methods made things worse on their own.** Momentum and Adam fit the training data faster but caught fewer defaulters, dropping recall to as low as 35%. The models learned to favour the majority "repaid" group.
2. **Overfitting controls were what made the difference.** Dropout and early stopping recovered performance, and only then did a deeper network add value.
3. **More regularisation hit diminishing returns.** Adding an L2 penalty or slowing learning reduced performance, suggesting Config 5 was already well balanced.
4. **The gain over the simple baseline was small.** Config 5 improved recall from 0.634 to 0.641 and AUC from 0.722 to 0.740. For a lender, that small gain has to be weighed against the harder-to-explain model.

## 6. Recommendations

- **Use the model as a triage tool.** Send flagged applicants to manual underwriting instead of rejecting them automatically.
- **Set the decision threshold using business costs, not the default 0.5.** The right cut-off depends on the average loss from a default compared with the profit lost from a rejected good customer.
- **Benchmark against an interpretable model before deployment.** Given how close the simple baseline came, a logistic regression or tree-based model may give similar results with clearer explanations, which matters under UK regulatory expectations such as the FCA's Consumer Duty.

## 7. Limitations

- Simulated data, so results won't transfer directly to real lending.
- All metrics use a fixed 0.5 threshold, which ignores the unequal cost of errors.
- Neural networks are hard to explain to applicants and regulators. No explainability analysis was performed.
- A single train/validation/test split; cross-validation would give more reliable estimates.

## 8. Next steps

- [ ] Cost-based threshold analysis to find the cut-off that minimises expected loss
- [ ] Logistic regression and gradient-boosting benchmarks
- [ ] Feature importance / SHAP analysis to show which factors drive risk
- [ ] Interactive dashboard showing the trade-off at different thresholds

---

## Tools

Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · Matplotlib · Jupyter

---

*Completed as part of an MSc in Business Analytics with AI. AI tools supported coding and debugging; all modelling decisions and interpretations are my own.*
