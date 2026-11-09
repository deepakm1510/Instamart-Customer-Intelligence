# Instamart Customer Intelligence & Support Copilot

**Capstone Project - IIT Patna**

## 1. Project Overview

This project builds an end-to-end customer intelligence and support copilot for an Instamart-style quick-commerce platform. It covers data quality, SQL analytics, customer segmentation, lapse prediction, support-ticket triage, policy retrieval, and a tool-using refund-support agent.

### Project Modules

- **Part A – Data Foundation:** Raw-data audit, cleaning, SQLite, SQL analytics and Pandas analysis.
- **Part B – Customer Intelligence:** RFM analysis, K-Means segmentation and customer-lapse prediction.
- **Part C – Support Ticket Triage:** PII masking, zero-shot/few-shot classification, validation, routing and budget controls.
- **Part D – Policy RAG and Refund Agent:** TF-IDF retrieval, policy-grounded answers, refund tools, safeguards and checkpoint recovery.

## 2. Repository Setup and Reproduction

Run commands from the project root. The following commands use Windows PowerShell.

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python generate_data.py
```

Run each module:

```powershell
# Part A
python src/audit_raw_data.py
python src/data_pipeline.py
python src/validate_data.py
python src/run_sql_queries.py
python src/verify_sql_details.py
python src/pandas_analysis.py

# Part B
python src/customer_models.py

# Part C
python src/triage.py

# Part D
python src/policy_rag.py
python src/refund_agent.py
```

The project uses generated/synthetic assignment data and a supplied mock LLM. Results are reproducible with the fixed random seeds and the same dependency versions/runtime.

---

# Part A – Data Foundation

## 3. Raw Data Audit

The raw files contain the following records and columns. Missing percentages are calculated per column using the raw CSV files.

### Customers (`customers.csv`): 1,200 rows × 5 columns

| Column | Pandas dtype | Missing | Missing % |
|---|---|---:|---:|
| customer_id | object | 0 | 0.00% |
| city | object | 0 | 0.00% |
| signup_date | object | 0 | 0.00% |
| age | float64 | 61 | 5.08% |
| gender | object | 30 | 2.50% |

### Products (`products.csv`): 40 rows × 4 columns

| Column | Pandas dtype | Missing | Missing % |
|---|---|---:|---:|
| product_id | object | 0 | 0.00% |
| product_name | object | 0 | 0.00% |
| category | object | 0 | 0.00% |
| price_inr | int64 | 0 | 0.00% |

### Orders (`orders.csv`): 13,084 rows × 9 columns

| Column | Pandas dtype | Missing | Missing % |
|---|---|---:|---:|
| order_id | object | 0 | 0.00% |
| customer_id | object | 0 | 0.00% |
| product_id | object | 0 | 0.00% |
| quantity | int64 | 0 | 0.00% |
| order_ts | object | 0 | 0.00% |
| status | object | 0 | 0.00% |
| discount_pct | int64 | 0 | 0.00% |
| order_amount | object | 53 | 0.41% |
| delivery_minutes | float64 | 2,541 | 19.42% |

### Additional Raw-Data Anomalies

- Customers: 61 blank ages, 6 ages outside the 10–100 audit range, 24 raw city spellings, 10 nonblank gender spellings and 30 blank genders.
- Orders: 119 exact duplicate rows, 12,309 distinct order IDs, 656 pending rows, 120 nonnumeric order amounts, 106 negative delivery times and 2,541 blank delivery times.

## 4. Cleaning and Validation

The pipeline standardizes customer attributes, handles missing/invalid values, removes duplicate order rows, resolves repeated order IDs, parses dates and numeric fields, and writes cleaned tables. The exact rules are implemented in `src/data_pipeline.py`.

| Output | Rows |
|---|---:|
| Clean customers | 1,200 |
| Clean products | 40 |
| Clean orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |

The validation script reports **ALL CHECKPOINTS PASSED**. The SQLite database `data/instamart.db` contains `customers`, `products` and `orders` tables.

## 5. SQL Analysis

Seven SQL questions are implemented in `sql/queries.sql`. The query outputs contain 30, 30, 11, 25, 550, 1,175 and 183 rows respectively (Q1–Q7).

Key cross-checks:

- Completed-order revenue: **INR 2,998,668.80**.
- Q5 customers above the average completed-order spend: **550**.
- Q6 uses ranking functions. Ties can cause `RANK()` and `ROW_NUMBER()` to return different customer selections; the verification found **8 differing rows** between the compared outputs. This is expected tie behaviour, not necessarily a query error.
- Q7 cumulative revenue ends at **INR 2,998,668.80**.

## 6. Pandas Analysis

Pandas independently reproduces the SQL completed-order revenue total.

### City-Wise Completed-Order Analysis

| City | Revenue (INR) | Average order value (INR) | Completed orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |
| **Total** | **2,998,668.80** | **269.91 overall** | **11,110** |

Hyderabad has the highest total revenue in this summary; Mumbai has the highest average order value. These are descriptive results for this dataset, not evidence that city causes higher spending.

### Order Amount Distribution and Outliers

| Statistic | Value (INR unless stated) |
|---|---:|
| Mean | 269.91 |
| Median | 135.00 |
| Skewness | 4.61 |
| First quartile (Q1) | 64.80 |
| Third quartile (Q3) | 297.00 |
| Lower IQR fence | -283.50 |
| Upper IQR fence | 645.30 |
| Outliers below lower fence | 0 |
| Outliers above upper fence | 897 |

The mean is higher than the median and skewness is positive, indicating a right-skewed order-amount distribution. The IQR rule flags 897 high-value observations; an outlier flag does not automatically mean the record is incorrect.

### Correlation

The observed correlations with `order_amount` are: `discount_pct` −0.095, `quantity` 0.087 and `delivery_minutes` −0.003. These values indicate weak linear relationships in this dataset. **Correlation does not establish causation**, and a near-zero linear correlation does not rule out every possible relationship.

---

# Part B – Customer Intelligence

## 7. RFM Segmentation

RFM features are calculated from completed orders for **1,175 customers**:

- **Recency:** Days since the customer's latest completed order.
- **Frequency:** Number of completed orders.
- **Monetary:** Total completed-order spend.

K-Means was evaluated for K = 2 through 8. K = 4 produced the highest silhouette score (**0.526**) among the tested values.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.76 | 0.465 |
| 3 | 1,268.23 | 0.445 |
| 4 | 743.07 | **0.526** |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

### Segment Profiles

| Segment | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) |
|---|---:|---:|---:|---:|
| Occasional | 436 | 28.86 | 3.77 | 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Loyal | 419 | 8.77 | 19.40 | 3,410.02 |
| Dormant | 182 | 109.04 | 2.20 | 685.99 |

Possible business actions include introductory offers for Occasional customers, personalized high-value service for Premium customers, loyalty benefits for Loyal customers and carefully targeted win-back campaigns for Dormant customers. These are suggested actions, not experimentally proven outcomes.

PCA explains **63.4%** of variance with PC1 and **21.8%** with PC2, or **85.2%** together.

## 8. Customer Lapse Prediction

The lapse cohort contains **1,156 customers**, of whom **443 (38.32%)** were labelled lapsed under the project's September activity definition. The split contains 924 training rows and 232 test rows.

A majority-class baseline achieved accuracy **0.6164** and F1 **0.0000**. Logistic Regression and Random Forest were tuned with cross-validation. The selected model is an oversampled Logistic Regression model because the project prioritizes finding likely-to-lapse customers (recall).

### Test-Set Comparison

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.6444 | 0.6517 | 0.6480 |
| Random Forest | 0.6404 | 0.6404 | 0.6404 |
| Oversampled Logistic Regression | 0.5891 | **0.8539** | **0.6972** |

Oversampling is applied to training data only. The test set is held out for final evaluation, not used to fit the scaler, tune preprocessing or resample training examples. The preprocessing steps are part of the model pipeline. The reported test confusion matrices are:

- Logistic Regression: `[[111, 32], [31, 58]]`
- Random Forest: `[[111, 32], [32, 57]]`
- Oversampled Logistic Regression: `[[90, 53], [13, 76]]`

The oversampled model catches more lapsed customers (higher recall) but produces more false positives and has lower precision. This trade-off may be acceptable when missing a potentially lapsed customer is considered more costly than contacting some customers who would not lapse.

### Feature Interpretation

Random Forest's leading feature importances were monetary (**0.277830**), frequency (**0.276787**), recency (**0.111219**), average delivery (**0.061649**) and average discount (**0.053178**).

In the oversampled Logistic Regression coefficients, frequency had a negative coefficient (−1.465240), and the `favourite_category_Electronics` (−0.828886) and `city_Delhi` (−0.495280) indicators were also negative. A negative coefficient means a lower modelled log-odds of lapse relative to the model's reference category, holding other encoded features constant; it is not proof of causation.

The saved model was reloaded and its prediction check passed. The retention shortlist is an illustrative output (`outputs/retention_list.csv`): 2 of the top 10 were labelled lapsed. The Premium segment is assigned using the full analysis window, so using it in this retrospective shortlist is **not a leakage-free prospective evaluation**.

---

# Part C – Support Ticket Triage

## 9. Ticket Classification and Routing

The triage pipeline classifies tickets into billing, delivery, technical and fraud categories. It includes zero-shot and few-shot prompts, JSON validation, keyword fallback, retries with deterministic jitter, input-length and budget controls, and routing rules.

Before model processing, email addresses, phone numbers and order IDs are masked. The run detected **278 PII occurrences**:

| PII type | Occurrences |
|---|---:|
| Email | 55 |
| Phone | 91 |
| Order ID | 132 |
| **Total** | **278** |

Assertions check that the original PII patterns are absent after masking.

### Classification Results

| Measure | Zero-shot | Few-shot |
|---|---:|---:|
| Accuracy | 85.71% | 100.00% |
| Tokens | 10,804 | 20,320 |
| Classified tickets | 147 | 147 |
| Tokens per classified ticket | 73.50 | 138.23 |

Few-shot examples improved evaluation accuracy in this run, while token usage increased. This result is specific to the supplied evaluation tickets and mock LLM; it does not establish the same accuracy on real customer traffic. A zero-shot confusion matrix is not included here because the reviewed saved output does not preserve the zero-shot per-ticket predictions and true labels together.

For the few-shot run, the confusion matrix diagonal counts were billing 38, delivery 36, fraud 34 and technical 39, consistent with the reported 100% accuracy on 147 classified tickets.

### Few-Shot Execution and Routing

The full run produced 164 output rows. Processing statuses included 135 JSON classifications, 12 keyword fallbacks, 9 invalid responses, 4 long-input cases and 4 empty-input cases. Routing totals were:

| Route | Tickets |
|---|---:|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

Priority totals: **119 normal** and **45 high**. Invalid or uncertain classifications are not silently treated as reliable predictions; they can be sent for human review. Fraud/security cases are routed to the security queue according to the implemented rules.

The budget demonstration used a 6,000-token budget and ended with 76 JSON, 6 keyword, 5 invalid, 4 empty, 4 long-input and 69 budget-exhausted statuses. Budget-exhausted tickets are stopped rather than continuing to spend tokens.

---

# Part D – Policy RAG and Refund Agent

## 10. Policy Retrieval

Five policy text documents are split into word-based chunks (chunk size 40, overlap 10). The current retriever uses TF-IDF vectors and cosine similarity and selects the top 3 chunks.

| Evaluation | Overlap 10 | Zero overlap |
|---|---:|---:|
| Expected document retrieved in top 3 | 12/12 | 12/12 |
| Expected fact present in answer | 12/12 | 10/12 |
| Correct fallback for unanswerable questions | 2/2 | 2/2 |

The two facts missed with zero-overlap chunking concerned (1) the delivery delay threshold for a full refund (more than 45 minutes) and (2) the 48-hour refund timeline when a payment is deducted but the order is not placed. In both cases, relevant information was split across chunk boundaries. Overlap preserved enough context for all 12 answerable evaluation facts.

**Limitations:** TF-IDF is a lexical retrieval method. It may miss semantically related wording, and its performance depends on chunk size, overlap and the evaluation questions. The evaluation set is small and does not prove production-level retrieval quality. The current answer-generation helper receives the retrieved chunks from its caller; retrieval and answer generation are therefore separate steps in the current implementation.

## 11. Refund-Support Agent

The ReAct-style agent uses these tools:

1. `get_order` – retrieve order details.
2. `search_policy` – retrieve relevant policy context.
3. `calculate_refund` – calculate an eligible refund.
4. `issue_refund` – record an approved refund.
5. `escalate_to_human` – hand off cases that require human judgement.

The agent includes policy checks, refund limits, iteration limits, token limits, timeout handling and checkpoint-based crash recovery. It writes execution records to `outputs/agent_runs.jsonl` and refund records to `outputs/refund_ledger.jsonl`.

### Recorded Scenario Results

| Scenario | Outcome | Tool calls | Tokens |
|---|---|---:|---:|
| S1 | REFUND | 5 | 1,302 |
| S2 | REFUND | 5 | 1,302 |
| S3 | ESCALATE | 5 | 1,294 |
| S4 | DENIED | 4 | 929 |
| S5 | DENIED | 3 | 580 |
| S6 | NOT_FOUND | 2 | 234 |
| S7 | FALLBACK_MAX_ITERATIONS | 6 | 3,486 |
| **Total** | **7 scenarios** | **30** | **9,127** |

The crash-recovery demonstration resumed from a saved checkpoint and completed the remaining sequence without repeating already completed tool actions. The demonstrated refund amount was INR 128.25. Separate safety demonstrations produced `FALLBACK_TOKEN_LIMIT` (264 tokens) and `FALLBACK_TIMEOUT` (0 LLM calls).

### ReAct and Plan-and-Execute

ReAct alternates between a model decision and a tool observation. This allows the next action to respond to the latest order or policy result. In the recorded seven-scenario run, ReAct used 30 calls and 9,127 tokens. Iteration, token and timeout limits prevent uncontrolled execution.

Plan-and-Execute separates planning from execution and can provide a more fixed, auditable sequence. However, a preplanned sequence may need adjustment when an early tool result changes what should happen next. **A separate Plan-and-Execute implementation was not benchmarked in this project**, so no numerical performance comparison is claimed.

---

## 12. Generated Outputs

- `data/instamart.db` – SQLite database.
- `outputs/customer_rfm.csv` – RFM features.
- `outputs/customer_segments.csv` – customer segment assignments.
- `outputs/clustering_evaluation.csv` – K-Means evaluation.
- `outputs/lapse_features.csv` – lapse modelling dataset.
- `outputs/lapse_model.joblib` – saved prediction pipeline.
- `outputs/retention_list.csv` – illustrative retention shortlist.
- `outputs/triage_results.csv` – ticket triage results.
- `outputs/triage_budget_demo.csv` – budget demonstration.
- `outputs/triage_run_log.txt` – triage run log.
- `outputs/policy_eval_results.csv` – overlapping-chunk evaluation.
- `outputs/policy_eval_zero_overlap.csv` – zero-overlap evaluation.
- `outputs/agent_runs.jsonl` – refund-agent scenario logs.
- `outputs/refund_ledger.jsonl` – refund action ledger.
- `state/` – scenario and recovery checkpoints.

## 13. Limitations and Future Work

- The project uses synthetic assignment data and a mock LLM, not live Instamart customer systems.
- Model metrics depend on the generated dataset and the chosen lapse definition.
- Retention recommendations are illustrative; campaign impact requires a controlled experiment.
- Policy retrieval was evaluated on a small, fixed question set.
- The current answer helper takes retrieved chunks as an argument rather than performing retrieval internally.
- A separate Plan-and-Execute agent was not implemented or measured.
- Future work could test on representative real-world data, monitor model drift, add semantic retrieval, improve prospective evaluation and measure retention campaign outcomes through A/B testing.

## 14. Conclusion

This capstone demonstrates a complete workflow from raw-data quality checks to customer analytics, machine-learning predictions, support-ticket routing and policy-grounded refund assistance. The project includes validation, evaluation outputs and execution safeguards. Its results are useful as an educational prototype, while real-world deployment would require production data, security review, stronger evaluation and operational monitoring.