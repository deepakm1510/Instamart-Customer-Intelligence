# Instamart Customer Intelligence & Support Copilot

**Capstone Project - IIT Patna**

## Overview

This project builds a customer intelligence and support copilot for an Instamart-style quick-commerce platform.

It combines:
- Raw data auditing and cleaning
- SQLite database creation and SQL analytics
- Customer segmentation using RFM and K-Means
- Customer lapse prediction using machine learning
- Support-ticket classification and triage
- Policy retrieval using TF-IDF and cosine similarity
- Autonomous refund-support agent using a ReAct workflow

## Reproduce

Run all commands from the project root directory.

### 1. Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Generate the project data

```powershell
python generate_data.py
```

### 4. Run Part A - Data Foundation

```powershell
python src/audit_raw_data.py
python src/data_pipeline.py
python src/validate_data.py
python src/run_sql_queries.py
python src/verify_sql_details.py
python src/pandas_analysis.py
```

### 5. Run Part B - Customer Intelligence

```powershell
python src/customer_models.py
```

### 6. Run Part C - Ticket Triage

```powershell
python src/triage.py
```

### 7. Run Part D - Policy Retrieval and Refund Agent

```powershell
python src/policy_rag.py
python src/refund_agent.py
```

## Part A

### Data Cleaning and Storage

- Audited the raw customer, order, and product data.
- Removed exact duplicate orders and retained the latest record for each order ID.
- Standardized customer city and gender values.
- Handled missing, invalid, and negative values using documented cleaning rules.
- Created a SQLite database containing customers, products, and orders tables.
- All validation checkpoints passed.

### Key Results

- Cleaned orders: **12,309**
- Completed orders: **11,110**
- Cancelled orders: **1,199**
- Completed-order revenue: **INR 2,998,668.80**
- Customers in the cleaned dataset: **1,200**
- Average completed-order amount: **INR 269.91**
- Customers with above-average completed spend: **550**

Pandas analysis independently confirmed the completed-order revenue of INR 2,998,668.80.

## Part B

### Customer Segmentation and Lapse Prediction

- Built RFM features for **1,175 customers**.
- Evaluated K-Means clustering for K values from 2 to 8.
- Selected **K = 4**, with a silhouette score of **0.526**.
- Identified four segments: Occasional, Premium, Loyal, and Dormant.
- Built and compared Logistic Regression and Random Forest lapse-prediction models.
- Selected Oversampled Logistic Regression based on the business objective of prioritizing lapse recall.

### Key Results

- Lapse cohort: **1,156 customers**
- Lapsed customers: **443 (38.32%)**
- Test recall: **85.39%**
- Test F1-score: **0.6972**
- Model reload prediction check: **Passed**

The model is intended to support customer-retention decisions, not to guarantee future customer behaviour.

### Retention Shortlist

- Generated a retention shortlist in `outputs/retention_list.csv`.
- The shortlist is an illustrative prioritization output for retention campaigns.
- The top 10 shortlisted customers included 2 customers who were actually lapsed.
- Customer segments are derived from the full analysis window, so the shortlist should not be interpreted as a leakage-free prospective evaluation.

## Part C

### Support Ticket Classification and Triage

- Implemented zero-shot and few-shot ticket classification.
- Added PII masking for email addresses, phone numbers, and order IDs.
- Added output validation, retry handling, deterministic jitter, and budget controls.
- Implemented routing to support, security, human-review, and premium queues.

### Key Results

- Zero-shot accuracy: **85.71%**
- Few-shot accuracy: **100%**
- Zero-shot tokens: **10,804**
- Few-shot tokens: **20,320**
- Few-shot classified tickets: **147**
- Few-shot tokens per classified ticket: **138.23**
- Detected PII occurrences: **278**
  - Emails: **55**
  - Phone numbers: **91**
  - Order IDs: **132**

The few-shot run improved classification accuracy on the evaluation data but used more tokens.

## Part D

### Policy Retrieval and Refund Agent

- Implemented policy retrieval using TF-IDF and cosine similarity.
- Built a ReAct refund-support agent using the supplied mock LLM.
- Added tool validation, refund limits, iteration limits, token limits, and timeout handling.
- Added checkpoint-based crash recovery to resume interrupted workflows.
- Recorded agent execution details in JSONL files and refund actions in a refund ledger.

### Policy Retrieval Evaluation

The retrieval system was evaluated using overlapping and zero-overlap text chunks.

| Evaluation Metric | Overlapping Chunks | Zero-Overlap Chunks |
|---|---:|---:|
| Expected document in top 3 | 12/12 | 12/12 |
| Expected fact in answer | 12/12 | 10/12 |
| Correct fallback | 2/2 | 2/2 |

The overlapping-chunk configuration retrieved all expected facts in the evaluation questions. With zero-overlap chunks, 2 of 12 expected facts were missed, demonstrating the benefit of retaining contextual overlap between chunks.

### Refund Agent Key Results

- Scenarios executed: **7**
- Total LLM calls: **30**
- Total tokens used: **9,127**
- Refund outcomes: **2**
- Escalation outcomes: **1**
- Crash-recovery demonstration: **Completed successfully**
- Token-limit and timeout fallback demonstrations: **Completed**

### ReAct vs Plan-and-Execute

The ReAct agent was selected because it alternates reasoning and tool execution. Each next action can depend on the previous tool observation. Across seven scenarios, the agent used 30 LLM calls and 9,127 tokens. It stopped early when further actions were unnecessary, such as when an order was cancelled or not found.

A Plan-and-Execute agent could provide a more predictable, auditable sequence, but may perform unnecessary steps if an early observation changes the plan. ReAct is more suitable for this workflow, with iteration, token, and timeout limits and checkpoint-based crash recovery.

## Generated Outputs

### Data and Database

- `data/instamart.db` - SQLite database containing the cleaned project tables.

### Part B - Customer Intelligence

- `outputs/customer_rfm.csv`
- `outputs/customer_segments.csv`
- `outputs/clustering_evaluation.csv`
- `outputs/lapse_features.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

### Part C - Ticket Triage

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_run_log.txt`

### Part D - Policy Retrieval and Refund Agent

- `outputs/policy_eval_results.csv`
- `outputs/policy_eval_zero_overlap.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`

### Agent Checkpoints

The `state/` directory contains saved scenario checkpoints and demonstrations for crash recovery, token limits, and timeout handling.

## Limitations and Next Steps

- Refund scenarios use the assignment's assumed 7-day request window because scenario timestamps are not available.
- The project uses the supplied mock LLM rather than a live production LLM.
- Model performance is based on the available synthetic dataset.
- Customer lapse predictions are not guaranteed outcomes.
- The retention shortlist is illustrative and is not a leakage-free prospective evaluation.
- Policy retrieval evaluation is based on the supplied evaluation questions and policy documents.
- Future improvements could include testing on real-world data, monitoring model drift, and evaluating additional retention strategies.