# Instamart Customer Intelligence & Support Copilot

**Capstone Project – IIT Patna**

## Project Overview

This project builds a customer intelligence and support copilot for an Instamart-style quick-commerce platform.

It combines:
- Raw data auditing and cleaning
- SQLite database creation and SQL analytics
- Customer segmentation using RFM and K-Means
- Customer lapse prediction using machine learning
- Support-ticket classification and triage
- Policy retrieval using TF-IDF and cosine similarity
- Autonomous refund-support agent using a ReAct workflow

## Technology Stack

- Python 3
- Pandas and NumPy
- Scikit-learn
- SQLite
- TF-IDF and cosine similarity
- Mock LLM (`mock_llm.py`)
- JSONL for agent logs and refund ledger

## Setup Instructions

### 1. Create Virtual Environment

```powershell
python -m venv .venv
```

### 2. Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```
## Run the Project

Run commands from the project root folder.

### Part A – Data Foundation

```powershell
python src/audit_raw_data.py
python src/data_pipeline.py
python src/validate_data.py
python src/run_sql_queries.py
python src/verify_sql_details.py
python src/pandas_analysis.py
```
### Part B – Customer Intelligence and Lapse Prediction

```powershell
python src/customer_models.py
```
### Part C – Support Ticket Classification and Triage

```powershell
python src/triage.py
```
### Part D – Policy Retrieval and Refund Agent

```powershell
python src/policy_rag.py
python src/refund_agent.py
```
## Key Results

### Part A – Data Foundation
- Cleaned 12,309 unique orders.
- Created SQLite database with customers, products, and orders tables.
- Validated completed-order revenue: INR 2,998,668.80.

### Part B – Customer Intelligence
- Identified 4 customer segments using K-Means.
- Selected Oversampled Logistic Regression for lapse prediction.
- Test recall: 85.39%.
- Test F1-score: 0.6972.

### Part C – Ticket Triage
- Classified support tickets using zero-shot and few-shot approaches.
- Applied PII masking, validation, retry handling, and budget controls.

### Part D – Policy and Refund Agent
- Implemented policy retrieval and a ReAct refund-support agent.
- Added checkpoint-based crash recovery.
- Recorded agent runs and refund actions in JSONL files.

## Assumptions and Limitations

- Refund scenarios use the assignment's assumed 7-day request window.
- The project uses a mock LLM rather than a live production LLM.
- Model performance is based on the available synthetic dataset.
- Predictions support business decisions and should not be treated as guaranteed outcomes.
## ReAct vs Plan-and-Execute

The ReAct agent was selected because it alternates reasoning and tool execution. This allows each next action to depend on the previous tool observation. Across seven scenarios, the agent used 30 LLM calls and 9,129 tokens. It stopped early when further actions were unnecessary, such as when an order was cancelled or not found.

A Plan-and-Execute agent could provide a more predictable, auditable sequence, but may perform unnecessary steps if an early observation changes the plan. ReAct is more suitable for this workflow, with iteration, token, and timeout limits and checkpoint-based crash recovery.