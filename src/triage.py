from pathlib import Path
import sys
import re
import json
import time
import random
import pandas as pd


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

import mock_llm

TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENTS_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05

CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# PII patterns
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")


# --------------------------------------------------
# 1. LOAD AND REDACT TICKETS
# --------------------------------------------------

def redact_text(text):
    """Redact email, phone number and order ID in that order."""

    text = str(text)

    counts = {
        "email": len(EMAIL_PATTERN.findall(text)),
        "phone": len(PHONE_PATTERN.findall(text)),
        "order_id": len(ORDER_PATTERN.findall(text)),
    }

    # Redaction order is important.
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    text = ORDER_PATTERN.sub("[ORDER_ID]", text)

    # Verify that no original PII patterns remain.
    assert not EMAIL_PATTERN.search(text)
    assert not PHONE_PATTERN.search(text)
    assert not ORDER_PATTERN.search(text)

    return text, counts


def prepare_tickets():

    # Preserve empty strings instead of converting them to NaN.
    tickets = pd.read_csv(
        TICKETS_FILE,
        keep_default_na=False
    )

    required = {
        "ticket_id",
        "customer_id",
        "order_id",
        "text",
        "true_category",
    }

    missing = required - set(tickets.columns)

    if missing:
        raise ValueError(f"Missing ticket columns: {missing}")

    redacted_texts = []

    total_counts = {
        "email": 0,
        "phone": 0,
        "order_id": 0
    }

    for value in tickets["text"]:

        if not str(value).strip():
            redacted_texts.append("")
            continue

        redacted, counts = redact_text(value)

        redacted_texts.append(redacted)

        for key in total_counts:
            total_counts[key] += counts[key]

    tickets["redacted_text"] = redacted_texts

    print("\nPII REDACTION COUNTS")
    print(total_counts)
    print("Total PII instances:", sum(total_counts.values()))

    return tickets


# --------------------------------------------------
# 2. PROMPTS
# --------------------------------------------------

ZERO_SHOT_SYSTEM = """

CLASSIFY the customer support ticket into exactly one category.

Allowed categories:

billing, delivery, technical, fraud.

Return JSON only, using exactly this structure:

{"category":"one_allowed_category","reason":"short reason"}

Do not return explanations outside the JSON.

"""


FEW_SHOT_SYSTEM = """

CLASSIFY the customer support ticket into exactly one category.

Allowed categories:

billing, delivery, technical, fraud.

EXAMPLES:

Ticket: I was charged twice for my order.

Output: {"category":"billing","reason":"duplicate charge"}

Ticket: My payment was made but the order was not placed.

Output: {"category":"billing","reason":"payment issue"}

Ticket: Someone accessed my account without permission.

Output: {"category":"fraud","reason":"unauthorized access"}

Ticket: The app crashes whenever I open it.

Output: {"category":"technical","reason":"app malfunction"}

Return JSON only, using exactly this structure:

{"category":"one_allowed_category","reason":"short reason"}

Do not return explanations outside the JSON.

"""


# --------------------------------------------------
# 3. OUTPUT VALIDATION
# --------------------------------------------------

def parse_response(content):
    """Validate JSON first, then attempt keyword fallback."""

    try:

        data = json.loads(content)

        if (
            isinstance(data, dict)
            and data.get("category") in CATEGORIES
            and isinstance(data.get("reason"), str)
        ):
            return data["category"], "classified_json"

        return None, "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):

        words = re.findall(
            r"[a-z]+",
            str(content).lower()
        )

        matches = [
            word for word in words
            if word in CATEGORIES
        ]

        if len(matches) == 1:
            return matches[0], "classified_keyword"

        return None, "flagged_invalid_output"


# --------------------------------------------------
# 4. SINGLE TICKET CLASSIFICATION WITH RETRIES
# --------------------------------------------------

def classify_ticket(
    text,
    system_prompt,
    ticket_id,
    rng
):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": text
        },
    ]

    # Verify that the model receives exactly the redacted text.
    assert messages[1]["content"] == text

    retry_number = 0

    while True:

        try:

            response = mock_llm.chat(messages)

            content = response.choices[0].message.content
            tokens = response.usage.total_tokens

            category, status = parse_response(content)

            return category, status, tokens

        except Exception as error:

            if error.__class__.__name__ != "TransientLLMError":
                raise

            if retry_number >= MAX_RETRIES:

                print(
                    f"Ticket {ticket_id}: retries exhausted."
                )

                return None, "flagged_invalid_output", 0

            # Exponential backoff with deterministic jitter.
            delay = (
                BASE_DELAY
                * (2 ** retry_number)
                * rng.uniform(0.9, 1.15)
            )

            # Print the current retry number before incrementing.
            print(
                f"Ticket {ticket_id}: retry "
                f"{retry_number}/{MAX_RETRIES}, "
                f"waiting {delay:.4f} seconds"
            )

            # Increment AFTER printing.
            retry_number += 1

            time.sleep(delay)


# --------------------------------------------------
# 5. FULL TRIAGE PIPELINE
# --------------------------------------------------

def run_pipeline(tickets, system_prompt, budget=None):

    # Reset mock LLM immediately before every full run.
    mock_llm.reset()

    # Fixed random seed for reproducible retry jitter.
    rng = random.Random(42)

    results = []
    running_tokens = 0

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]

        raw_text = ticket["text"]
        text = ticket["redacted_text"]

        category = None
        tokens = 0

        # Empty ticket guardrail.
        if not str(raw_text).strip():

            status = "skipped_empty"

        # Long ticket guardrail.
        elif mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:

            status = "flagged_long"

        # Budget check BEFORE model call.
        elif budget is not None and running_tokens >= budget:

            status = "budget_exhausted"

        else:

            category, status, tokens = classify_ticket(
                text,
                system_prompt,
                ticket_id,
                rng
            )

            running_tokens += tokens

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "tokens_used": tokens,
            "true_category": ticket["true_category"],
        })

    result_df = pd.DataFrame(results)

    return result_df, running_tokens


# --------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------

def evaluate_results(results, label):

    classified = results[
        results["status"].isin(
            [
                "classified_json",
                "classified_keyword"
            ]
        )
    ].copy()

    print(f"\n{'=' * 55}")
    print(label)
    print("=" * 55)

    print("Total tickets:", len(results))
    print("Successfully classified:", len(classified))

    print("\nStatus counts:")
    print(results["status"].value_counts())

    if len(classified) == 0:
        return

    accuracy = (
        classified["category"]
        == classified["true_category"]
    ).mean()

    total_tokens = results["tokens_used"].sum()

    tokens_per_ticket = (
        total_tokens / len(classified)
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print("Total tokens:", total_tokens)

    print(
        f"Tokens per classified ticket: {tokens_per_ticket:.2f}"
    )

    print("\nConfusion matrix:")

    print(
        pd.crosstab(
            classified["true_category"],
            classified["category"],
            rownames=["Actual"],
            colnames=["Predicted"],
        )
    )


# --------------------------------------------------
# 7. CUSTOMER SEGMENT ROUTING
# --------------------------------------------------

def apply_routing(results):

    segments = pd.read_csv(SEGMENTS_FILE)

    required = {
        "customer_id",
        "segment_name",
        "is_premium"
    }

    missing = required - set(segments.columns)

    if missing:
        raise ValueError(
            f"Missing customer segment columns: {missing}"
        )

    segment_lookup = segments.set_index(
        "customer_id"
    ).to_dict("index")

    final_rows = []

    for _, row in results.iterrows():

        customer_id = row["customer_id"]
        status = row["status"]
        category = row["category"]

        customer_segment = segment_lookup.get(
            customer_id,
            {}
        )

        segment_name = customer_segment.get(
            "segment_name",
            "Unknown"
        )

        is_premium = (
            int(customer_segment.get("is_premium", 0)) == 1
        )

        # Routing priority order.
        if status in [
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        ]:

            priority = "NORMAL"
            route = "human_review"

        elif category == "fraud":

            priority = "HIGH"
            route = "security_team"

        elif is_premium:

            priority = "HIGH"
            route = "premium_desk"

        else:

            priority = "NORMAL"
            route = "support_queue"

        final_rows.append({

            "ticket_id": row["ticket_id"],
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": row["tokens_used"],

        })

    # Required final output column order.
    return pd.DataFrame(
        final_rows,
        columns=[
            "ticket_id",
            "customer_id",
            "status",
            "category",
            "segment_name",
            "priority",
            "route",
            "tokens_used",
        ]
    )


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    tickets = prepare_tickets()

    # ----------------------------------------------
    # Zero-shot full run
    # ----------------------------------------------

    zero_results, zero_tokens = run_pipeline(
        tickets,
        ZERO_SHOT_SYSTEM
    )

    evaluate_results(
        zero_results,
        "ZERO-SHOT RESULTS"
    )

    # ----------------------------------------------
    # Few-shot full run
    # ----------------------------------------------

    few_results, few_tokens = run_pipeline(
        tickets,
        FEW_SHOT_SYSTEM
    )

    evaluate_results(
        few_results,
        "FEW-SHOT RESULTS"
    )

    # ----------------------------------------------
    # Budget demonstration
    # ----------------------------------------------

    budget_results, budget_tokens = run_pipeline(
        tickets,
        ZERO_SHOT_SYSTEM,
        budget=6000
    )

    if (
        "budget_exhausted"
        not in budget_results["status"].values
    ):

        print(
            "\n6000-token budget was not exhausted. "
            "Trying the lower 5000-token budget."
        )

        budget_results, budget_tokens = run_pipeline(
            tickets,
            ZERO_SHOT_SYSTEM,
            budget=5000
        )

    budget_results.to_csv(
        OUTPUT_DIR / "triage_budget_demo.csv",
        index=False
    )

    print("\nBudget demonstration saved.")

    # ----------------------------------------------
    # Final few-shot run with explicit 100000 budget
    # ----------------------------------------------

    final_results, final_tokens = run_pipeline(
        tickets,
        FEW_SHOT_SYSTEM,
        budget=100000
    )

    routed_results = apply_routing(final_results)

    routed_results.to_csv(
        OUTPUT_DIR / "triage_results.csv",
        index=False
    )

    print("\nFINAL TRIAGE RESULTS")

    print(
        routed_results["status"].value_counts()
    )

    print("\nRouting counts:")

    print(
        routed_results["route"].value_counts()
    )

    print("\nPriority counts:")

    print(
        routed_results["priority"].value_counts()
    )

    print("\nFinal tokens used:", final_tokens)

    print("\nSaved:")

    print(OUTPUT_DIR / "triage_results.csv")
    print(OUTPUT_DIR / "triage_budget_demo.csv")

    print("\nPART C TRIAGE PIPELINE COMPLETED!")


if __name__ == "__main__":
    main()
