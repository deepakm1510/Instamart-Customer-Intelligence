
from pathlib import Path
import sys
import json
import sqlite3
import time
import uuid
import re
from datetime import datetime, timezone

import pandas as pd

# --------------------------------------------------
# 1. PROJECT PATHS AND IMPORTS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm

from policy_rag import load_policies, chunk_documents, PolicyRetriever


DB_PATH = ROOT / "data" / "instamart.db"
SCENARIO_FILE = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"

LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)


# All scenario requests are assumed to be within the
# 7-day request window, as specified in the assignment.
# The historical dataset has no request timestamp, so
# we do not compare it with the computer's current date.


# --------------------------------------------------
# 2. POLICY RETRIEVER
# --------------------------------------------------

documents = load_policies()

policy_chunks = chunk_documents(
    documents,
    chunk_size=40,
    overlap=10
)

POLICY_RETRIEVER = PolicyRetriever(policy_chunks)


# --------------------------------------------------
# 3. TOOL DEFINITIONS
# --------------------------------------------------

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get an order from the SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search the policy documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund based on delivery time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an approved refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Send a refund above INR 1000 for human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]


# --------------------------------------------------
# 4. TOOL IMPLEMENTATIONS
# --------------------------------------------------

def get_order(order_id):
    """Read order details from SQLite."""

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT
                order_id,
                status,
                order_amount,
                delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

    if row is None:
        return {
            "error": "Order not found",
            "order_id": order_id
        }

    return {
        "order_id": row["order_id"],
        "status": row["status"],
        "order_amount": float(row["order_amount"]),
        "delivery_minutes": (
            float(row["delivery_minutes"])
            if row["delivery_minutes"] is not None
            else None
        )
    }


def search_policy(query):
    """Retrieve the top three relevant policy chunks."""

    results = POLICY_RETRIEVER.retrieve(query, k=3)

    return {
        "results": results
    }


def calculate_refund(order_amount, delivery_minutes):
    """Apply the 45-minute late-delivery refund rule."""

    if delivery_minutes is None:
        refund = 0.0
    elif float(delivery_minutes) > 45:
        refund = float(order_amount)
    else:
        refund = 0.0

    return {
        "refund_inr": round(refund, 2)
    }


def write_ledger(tool_name, order_id, refund_inr):
    """Append a refund action to the JSONL ledger."""

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with LEDGER_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


def issue_refund(order_id, refund_inr):
    """Record an automatically approved refund."""

    write_ledger(
        "issue_refund",
        order_id,
        refund_inr
    )

    return {
        "refund_inr": round(float(refund_inr), 2)
    }


def escalate_to_human(order_id, refund_inr):
    """Record a refund requiring human approval."""

    write_ledger(
        "escalate_to_human",
        order_id,
        refund_inr
    )

    return {
        "refund_inr": round(float(refund_inr), 2),
        "message": "Human approval required"
    }


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# --------------------------------------------------
# 5. CHECKPOINT MANAGEMENT
# --------------------------------------------------

def checkpoint_path(run_id):
    return STATE_DIR / run_id


def save_checkpoint(state):
    """Save agent state after every tool observation."""

    path = checkpoint_path(state["run_id"])

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            state,
            file,
            indent=2,
            ensure_ascii=False
        )


def load_checkpoint(run_id):
    """Load a saved agent checkpoint."""

    path = checkpoint_path(run_id)

    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run_id: {run_id}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# 6. AGENT SYSTEM PROMPT
# --------------------------------------------------

AGENT_SYSTEM = """
You are an AGENT for Instamart customer refund support.

Use the available tools to investigate refund requests.
Follow the supplied refund policy.
Never invent order information or refund amounts.

Only completed orders can receive refunds.
Orders delivered in more than 45 minutes qualify for
a full refund of their order amount.
Orders delivered in 45 minutes or less receive no refund.
Refunds above INR 1000 must be escalated to a human lead.
"""


# --------------------------------------------------
# 7. HELPER FUNCTIONS
# --------------------------------------------------

def parse_tool_arguments(arguments):
    """Support both dictionary and JSON-string arguments."""

    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        return json.loads(arguments)

    return {}


def fallback_answer(reason, state):
    """Create a human handoff summary."""

    tools_used = [
        item["tool_name"]
        for item in state["tools_executed"]
    ]

    summary = (
        f"FALLBACK_{reason}: "
        f"Human handoff required. "
        f"Scenario: {state['scenario_id']}. "
        f"Customer request: {state['user_message']}. "
        f"Tools already executed: {tools_used}. "
        f"Please review the case and continue manually."
    )

    return {
        "status": f"FALLBACK_{reason}",
        "final_answer": summary,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "run_id": state["run_id"],
        "tools_executed": state["tools_executed"]
    }


def get_refund_amount(state):
    """Get the amount from the executed refund-related tool."""

    for item in reversed(state["tools_executed"]):
        if item["tool_name"] in (
            "issue_refund",
            "escalate_to_human"
        ):
            return float(item["observation"]["refund_inr"])

    return 0.0


# --------------------------------------------------
# 8. REACT AGENT LOOP
# --------------------------------------------------

def continue_agent(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tool_executions=None
):
    """Continue the ReAct loop, optionally from a checkpoint."""

    start_time = time.monotonic()

    while state["llm_calls"] < max_iterations:

        # TIME LIMIT CHECK BEFORE MODEL CALL
        elapsed = time.monotonic() - start_time

        if elapsed >= time_limit_seconds:
            result = fallback_answer("TIMEOUT", state)
            save_checkpoint(state)
            return result

        # CALL THE MOCK LLM
        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC
        )

        state["llm_calls"] += 1

        call_tokens = response.usage.total_tokens
        state["tokens_used"] += call_tokens

        # TOKEN LIMIT CHECK BEFORE EXECUTING TOOLS
        if state["tokens_used"] > token_limit:
            result = fallback_answer("TOKEN_LIMIT", state)
            save_checkpoint(state)
            return result

        # CHECK TIME AGAIN AFTER MODEL RESPONSE
        elapsed = time.monotonic() - start_time

        if elapsed > time_limit_seconds:
            result = fallback_answer("TIMEOUT", state)
            save_checkpoint(state)
            return result

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # NO TOOL CALL MEANS THE MODEL HAS FINISHED
        if not tool_calls:

            final_answer = message.content or ""

            status = (
                final_answer.split(":", 1)[0].strip()
                if ":" in final_answer
                else final_answer.strip()
            )

            save_checkpoint(state)

            return {
                "status": status,
                "final_answer": final_answer,
                "steps": state["steps"],
                "llm_calls": state["llm_calls"],
                "tokens_used": state["tokens_used"],
                "run_id": state["run_id"],
                "tools_executed": state["tools_executed"]
            }

        # SAVE ASSISTANT TOOL-CALL MESSAGE
        assistant_tool_calls = []

        for tool_call in tool_calls:
            assistant_tool_calls.append({
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": parse_tool_arguments(
                    tool_call.arguments
                )
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls
        })

        # EXECUTE EACH REQUESTED TOOL
        for tool_call in tool_calls:

            tool_name = tool_call.name

            if tool_name not in TOOL_MAP:
                observation = {
                    "error": f"Unknown tool: {tool_name}"
                }
                arguments = {}
            else:
                arguments = parse_tool_arguments(
                    tool_call.arguments
                )

                observation = TOOL_MAP[tool_name](
                    **arguments
                )

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            }

            state["messages"].append(tool_message)

            step = {
                "step": len(state["steps"]) + 1,
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation
            }

            state["steps"].append(step)

            state["tools_executed"].append({
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            # SAVE AFTER EVERY TOOL OBSERVATION
            save_checkpoint(state)

            # SIMULATED CRASH FOR RESUME DEMONSTRATION
            if (
                crash_after_tool_executions is not None
                and len(state["tools_executed"])
                >= crash_after_tool_executions
            ):
                raise RuntimeError(
                    f"Simulated crash after "
                    f"{len(state['tools_executed'])} tool executions. "
                    f"Run ID: {state['run_id']}"
                )

    # MAX ITERATIONS REACHED
    result = fallback_answer("MAX_ITERATIONS", state)
    save_checkpoint(state)
    return result


# --------------------------------------------------
# 9. RUN AGENT
# --------------------------------------------------

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tool_executions=None
):
    """Start a new ReAct agent run."""

    if run_id is None:
        run_id = uuid.uuid4().hex

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "user_message": user_message,
        "messages": [
            {
                "role": "system",
                "content": AGENT_SYSTEM
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "steps": [],
        "tools_executed": []
    }

    save_checkpoint(state)

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tool_executions=crash_after_tool_executions
    )


# --------------------------------------------------
# 10. RESUME AGENT
# --------------------------------------------------

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    """Resume from the last saved tool observation."""

    state = load_checkpoint(run_id)

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )


# --------------------------------------------------
# 11. SAVE SCENARIO RESULTS
# --------------------------------------------------

def save_agent_runs(results):
    """Write scenario results as JSON Lines."""

    with RUNS_FILE.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )


# --------------------------------------------------
# 12. MAIN: RUN ALL SEVEN SCENARIOS
# --------------------------------------------------

def main():

    scenarios = pd.read_csv(
        SCENARIO_FILE,
        dtype={"scenario_id": str, "customer_id": str}
    )

    results = []

    print("\nRUNNING SEVEN REFUND SCENARIOS")
    print("=" * 60)

    for _, row in scenarios.iterrows():

        mock_llm.reset()

        result = run_agent(
            user_message=row["message"],
            scenario_id=row["scenario_id"],
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10
        )

        result_record = {
            "scenario_id": row["scenario_id"],
            "customer_id": row["customer_id"],
            "expected_outcome": row["expected_outcome"],
            "expected_refund_inr": float(
                row["expected_refund_inr"]
            ),
            "status": result["status"],
            "refund_inr": get_refund_amount(
                {
                    "tools_executed": result["tools_executed"]
                }
            ),
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"],
            "final_answer": result["final_answer"],
            "run_id": result["run_id"]
        }

        results.append(result_record)

        print(
            row["scenario_id"],
            "| Status:", result["status"],
            "| Refund:", result_record["refund_inr"],
            "| Calls:", result["llm_calls"],
            "| Tokens:", result["tokens_used"]
        )

    save_agent_runs(results)

    print("\nScenario results saved:", RUNS_FILE)

    # --------------------------------------------------
    # 13. CHECKPOINT AND CRASH-RECOVERY DEMONSTRATION
    # --------------------------------------------------

    print("\nCRASH AND RESUME DEMONSTRATION")
    print("=" * 60)

    s1 = scenarios[
        scenarios["scenario_id"] == "S1"
    ].iloc[0]

    mock_llm.reset()

    demo_run_id = "crash_demo_" + uuid.uuid4().hex

    try:
        run_agent(
            user_message=s1["message"],
            scenario_id="S1_CRASH_DEMO",
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
            run_id=demo_run_id,
            crash_after_tool_executions=2
        )

    except RuntimeError as error:
        print("Simulated crash:", error)

    print("Resuming from saved checkpoint...")

    resumed = resume_agent(demo_run_id)

    print("Resumed status:", resumed["status"])

    print(
        "Tools executed:",
        [
            item["tool_name"]
            for item in resumed["tools_executed"]
        ]
    )

    print(
        "Total tool executions:",
        len(resumed["tools_executed"])
    )

    # --------------------------------------------------
    # 14. TOKEN LIMIT DEMONSTRATION
    # --------------------------------------------------

    print("\nTOKEN LIMIT DEMONSTRATION")
    print("=" * 60)

    mock_llm.reset()

    token_demo = run_agent(
        user_message=s1["message"],
        scenario_id="S1_TOKEN_LIMIT_DEMO",
        max_iterations=6,
        token_limit=200,
        time_limit_seconds=10
    )

    print("Token demo status:", token_demo["status"])
    print("Tokens used:", token_demo["tokens_used"])

    # --------------------------------------------------
    # 15. TIME LIMIT DEMONSTRATION
    # --------------------------------------------------

    print("\nTIME LIMIT DEMONSTRATION")
    print("=" * 60)

    mock_llm.reset()

    time_demo = run_agent(
        user_message=s1["message"],
        scenario_id="S1_TIMEOUT_DEMO",
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=0
    )

    print("Time demo status:", time_demo["status"])
    print("LLM calls:", time_demo["llm_calls"])

    # --------------------------------------------------
    # 16. FINAL SUMMARY
    # --------------------------------------------------

    print("\nPART D REFUND AGENT COMPLETED")
    print("=" * 60)

    print("Scenario results:", RUNS_FILE)
    print("Refund ledger:", LEDGER_FILE)
    print("Checkpoint folder:", STATE_DIR)


if __name__ == "__main__":
    main()
