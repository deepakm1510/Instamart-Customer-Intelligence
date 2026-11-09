
from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm


POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

CHUNK_SIZE = 40
OVERLAP = 10
TOP_K = 3

FALLBACK_ANSWER = (
    "I could not find that information in the policy documents."
)

SYSTEM_PROMPT = f"""
You are a POLICY ASSISTANT.
Answer using ONLY the supplied policy context.
Do not use outside knowledge.
If the answer cannot be found in the context, respond exactly:
{FALLBACK_ANSWER}
"""


def load_policies():
    """Load all five policy text files."""
    documents = {}

    for path in sorted(POLICY_DIR.glob("*.txt")):
        documents[path.name] = path.read_text(encoding="utf-8")

    if len(documents) != 5:
        raise ValueError(
            f"Expected 5 policy files, found {len(documents)}"
        )

    return documents


def chunk_documents(
    documents,
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP
):
    """Split documents into word-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size"
        )

    chunks = []
    step = chunk_size - overlap

    for source, document in documents.items():
        words = document.split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            # Ignore only a trailing fragment no larger than overlap.
            if len(chunk_words) <= overlap:
                continue

            chunks.append({
                "source": source,
                "text": " ".join(chunk_words)
            })

    return chunks


class PolicyRetriever:
    """Retrieve policy chunks using TF-IDF and cosine similarity."""

    def __init__(self, chunks):
        if not chunks:
            raise ValueError("No policy chunks supplied")

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()

        texts = [chunk["text"] for chunk in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query, k=TOP_K):
        """Return the top-k chunks for a question."""
        if not query or not str(query).strip():
            return []

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        ranked_indices = scores.argsort()[::-1][:k]

        results = []

        for index in ranked_indices:
            results.append({
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"]
            })

        return results


def answer(question, retrieved_chunks):
    """Generate an answer using only the supplied retrieved context."""

    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[{item['source']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )

    response = mock_llm.chat([
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        }
    ])

    answer_text = response.choices[0].message.content.strip()

    return {
        "answer": answer_text,
        "sources": list(dict.fromkeys(
            item["source"] for item in retrieved_chunks
        ))
    }


def evaluate(retriever, evaluation_data):
    """Evaluate document retrieval, exact fact matching and fallback."""

    rows = []

    for _, row in evaluation_data.iterrows():
        question = str(row["question"])
        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = str(row["expected_fact"])

        # Retrieve only once so the answer and evaluation use
        # exactly the same retrieved context.
        retrieved = retriever.retrieve(
            question,
            k=TOP_K
        )

        result = answer(question, retrieved)
        answer_text = result["answer"]

        retrieved_docs = [
            item["source"] for item in retrieved
        ]

        if expected_doc == "NONE":
            doc_match = None
            fact_match = None

            fallback_match = (
                answer_text == FALLBACK_ANSWER
            )

        else:
            doc_match = expected_doc in retrieved_docs

            # Assignment-compliant case-sensitive substring check.
            fact_match = expected_fact in answer_text

            fallback_match = None

        rows.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "retrieved_docs": ", ".join(retrieved_docs),
            "answer": answer_text,
            "doc_match": doc_match,
            "fact_match": fact_match,
            "fallback_match": fallback_match
        })

    return pd.DataFrame(rows)


def print_evaluation(results, title):
    """Print evaluation metrics and failed questions."""

    print(f"\n{title}")
    print("=" * 60)

    answerable = results[
        results["expected_doc"] != "NONE"
    ].copy()

    unanswerable = results[
        results["expected_doc"] == "NONE"
    ].copy()

    doc_correct = int(
        answerable["doc_match"].fillna(False).sum()
    )

    fact_correct = int(
        answerable["fact_match"].fillna(False).sum()
    )

    fallback_correct = int(
        unanswerable["fallback_match"].fillna(False).sum()
    )

    print(
        "Expected document in top 3:",
        f"{doc_correct}/{len(answerable)}"
    )

    print(
        "Expected fact in answer:",
        f"{fact_correct}/{len(answerable)}"
    )

    print(
        "Correct fallback:",
        f"{fallback_correct}/{len(unanswerable)}"
    )

    failed_facts = answerable[
        answerable["fact_match"].eq(False)
    ]

    if not failed_facts.empty:
        print("\nFAILED FACT QUESTIONS")
        print("-" * 60)

        for _, row in failed_facts.iterrows():
            print("\nQuestion:", row["question"])
            print("Expected fact:", row["expected_fact"])
            print("Actual answer:", row["answer"])

    failed_docs = answerable[
        answerable["doc_match"].eq(False)
    ]

    if not failed_docs.empty:
        print("\nFAILED DOCUMENT RETRIEVAL QUESTIONS")
        print("-" * 60)

        for _, row in failed_docs.iterrows():
            print("\nQuestion:", row["question"])
            print("Expected document:", row["expected_doc"])
            print("Retrieved documents:", row["retrieved_docs"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_policies()

    print("Policy documents loaded:", len(documents))

    evaluation_data = pd.read_csv(EVAL_FILE)

    required_columns = {
        "question",
        "expected_doc",
        "expected_fact"
    }

    missing_columns = required_columns - set(
        evaluation_data.columns
    )

    if missing_columns:
        raise ValueError(
            f"Evaluation CSV is missing columns: {sorted(missing_columns)}"
        )

    # --------------------------------------------------
    # 1. OVERLAPPING CHUNK EVALUATION
    # --------------------------------------------------

    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

    print("\nOverlapping chunks:", len(chunks))

    chunk_counts = pd.Series(
        [chunk["source"] for chunk in chunks]
    ).value_counts().sort_index()

    print("\nChunks per document:")
    print(chunk_counts.to_string())

    retriever = PolicyRetriever(chunks)

    test_query = "refund policy late delivery"
    test_results = retriever.retrieve(
        test_query,
        k=TOP_K
    )

    print("\nTop 3 results for:", test_query)

    for result in test_results:
        print(
            result["source"],
            "| score:",
            round(result["score"], 4)
        )

    mock_llm.reset()

    results = evaluate(
        retriever,
        evaluation_data
    )

    print_evaluation(
        results,
        "OVERLAPPING CHUNK EVALUATION"
    )

    results.to_csv(
        OUTPUT_DIR / "policy_eval_results.csv",
        index=False
    )

    # --------------------------------------------------
    # 2. ZERO-OVERLAP CHUNK EVALUATION
    # --------------------------------------------------

    zero_chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=0
    )

    print("\nZero-overlap chunks:", len(zero_chunks))

    zero_retriever = PolicyRetriever(zero_chunks)

    mock_llm.reset()

    zero_results = evaluate(
        zero_retriever,
        evaluation_data
    )

    print_evaluation(
        zero_results,
        "ZERO-OVERLAP EVALUATION"
    )

    zero_results.to_csv(
        OUTPUT_DIR / "policy_eval_zero_overlap.csv",
        index=False
    )

    print("\nEvaluation files saved:")
    print(OUTPUT_DIR / "policy_eval_results.csv")
    print(OUTPUT_DIR / "policy_eval_zero_overlap.csv")

    print("\nPart D policy RAG evaluation completed.")


if __name__ == "__main__":
    main()
