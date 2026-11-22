"""Check the locked stack against the configured provider.

Uses about 6 chat calls and 2 embedding requests, plus retries on 503. Run with:
    uv run python scripts/smoke_test.py
"""

import os
import sys
import tempfile
import time
from collections.abc import Callable
from typing import Literal

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field


MAX_TRANSIENT_RETRIES = 2  # Spec section 13.2


class ClaimDraft(BaseModel):
    local_ref: str
    text: str
    kind: Literal["reported", "interpretation"]


class ListSchema(BaseModel):
    claims: list[ClaimDraft] = Field(default_factory=list)


class MatrixCell(BaseModel):
    paper_id: str
    claim_refs: list[str]


class MatrixRowList(BaseModel):
    dimension: str
    cells: list[MatrixCell]


class MatrixRowDict(BaseModel):
    dimension: str
    claims_by_paper: dict[str, list[str]]


@tool
def lookup_accuracy(paper_id: str) -> str:
    """Return the reported accuracy for a paper ID."""
    return {"A": "91.5%", "B": "88.0%"}.get(paper_id, "unknown")


def build_chat() -> ChatGoogleGenerativeAI:
    fields = ChatGoogleGenerativeAI.model_fields
    retry_key = "retries" if "retries" in fields else "max_retries"
    return ChatGoogleGenerativeAI(model=os.environ["LLM_MODEL"], **{retry_key: 1})


def check_chat(llm) -> str:
    reply = llm.invoke([HumanMessage("Reply with exactly: ready")])
    assert isinstance(reply, AIMessage) and "ready" in reply.text.lower()
    return "AIMessage returned"


def tool_history(llm) -> list:
    messages = [
        SystemMessage("Use lookup_accuracy for every requested paper."),
        HumanMessage("What accuracy do papers A and B report?"),
    ]
    request = llm.bind_tools([lookup_accuracy]).invoke(messages)
    assert request.tool_calls, "model requested no tool call"
    messages.append(request)
    for call in request.tool_calls:
        messages.append(
            ToolMessage(lookup_accuracy.invoke(call["args"]), tool_call_id=call["id"])
        )
    return messages


def check_tools(llm, messages: list) -> str:
    final = llm.bind_tools([lookup_accuracy]).invoke(messages)
    assert "91.5" in final.text and "88" in final.text
    calls = sum(isinstance(m, ToolMessage) for m in messages)
    return f"{calls} tool result(s) used in final answer"


def check_structured(llm) -> str:
    result = llm.with_structured_output(MatrixRowList).invoke(
        "Create a row for dimension 'method'. Use the exact paper_id values A and B. "
        "A has claim ref c1. B has claim ref c2."
    )
    assert isinstance(result, MatrixRowList)
    assert {c.paper_id for c in result.cells} == {"A", "B"}
    return "list-based schema parsed"


def check_dict_schema(llm) -> str:
    result = llm.with_structured_output(MatrixRowDict).invoke(
        "Create a row for dimension 'method'. In claims_by_paper, key A maps to [c1] and key B maps to [c2]."
    )
    assert isinstance(result, MatrixRowDict)
    assert result.claims_by_paper == {"A": ["c1"], "B": ["c2"]}
    return "dict-based schema parsed"


def check_structured_after_tools(llm, messages: list) -> str:
    prompt = HumanMessage("Record one reported claim for each paper's accuracy.")
    result = llm.with_structured_output(ListSchema).invoke([*messages, prompt])
    assert isinstance(result, ListSchema) and len(result.claims) >= 2
    return "separate structured call works after tool history"


def check_embeddings_and_chroma() -> str:
    embeddings = GoogleGenerativeAIEmbeddings(model=os.environ["EMBEDDING_MODEL"])
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as path:
        store = Chroma(
            collection_name="smoke",
            embedding_function=embeddings,
            persist_directory=path,
            client_settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        store.add_documents(
            [
                Document("Graph neural networks classify molecules.", metadata={"paper_id": "A"}),
                Document("Bread recipes use flour and yeast.", metadata={"paper_id": "B"}),
            ],
            ids=["a", "b"],
        )
        hits = store.similarity_search("molecule classification", k=1, filter={"paper_id": "B"})
        assert len(hits) == 1 and hits[0].metadata["paper_id"] == "B"
        del store
    return "filtered search returned only the requested paper"


def run(name: str, check: Callable[[], str], secret: str) -> bool:
    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        try:
            print(f"PASS  {name}: {check()}")
            return True
        except Exception as exc:  # Report every check, with the key removed.
            message = str(exc).replace(secret, "[redacted]")[:300]
            if "503" in message and attempt < MAX_TRANSIENT_RETRIES:
                time.sleep(5 * (attempt + 1))
                continue
            print(f"FAIL  {name}: {type(exc).__name__}: {message}")
            return False
    return False


def main() -> int:
    load_dotenv(".env")
    missing = [
        name
        for name in ("LLM_PROVIDER", "LLM_MODEL", "EMBEDDING_MODEL", "GOOGLE_API_KEY", "DATA_DIR")
        if not os.getenv(name)
    ]
    if missing:
        print(f"Missing settings: {', '.join(missing)}")
        return 2

    secret = os.environ["GOOGLE_API_KEY"]
    llm = build_chat()
    history: list = []

    def tools_check() -> str:
        history[:] = tool_history(llm)
        return check_tools(llm, history)

    def after_tools_check() -> str:
        if not history:
            raise RuntimeError("tool check did not produce a history")
        return check_structured_after_tools(llm, history)

    results = [
        run("chat", lambda: check_chat(llm), secret),
        run("tool calling", tools_check, secret),
        run("structured output (lists)", lambda: check_structured(llm), secret),
        run("structured output (dict)", lambda: check_dict_schema(llm), secret),
        run("structured output after tools", after_tools_check, secret),
        run("embeddings + Chroma filter", check_embeddings_and_chroma, secret),
    ]
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
