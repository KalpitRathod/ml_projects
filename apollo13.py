#!/usr/bin/env python3


import argparse
import asyncio
import ast
import collections
import dataclasses
import difflib
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
EXECUTIVE_MODEL    = "llama3.2"
ENGINEER_MODEL     = "qwen3:4b"
EMBEDDING_MODEL    = "nomic-embed-text"

MAX_ENGINEER_RETRIES = 3
DOCKER_BASE_IMAGE    = "python:3.11-slim"
DOCKER_TIMEOUT       = 60
COVERAGE_THRESHOLD   = 80
MAX_CONTEXT_CHARS    = 8_000

# v5.0 – small-model tuning
VOTE_RUNS      = 1      # >1 enables self-consistency voting
STUB_FIRST     = False  # True enables two-phase stub→fill generation
ROLLBACK_DEPTH = 3
DIFF_EDIT_FROM = 2      # attempt index at which diff-edit is attempted first

# v5.2 – debate
DEBATE_ENGINEERS = 1    # >1 enables multi-agent debate mode

# Packages auto-approved without user confirmation
AUTO_APPROVE_PACKAGES = frozenset({
    "requests", "httpx", "aiohttp",
    "numpy", "scipy", "pandas",
    "pydantic", "attrs",
    "click", "rich", "typer",
    "python-dateutil", "pytz", "arrow",
    "more-itertools", "toolz", "sortedcontainers",
    "pytest", "pytest-cov",
})

# Stdlib modules generated code may import from
SAFE_IMPORT_ALLOWLIST = frozenset({
    "math", "json", "re", "datetime", "collections", "itertools",
    "functools", "typing", "types", "string", "random", "decimal",
    "fractions", "statistics", "heapq", "bisect", "copy", "enum",
    "dataclasses", "abc", "operator", "pathlib",
})

# v5.2 – language registry
LANGUAGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "python": {
        "extension":   ".py",
        "test_runner": ["pytest", "-q", "test_generated.py"],
        "src_subdir":  "src",
        "comment":     "#",
    },
    "typescript": {
        "extension":   ".ts",
        "test_runner": ["npx", "jest", "--passWithNoTests"],
        "src_subdir":  "src",
        "comment":     "//",
    },
    "go": {
        "extension":   ".go",
        "test_runner": ["go", "test", "./..."],
        "src_subdir":  ".",
        "comment":     "//",
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("apollo13")


# ──────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ──────────────────────────────────────────────────────────────────────────────
class FileSpec(BaseModel):
    path: str = Field(description="Relative path from project root")
    description: str = Field(description="One-sentence purpose of this file")
    dependencies: List[str] = Field(default_factory=list)
    is_test: bool = Field(default=False)
    test_content: Optional[str] = Field(default=None)
    language: str = Field(default="python", description="python | typescript | go")
    target_lines: int = Field(
        default=100,
        description="Suggested max lines — keeps files micro-sized for small models",
    )


class ProjectManifest(BaseModel):
    task_summary: str = Field(description="One-sentence description of the project")
    files: List[FileSpec] = Field(description="All source and test files to create")
    external_packages: List[str] = Field(default_factory=list)
    test_root: str = Field(default="tests")
    milestones: List[str] = Field(
        default_factory=list,
        description="Ordered list of milestone descriptions for human checkpoints",
    )


class AuditReview(BaseModel):
    is_approved: bool
    feedback: str
    security_score: int = Field(default=5, ge=0, le=10)


class CorrectedFile(BaseModel):
    path: str
    content: str
    change_summary: str


class IntegrationResult(BaseModel):
    corrections: List[CorrectedFile] = Field(default_factory=list)
    overall_notes: str


# ── v5.2 models ──────────────────────────────────────────────────────────────
class ChangePlan(BaseModel):
    """Targeted modification plan for --modify mode."""
    target_file: str
    reason: str
    unified_diff: str = Field(description="Unified diff to apply")


class DocPage(BaseModel):
    filename: str = Field(description="e.g. 'api_reference.md'")
    content: str


class CommitInfo(BaseModel):
    subject: str = Field(description="Conventional commit subject line ≤72 chars")
    body: str = Field(description="Full PR description in Markdown")


# ── v6.0 models ──────────────────────────────────────────────────────────────
class TaskNode(BaseModel):
    """Single atomic task in the v6.0 workflow engine."""
    id: str
    description: str
    file_spec: Optional[FileSpec] = None
    depends_on: List[str] = Field(default_factory=list)
    milestone: Optional[str] = None
    status: str = Field(default="pending")  # pending | running | done | failed


class Epic(BaseModel):
    title: str
    tasks: List[TaskNode]


class ProjectGoal(BaseModel):
    """Top-level decomposition produced by the executive in --autonomous mode."""
    summary: str
    epics: List[Epic]
    manifest: ProjectManifest


# ──────────────────────────────────────────────────────────────────────────────
# PROJECT STATE
# ──────────────────────────────────────────────────────────────────────────────
class ProjectState:
    """Persists file hashes and test results across runs."""

    def __init__(self, state_path: Path) -> None:
        self.path = state_path
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"files": {}, "manifest": None, "packages": [], "session": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def save_manifest(self, manifest: ProjectManifest) -> None:
        self.data["manifest"] = manifest.model_dump()
        self._save()

    def set_packages(self, packages: List[str]) -> None:
        self.data["packages"] = packages
        self._save()

    def save_session(self, session: Dict[str, Any]) -> None:
        """v6.0: persist workflow session so runs can be resumed."""
        self.data["session"] = session
        self._save()

    def load_session(self) -> Dict[str, Any]:
        return self.data.get("session", {})

    def mark_file(
        self,
        rel_path: str,
        content: str,
        test_passed: bool,
        audit_approved: bool,
        security_score: int = 5,
    ) -> None:
        self.data["files"][rel_path] = {
            "hash":           _sha256(content),
            "test_passed":    test_passed,
            "audit_approved": audit_approved,
            "security_score": security_score,
            "timestamp":      time.time(),
        }
        self._save()

    def is_fresh(self, rel_path: str, content: str) -> bool:
        rec = self.data["files"].get(rel_path)
        return bool(rec and rec["hash"] == _sha256(content) and rec["test_passed"])

    def pending_files(self, manifest: ProjectManifest, src_dir: str) -> List[FileSpec]:
        pending = []
        for spec in manifest.files:
            if spec.is_test:
                continue
            full = Path(src_dir) / spec.path
            if not full.exists():
                pending.append(spec)
                continue
            if not self.is_fresh(spec.path, full.read_text(encoding="utf-8")):
                pending.append(spec)
        return pending


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def extract_python_code(text: str) -> str:
    """Strip <think>/<thinking> blocks and return the first ```python``` block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks)
    parts = re.split(r"```python", text, flags=re.IGNORECASE)
    return parts[-1].strip() if len(parts) > 1 else text.strip()


def extract_code_block(text: str, lang: str = "python") -> str:
    """Generic version for multi-language support."""
    if lang == "python":
        return extract_python_code(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    pattern = rf"```{re.escape(lang)}(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    if blocks:
        return "\n\n".join(b.strip() for b in blocks)
    # Fallback: any fenced block
    blocks = re.findall(r"```[a-z]*(.*?)```", text, re.DOTALL)
    return blocks[0].strip() if blocks else text.strip()


def is_safe_code_ast(
    code: str, extra_allowed: frozenset = frozenset()
) -> Tuple[bool, str]:
    allowed = SAFE_IMPORT_ALLOWLIST | extra_allowed
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed:
                    return False, f"Disallowed import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in allowed:
                return False, f"Disallowed import from: {node.module}"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in {
                "exec", "eval", "__import__", "compile", "open", "breakpoint",
            }:
                return False, f"Dangerous built-in: {func.id}()"
            if isinstance(func, ast.Attribute) and func.attr in {
                "system", "popen", "run", "call", "Popen",
            }:
                return False, f"Dangerous call: .{func.attr}()"
    return True, "ok"


def extract_api_surface(code: str, max_chars: int = 2_000) -> str:
    """Return public signatures + one-line docstrings only."""
    try:
        tree = ast.parse(code)
        src_lines = code.splitlines()
        output: List[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name.startswith("_"):
                continue
            for dec in node.decorator_list:
                output.append(src_lines[dec.lineno - 1])
            output.append(src_lines[node.lineno - 1])
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            ):
                ds = str(node.body[0].value.s)[:150].replace("\n", " ")
                output.append(f'    """{ds}"""')
            output.append("    ...")
        result = "\n".join(output)
        return result[:max_chars] if result else code[:max_chars]
    except Exception:
        return code[:max_chars]


def extract_failure_summary(output: str) -> str:
    lines = []
    for line in output.splitlines():
        if any(m in line for m in ("E   ", "AssertionError", "FAILED", "Error", "Exception")):
            clean = line.strip()
            if clean and not clean.startswith("="):
                lines.append(clean)
    return "\n".join(lines[:20])


def score_result(
    passed: bool,
    audit_ok: bool,
    code: str,
    runtime: float,
    attempts: int,
    security_score: int = 5,
) -> float:
    score = 0.0
    if passed:   score += 100
    if audit_ok: score += 50
    score += security_score * 5
    score -= len(code) * 0.02
    score -= runtime * 2
    score += max(0, 5 - attempts) * 5
    return score


def safe_target_path(base_dir: str, rel_path: str) -> Optional[Path]:
    base = Path(base_dir).resolve()
    candidate = (base / rel_path).resolve()
    if not str(candidate).startswith(str(base)):
        logger.error(f"Path traversal blocked: {rel_path!r}")
        return None
    return candidate


def show_unified_diff(old: str, new: str, filename: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=3,
        )
    )


def dependency_waves(files: List[FileSpec]) -> List[List[FileSpec]]:
    remaining = {f.path: f for f in files if not f.is_test}
    waves: List[List[FileSpec]] = []
    resolved: Set[str] = set()
    while remaining:
        wave = [
            f for f in remaining.values()
            if all(dep in resolved or dep not in remaining for dep in f.dependencies)
        ]
        if not wave:
            cycle_paths = ", ".join(remaining.keys())
            logger.warning(
                f"Circular dependency detected among: [{cycle_paths}]. "
                "Grouping into one wave to break the cycle."
            )
            wave = list(remaining.values())
        for f in wave:
            remaining.pop(f.path)
            resolved.add(f.path)
        waves.append(wave)
    return waves


# ── v5.0 utilities ────────────────────────────────────────────────────────────
def apply_unified_diff(original: str, patch_text: str) -> Optional[str]:
    """Apply a unified diff patch; returns patched text or None on failure."""
    if not patch_text.strip():
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            orig_path  = Path(tmp) / "original.py"
            patch_path = Path(tmp) / "changes.patch"
            orig_path.write_text(original, encoding="utf-8")
            patch_path.write_text(patch_text, encoding="utf-8")
            result = subprocess.run(
                ["patch", "--quiet", str(orig_path), str(patch_path)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                logger.debug(f"patch failed: {result.stderr.strip()[:200]}")
                return None
            return orig_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.debug(f"apply_unified_diff error: {exc}")
        return None


def vote_on_candidates(candidates: List[str]) -> str:
    """Jaccard-token majority vote; returns single best candidate."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else ""

    def _tokens(s: str) -> collections.Counter:
        return collections.Counter(re.findall(r"\w+", s))

    def _jaccard(a: collections.Counter, b: collections.Counter) -> float:
        inter = sum((a & b).values())
        union = sum((a | b).values())
        return inter / union if union else 0.0

    token_sets = [_tokens(c) for c in candidates]
    scores = [
        sum(_jaccard(token_sets[i], token_sets[j]) for j in range(len(token_sets)) if j != i)
        / max(1, len(token_sets) - 1)
        for i in range(len(token_sets))
    ]
    return candidates[scores.index(max(scores))]


def extract_stub_only(code: str) -> str:
    """Reduce implementation to stubs (signatures + docstrings + ...)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines()
    output: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for dec in node.decorator_list:
            output.append(lines[dec.lineno - 1])
        output.append(lines[node.lineno - 1])
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            ds = str(node.body[0].value.s).replace("\n", " ")[:200]
            output.append(f'    """{ds}"""')
        output.append("    ...")
    return "\n".join(output) if output else code


# ──────────────────────────────────────────────────────────────────────────────
# WORKSPACE
# ──────────────────────────────────────────────────────────────────────────────
def setup_workspace() -> Tuple[str, str, str]:
    base   = os.getcwd()
    src    = os.path.join(base, "src")
    apollo = os.path.join(base, ".apollo13")
    os.makedirs(src,    exist_ok=True)
    os.makedirs(apollo, exist_ok=True)
    return base, src, apollo


# ──────────────────────────────────────────────────────────────────────────────
# VECTOR MEMORY  (base codebase context)
# ──────────────────────────────────────────────────────────────────────────────
def _file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


async def load_context(src_dir: str, vs_dir: str, query: str) -> str:
    vs = Chroma(
        collection_name="apollo13",
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=vs_dir,
    )
    try:
        existing = set(vs.get()["ids"])
    except Exception:
        existing = set()

    docs_to_add: List[Document] = []
    ids_to_add:  List[str]      = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            full   = os.path.join(root, f)
            rel    = os.path.relpath(full, src_dir)
            doc_id = f"{rel}::{_file_hash(full)}"
            if doc_id in existing:
                continue
            docs_to_add.append(Document(
                page_content=Path(full).read_text(encoding="utf-8"),
                metadata={"source": rel},
            ))
            ids_to_add.append(doc_id)

    if docs_to_add:
        logger.info(f"Embedding {len(docs_to_add)} new/changed file(s).")
        await asyncio.to_thread(vs.add_documents, docs_to_add, ids=ids_to_add)

    results = await asyncio.to_thread(vs.similarity_search, query, k=3)
    return "\n\n".join(
        f"// {r.metadata.get('source', '?')}\n{r.page_content}" for r in results
    )


# ──────────────────────────────────────────────────────────────────────────────
# SUCCESS MEMORY  (v5.0)
# ──────────────────────────────────────────────────────────────────────────────
class SuccessMemory:
    """
    Persists every generation that passed tests + audit.
    Retrieves the most-similar past success as a few-shot prompt example.
    """
    COLLECTION = "apollo13_successes"

    def __init__(self, persist_dir: str) -> None:
        self._vs = Chroma(
            collection_name=self.COLLECTION,
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=persist_dir,
        )

    async def add(self, file_spec: FileSpec, code: str, task_summary: str) -> None:
        doc_id = _sha256(f"{task_summary}::{file_spec.path}::{code}")
        doc = Document(
            page_content=code,
            metadata={
                "task":        task_summary,
                "path":        file_spec.path,
                "description": file_spec.description,
                "language":    file_spec.language,
            },
        )
        try:
            await asyncio.to_thread(self._vs.add_documents, [doc], ids=[doc_id])
        except Exception as exc:
            logger.debug(f"SuccessMemory.add non-fatal: {exc}")

    async def get_example(self, query: str, language: str = "python") -> Optional[str]:
        try:
            results = await asyncio.to_thread(self._vs.similarity_search, query, k=3)
            # Prefer same-language results
            same_lang = [r for r in results if r.metadata.get("language") == language]
            r = same_lang[0] if same_lang else (results[0] if results else None)
            if r is None:
                return None
            m = r.metadata
            return (
                f"# Past success — {m.get('path', '?')}\n"
                f"# Task: {m.get('task', '?')}\n"
                f"# Purpose: {m.get('description', '?')}\n"
                f"{r.page_content}"
            )
        except Exception as exc:
            logger.debug(f"SuccessMemory.get_example non-fatal: {exc}")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# FAILURE MEMORY  (v5.0)
# ──────────────────────────────────────────────────────────────────────────────
class FailureMemory:
    """
    Stores (error_pattern → fix_hint) pairs.
    Surfaces similar past fixes before each retry.
    """
    COLLECTION = "apollo13_failures"

    def __init__(self, persist_dir: str) -> None:
        self._vs = Chroma(
            collection_name=self.COLLECTION,
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=persist_dir,
        )

    async def add(self, error_summary: str, fix_hint: str) -> None:
        if not error_summary.strip():
            return
        doc_id = _sha256(error_summary)
        doc = Document(
            page_content=f"ERROR:\n{error_summary}\n\nFIX HINT:\n{fix_hint}",
            metadata={"error": error_summary[:200]},
        )
        try:
            await asyncio.to_thread(self._vs.add_documents, [doc], ids=[doc_id])
        except Exception as exc:
            logger.debug(f"FailureMemory.add non-fatal: {exc}")

    async def get_hints(self, error_summary: str, k: int = 2) -> str:
        if not error_summary.strip():
            return ""
        try:
            results = await asyncio.to_thread(
                self._vs.similarity_search, error_summary, k=k
            )
            if not results:
                return ""
            return "Similar past failures and their fixes:\n" + "\n\n".join(
                r.page_content for r in results
            )
        except Exception as exc:
            logger.debug(f"FailureMemory.get_hints non-fatal: {exc}")
            return ""


# ──────────────────────────────────────────────────────────────────────────────
# ERROR CLUSTER  (v5.1)
# ──────────────────────────────────────────────────────────────────────────────
class ErrorCluster:
    """
    Lightweight in-memory clustering of failure patterns.
    Groups errors by their leading token (ImportError, AssertionError, etc.)
    and returns the most common fix recipe for each cluster.
    """

    def __init__(self) -> None:
        # cluster_key → {error_text → fix_hint}
        self._clusters: Dict[str, Dict[str, str]] = collections.defaultdict(dict)

    def _key(self, error_summary: str) -> str:
        first_line = error_summary.strip().splitlines()[0] if error_summary.strip() else ""
        match = re.match(r"([A-Za-z]+Error|[A-Za-z]+Exception|FAILED)", first_line)
        return match.group(1) if match else "Other"

    def record(self, error_summary: str, fix_hint: str) -> None:
        if error_summary.strip():
            self._clusters[self._key(error_summary)][error_summary] = fix_hint

    def recipe_for(self, error_summary: str) -> Optional[str]:
        """Return the most recently stored fix for this error cluster, or None."""
        key = self._key(error_summary)
        bucket = self._clusters.get(key)
        if not bucket:
            return None
        # Return the last recorded fix (most recent)
        return list(bucket.values())[-1]

    def summary(self) -> str:
        lines = [f"  {k}: {len(v)} recorded fix(es)" for k, v in self._clusters.items()]
        return "\n".join(lines) if lines else "  (no errors recorded)"


# ──────────────────────────────────────────────────────────────────────────────
# STYLE GUIDE  (v5.1)
# ──────────────────────────────────────────────────────────────────────────────
class StyleGuide:
    """
    Extracts naming conventions and docstring patterns from existing source files
    and injects them as a style section in the engineer prompt.
    """

    def __init__(self, src_dir: str) -> None:
        self.src_dir = Path(src_dir)
        self._guide: Optional[str] = None

    def build(self) -> str:
        if self._guide is not None:
            return self._guide

        py_files = list(self.src_dir.rglob("*.py"))
        if not py_files:
            self._guide = ""
            return ""

        naming_samples: List[str] = []
        docstring_samples: List[str] = []

        for path in py_files[:10]:  # cap at 10 files to stay token-lean
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    naming_samples.append(node.name)
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                    ):
                        ds = str(node.body[0].value.s).strip().splitlines()[0][:80]
                        docstring_samples.append(ds)

        if not naming_samples:
            self._guide = ""
            return ""

        # Detect naming style
        snake = sum(1 for n in naming_samples if "_" in n and n == n.lower())
        camel = sum(1 for n in naming_samples if n[0].islower() and any(c.isupper() for c in n))
        style = "snake_case" if snake >= camel else "camelCase"

        doc_example = docstring_samples[0] if docstring_samples else "Returns the result."

        self._guide = (
            f"Project style guide (extracted from existing source):\n"
            f"- Function naming: {style}\n"
            f"- Docstring style: one-line imperative, e.g. '{doc_example}'\n"
            f"- Keep functions focused; target ≤50 lines each.\n"
        )
        return self._guide


# ──────────────────────────────────────────────────────────────────────────────
# USER FEEDBACK STORE  (v5.1)
# ──────────────────────────────────────────────────────────────────────────────
class UserFeedbackStore:
    """
    Persists rejection reasons from interactive diff reviews.
    Retrieved as additional constraints in future similar-file prompts.
    """
    COLLECTION = "apollo13_feedback"

    def __init__(self, persist_dir: str) -> None:
        self._vs = Chroma(
            collection_name=self.COLLECTION,
            embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=persist_dir,
        )

    async def record(self, file_description: str, rejection_reason: str) -> None:
        if not rejection_reason.strip():
            return
        doc_id = _sha256(f"{file_description}::{rejection_reason}::{time.time()}")
        doc = Document(
            page_content=f"File: {file_description}\nRejection: {rejection_reason}",
            metadata={"description": file_description},
        )
        try:
            await asyncio.to_thread(self._vs.add_documents, [doc], ids=[doc_id])
        except Exception as exc:
            logger.debug(f"UserFeedbackStore.record non-fatal: {exc}")

    async def get_constraints(self, query: str) -> str:
        try:
            results = await asyncio.to_thread(self._vs.similarity_search, query, k=2)
            if not results:
                return ""
            return "User previously rejected similar code for these reasons:\n" + "\n".join(
                f"- {r.page_content}" for r in results
            )
        except Exception as exc:
            logger.debug(f"UserFeedbackStore.get_constraints non-fatal: {exc}")
            return ""


# ──────────────────────────────────────────────────────────────────────────────
# REGRESSION SUITE  (v5.1)
# ──────────────────────────────────────────────────────────────────────────────
class RegressionSuite:
    """
    Appends every passing test file to tests/regression/ as a permanent fixture.
    On future runs these are re-executed as part of the project security sweep,
    ensuring previously correct behaviour is never silently broken.
    """

    def __init__(self, base_dir: str) -> None:
        self.regression_dir = Path(base_dir) / "tests" / "regression"
        self.regression_dir.mkdir(parents=True, exist_ok=True)

    def add(self, file_spec: FileSpec, test_content: str) -> None:
        if not test_content.strip():
            return
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", file_spec.path)
        dest = self.regression_dir / f"test_regression_{safe_name}.py"
        header = (
            f"# Regression test — auto-generated from: {file_spec.path}\n"
            f"# Do not edit manually.\n\n"
        )
        dest.write_text(header + test_content, encoding="utf-8")
        logger.info(f"Regression test saved: {dest.name}")


# ──────────────────────────────────────────────────────────────────────────────
# DOCKER
# ──────────────────────────────────────────────────────────────────────────────
async def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def cleanup_containers() -> None:
    ret, out, _ = await run_cmd(["docker", "ps", "-aq", "--filter", "label=apollo13"])
    if ret == 0 and out.strip():
        await run_cmd(["docker", "rm", "-f"] + out.strip().split())


async def run_docker_cmd(
    workspace: Path,
    image: str,
    cmd: List[str],
    timeout: int = DOCKER_TIMEOUT,
    writable: bool = False,
) -> Tuple[int, str, str]:
    container_name = f"apollo13_{uuid.uuid4().hex[:8]}"
    mount_flag = f"{workspace.absolute()}:/app" + ("" if writable else ":ro")
    docker_cmd = [
        "docker", "run", "--rm",
        "--label",        "apollo13",
        "--name",         container_name,
        "--network",      "none",
        "--read-only",
        "--tmpfs",        "/tmp:noexec,nosuid,size=64m",
        "--pids-limit",   "128",
        "--memory",       "512m",
        "--cpus",         "1.0",
        "--security-opt", "no-new-privileges",
        "-u",             "1000:1000",
        "-v",             mount_flag,
        "-w",             "/app",
        image,
    ] + cmd

    timed_out = False
    try:
        code, out, err = await asyncio.wait_for(run_cmd(docker_cmd), timeout=timeout)
        return code, out, err
    except asyncio.TimeoutError:
        timed_out = True
        return -1, "", "Docker command timed out."
    finally:
        if timed_out:
            def _log(t: "asyncio.Task[Any]") -> None:
                if not t.cancelled() and t.exception():
                    logger.warning(f"Container removal failed ({container_name}): {t.exception()}")
            fut = asyncio.ensure_future(run_cmd(["docker", "rm", "-f", container_name]))
            fut.add_done_callback(_log)


async def prepare_sandbox(packages: List[str]) -> str:
    base_tools = "pytest pytest-cov bandit ruff safety"
    pkg_str    = " ".join(sorted(packages))
    cache_key  = hashlib.sha256(f"{base_tools}|{pkg_str}".encode()).hexdigest()[:12]
    image_tag  = f"apollo13-sandbox:{cache_key}"

    ret, _, _ = await run_cmd(["docker", "image", "inspect", image_tag])
    if ret == 0:
        logger.info(f"Using cached sandbox image: {image_tag}")
        return image_tag

    logger.info(f"Building sandbox image {image_tag}  packages=[{pkg_str or 'none'}]")
    all_pkgs   = f"{base_tools} {pkg_str}".strip()
    dockerfile = (
        f"FROM {DOCKER_BASE_IMAGE}\n"
        f"RUN pip install --no-cache-dir {all_pkgs}\n"
        "RUN useradd -u 1000 -m apollo\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "Dockerfile").write_text(dockerfile)
        ret, out, err = await run_cmd(["docker", "build", "-t", image_tag, tmp])
        if ret != 0:
            logger.error(f"Sandbox build failed:\n{err}")
            raise RuntimeError(f"Docker build failed for {image_tag}")
    return image_tag


async def run_tests_in_docker(
    workspace: Path, image: str, test_cmd: Optional[List[str]] = None
) -> Tuple[bool, str, float]:
    start = time.time()
    cmd = test_cmd or ["pytest", "-q", "test_generated.py"]
    code, out, err = await run_docker_cmd(
        workspace, image, cmd, timeout=DOCKER_TIMEOUT, writable=True,
    )
    return code == 0, out + "\n" + err, time.time() - start


async def run_tests_local(
    workspace: Path, test_cmd: Optional[List[str]] = None
) -> Tuple[bool, str, float]:
    start = time.time()
    cmd = test_cmd or [sys.executable, "-m", "pytest", "-q", "test_generated.py"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=DOCKER_TIMEOUT)
        return (
            proc.returncode == 0,
            out.decode(errors="replace") + "\n" + err.decode(errors="replace"),
            time.time() - start,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return False, "Local test execution timed out.", DOCKER_TIMEOUT


# ──────────────────────────────────────────────────────────────────────────────
# SECURITY PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
async def run_security_pipeline(
    workspace: Path,
    image: str,
    src_subdir: str = "src",
) -> Tuple[bool, Dict[str, str]]:
    reports: Dict[str, str] = {}
    issues = 0

    _, out, err = await run_docker_cmd(
        workspace, image, ["bandit", "-r", src_subdir, "-ll", "-f", "text"], timeout=30,
    )
    reports["bandit"] = (out + err).strip()[:2_000]
    if re.search(r"Severity: (High|Medium)", out):
        issues += 1

    code, out, err = await run_docker_cmd(
        workspace, image, ["ruff", "check", src_subdir, "--output-format=text"], timeout=20,
    )
    reports["ruff"] = (out + err).strip()[:2_000]
    if code != 0:
        issues += 1

    _, out, err = await run_docker_cmd(
        workspace, image, ["safety", "check", "--bare"], timeout=20,
    )
    reports["safety"] = (out + err).strip()[:1_000]

    code, out, err = await run_docker_cmd(
        workspace, image,
        ["pytest", f"--cov={src_subdir}", "--cov-report=term-missing", "-q"],
        timeout=DOCKER_TIMEOUT, writable=True,
    )
    reports["pytest_cov"] = (out + err).strip()[-3_000:]
    cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", out)
    coverage  = int(cov_match.group(1)) if cov_match else 0
    reports["coverage_pct"] = str(coverage)
    if code != 0:
        issues += 1
    if coverage < COVERAGE_THRESHOLD:
        logger.warning(f"Coverage {coverage}% below threshold {COVERAGE_THRESHOLD}%")

    return issues == 0, reports


# ──────────────────────────────────────────────────────────────────────────────
# DEPENDENCY RESOLVER
# ──────────────────────────────────────────────────────────────────────────────
async def resolve_dependencies(
    proposed: List[str], interactive: bool = False,
) -> List[str]:
    approved: List[str] = []
    for pkg in proposed:
        root = re.split(r"[><=\[!]", pkg)[0].strip()
        if root in AUTO_APPROVE_PACKAGES:
            logger.info(f"  [auto-approve] {root}")
            approved.append(root)
        elif interactive:
            answer = await asyncio.to_thread(input, f"  Allow package '{root}'? [y/N] ")
            if answer.strip().lower() == "y":
                approved.append(root)
                logger.info(f"  [user-approved] {root}")
            else:
                logger.warning(f"  [rejected] {root}")
        else:
            logger.warning(
                f"  [rejected] '{root}' is not auto-approved. "
                "Re-run with --interactive to approve it."
            )
    return approved


# ──────────────────────────────────────────────────────────────────────────────
# EXECUTIVE
# ──────────────────────────────────────────────────────────────────────────────
async def executive_plan_project(prompt: str, context: str) -> ProjectManifest:
    """Produce a ProjectManifest with micro-files and milestone markers."""
    llm        = ChatOllama(model=EXECUTIVE_MODEL, temperature=0.0)
    structured = llm.with_structured_output(ProjectManifest)
    tpl = ChatPromptTemplate.from_template("""
You are an AI software architect. Plan a complete, minimal Python project.

Existing codebase context:
{context}

User request:
{task}

Rules:
- Source files go under src/, test files under tests/.
- Each source file has ONE clear responsibility and targets ≤100 lines.
- For EVERY source file, include a corresponding test file (is_test=true).
- Test files must contain complete, runnable pytest code in test_content.
- Prefer stdlib; only request external packages when genuinely necessary.
- Define 1-3 milestones in the milestones list (major deliverable descriptions).

Allowed stdlib imports: {allowlist}
Auto-approved packages: {auto_packages}

Return a JSON ProjectManifest.
""")
    return await (tpl | structured).ainvoke({
        "task":          prompt,
        "context":       context or "(empty project)",
        "allowlist":     ", ".join(sorted(SAFE_IMPORT_ALLOWLIST)),
        "auto_packages": ", ".join(sorted(AUTO_APPROVE_PACKAGES)),
    })


async def executive_audit(
    code: str,
    filename: str,
    test_output: str = "",
    passed: bool = False,
    security_reports: Optional[Dict[str, str]] = None,
    approved_packages: Optional[List[str]] = None,
) -> AuditReview:
    extra = frozenset(approved_packages or [])
    safe, reason = is_safe_code_ast(code, extra_allowed=extra)
    if not safe:
        return AuditReview(is_approved=False, feedback=f"AST safety: {reason}", security_score=0)

    llm        = ChatOllama(model=EXECUTIVE_MODEL, temperature=0.0)
    structured = llm.with_structured_output(AuditReview)

    sec_section = ""
    if security_reports:
        parts = [
            f"### {tool}\n{report[:600]}"
            for tool, report in security_reports.items()
            if tool != "coverage_pct"
        ]
        sec_section = "\n\n--- SECURITY REPORTS ---\n" + "\n\n".join(parts)

    tpl = ChatPromptTemplate.from_template("""
Review '{file}'.

--- CODE ---
```python
{code}
```

--- TEST RESULT ---
Passed: {passed}
Output (last 40 lines):
{test_output}
{sec_section}

Evaluate correctness, safety, style, security.
Return JSON: is_approved (bool), feedback (str), security_score (int 0-10).
""")
    try:
        return await (tpl | structured).ainvoke({
            "code":        code,
            "file":        filename,
            "passed":      passed,
            "test_output": "\n".join(test_output.splitlines()[-40:]),
            "sec_section": sec_section,
        })
    except Exception as exc:
        logger.warning(f"executive_audit LLM call failed: {exc}")
        return AuditReview(is_approved=False, feedback=f"Audit error: {exc}", security_score=0)


async def integrate_project(
    manifest: ProjectManifest, generated: Dict[str, str],
) -> IntegrationResult:
    llm        = ChatOllama(model=EXECUTIVE_MODEL, temperature=0.0)
    structured = llm.with_structured_output(IntegrationResult)
    surfaces = "\n\n".join(
        f"### {path}\n```python\n{extract_api_surface(content)}\n```"
        for path, content in generated.items()
    )
    tpl = ChatPromptTemplate.from_template("""
You are a senior engineer reviewing a multi-file Python project.

API surfaces:
{surfaces}

Check: import paths, circular imports, API mismatches, naming inconsistencies.
For each file needing change, provide the corrected COMPLETE file content.
Return JSON IntegrationResult.
""")
    try:
        return await (tpl | structured).ainvoke({"surfaces": surfaces[:12_000]})
    except Exception as exc:
        logger.warning(f"integrate_project failed: {exc}")
        return IntegrationResult(corrections=[], overall_notes=f"Skipped: {exc}")


# ── v5.2 executive functions ──────────────────────────────────────────────────
async def executive_plan_modification(
    target_file: str, current_code: str, instruction: str,
) -> ChangePlan:
    """--modify mode: produce a targeted unified diff rather than full regeneration."""
    llm        = ChatOllama(model=EXECUTIVE_MODEL, temperature=0.0)
    structured = llm.with_structured_output(ChangePlan)
    tpl = ChatPromptTemplate.from_template("""
You are a senior Python engineer making a targeted modification.

File: {target_file}
Current code:
```python
{current_code}
```

Instruction: {instruction}

Produce the minimal unified diff that implements the change.
Return JSON ChangePlan with target_file, reason, and unified_diff.
""")
    return await (tpl | structured).ainvoke({
        "target_file":  target_file,
        "current_code": current_code[:6_000],
        "instruction":  instruction,
    })


async def generate_docs(
    manifest: ProjectManifest, generated: Dict[str, str],
) -> List[DocPage]:
    """--docs mode: produce Markdown documentation from API surfaces."""
    llm        = ChatOllama(model=EXECUTIVE_MODEL, temperature=0.0)
    structured = llm.with_structured_output(DocPage)

    surfaces = "\n\n".join(
        f"### {path}\n```python\n{extract_api_surface(content)}\n```"
        for path, content in generated.items()
    )
    tpl = ChatPromptTemplate.from_template("""
You are a technical writer. Write comprehensive Markdown documentation
for the following Python project.

Project: {summary}

API surfaces:
{surfaces}

Produce a DocPage with filename='api_reference.md' and complete Markdown content.
""")
    try:
        page = await (tpl | structured).ainvoke({
            "summary":  manifest.task_summary,
            "surfaces": surfaces[:10_000],
        })
        return [page]
    except Exception as exc:
        logger.warning(f"generate_docs failed: {exc}")
        return []


async def compose_commit(
    manifest: ProjectManifest,
    all_results: Dict[str, Any],
    integration_notes: str,
) -> CommitInfo:
    """--commit mode: write a conventional commit message + PR body."""
    llm        = ChatOllama(model=EXECUTIVE_MODEL, temperature=0.0)
    structured = llm.with_structured_output(CommitInfo)

    changed = list(all_results.keys())
    tpl = ChatPromptTemplate.from_template("""
Write a conventional commit for these changes.

Project: {summary}
Files changed: {files}
Integration notes: {notes}

subject: one-line ≤72 chars, format "type(scope): description"
body: full PR description in Markdown

Return JSON CommitInfo.
""")
    try:
        return await (tpl | structured).ainvoke({
            "summary": manifest.task_summary,
            "files":   ", ".join(changed[:20]),
            "notes":   integration_notes[:500],
        })
    except Exception as exc:
        logger.warning(f"compose_commit failed: {exc}")
        return CommitInfo(subject="chore: auto-generated changes", body="See generated files.")


# ── v6.0 executive functions ──────────────────────────────────────────────────
async def executive_decompose_goal(goal: str, context: str) -> ProjectGoal:
    """--autonomous mode: decompose a high-level goal into epics and tasks."""
    llm        = ChatOllama(model=EXECUTIVE_MODEL, temperature=0.0)
    structured = llm.with_structured_output(ProjectGoal)
    tpl = ChatPromptTemplate.from_template("""
You are an AI software project manager.

Goal: {goal}

Existing codebase context:
{context}

Decompose this goal into:
1. A ProjectManifest (complete list of files to generate)
2. Epics (major feature areas, 2-4 epics)
3. Tasks within each epic, each referencing a specific file from the manifest

Rules:
- Each TaskNode has a unique id, description, optional file_spec path,
  depends_on list of other task ids, and optional milestone label.
- Keep tasks atomic: one file = one task.
- The manifest milestones list should match the epic titles.

Return JSON ProjectGoal.
""")
    try:
        return await (tpl | structured).ainvoke({
            "goal":    goal,
            "context": context or "(empty project)",
        })
    except Exception as exc:
        logger.warning(f"executive_decompose_goal failed: {exc}")
        # Fallback: treat as a simple manifest plan
        manifest = await executive_plan_project(goal, context)
        tasks = [
            TaskNode(
                id=f"task_{i}",
                description=f.description,
                file_spec=f,
                depends_on=[f"task_{j}" for j, other in enumerate(manifest.files[:i]) if other.path in f.dependencies],
            )
            for i, f in enumerate(manifest.files)
            if not f.is_test
        ]
        return ProjectGoal(
            summary=manifest.task_summary,
            epics=[Epic(title="Main", tasks=tasks)],
            manifest=manifest,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-AGENT DEBATE  (v5.2)
# ──────────────────────────────────────────────────────────────────────────────
async def run_debate(
    file_spec: FileSpec,
    manifest: ProjectManifest,
    project_context: str,
    approved_packages: List[str],
    n_engineers: int = 2,
) -> str:
    """
    Run n_engineers at staggered temperatures and have a Judge pick the best.
    Returns the winning code string.
    """
    allowed_imports = SAFE_IMPORT_ALLOWLIST | frozenset(approved_packages)

    async def _generate(temperature: float) -> str:
        llm = ChatOllama(model=ENGINEER_MODEL, temperature=temperature)
        tpl = ChatPromptTemplate.from_template("""
You are a Python engineer. Implement this file:

Project: {task_summary}
File: {path}
Purpose: {description}
Depends on: {deps}
Allowed imports: {allowlist}

API context:
{context}

Output ONLY a single ```python``` code block.
""")
        resp = await (tpl | llm).ainvoke({
            "task_summary": manifest.task_summary,
            "path":         file_spec.path,
            "description":  file_spec.description,
            "deps":         ", ".join(file_spec.dependencies) or "none",
            "allowlist":    ", ".join(sorted(allowed_imports)),
            "context":      project_context[:MAX_CONTEXT_CHARS],
        })
        return extract_python_code(resp.content)

    temperatures = [0.1 + 0.2 * i for i in range(n_engineers)]
    candidates   = await asyncio.gather(*[_generate(t) for t in temperatures])
    valid        = [c for c in candidates if c]

    if len(valid) == 1:
        return valid[0]
    if not valid:
        return ""

    # Judge agent picks the best
    llm = ChatOllama(model=EXECUTIVE_MODEL, temperature=0.0)
    judge_tpl = ChatPromptTemplate.from_template("""
You are a senior Python engineer acting as a judge.

Two candidates implement: {path} — {description}

Candidate A:
```python
{code_a}
```

Candidate B:
```python
{code_b}
```

Which is better? Consider correctness, style, and edge-case handling.
Reply with ONLY "A" or "B" and one sentence of reasoning.
""")
    try:
        resp = await (judge_tpl | llm).ainvoke({
            "path":        file_spec.path,
            "description": file_spec.description,
            "code_a":      valid[0][:2_000],
            "code_b":      valid[1][:2_000],
        })
        verdict = resp.content.strip().upper()
        winner  = valid[0] if verdict.startswith("A") else valid[1]
        logger.info(f"[Debate | {file_spec.path}] Judge verdict: {verdict[:60]}")
        return winner
    except Exception:
        return vote_on_candidates(valid)


# ──────────────────────────────────────────────────────────────────────────────
# ENGINEER  (per file, self-correcting — all v5.x features)
# ──────────────────────────────────────────────────────────────────────────────
async def run_engineer_on_file(
    instance_id:        int,
    file_spec:          FileSpec,
    manifest:           ProjectManifest,
    project_context:    str,
    temp_dir:           str,
    sandbox_image:      str,
    approved_packages:  List[str],
    success_memory:     SuccessMemory,
    failure_memory:     FailureMemory,
    error_cluster:      ErrorCluster,
    style_guide:        StyleGuide,
    feedback_store:     UserFeedbackStore,
    regression_suite:   RegressionSuite,
    use_docker:         bool = True,
    interactive:        bool = False,
) -> Dict[str, Any]:
    """
    Full v5.x iterative engineer with all small-model maximizer features.
    """
    lang_cfg  = LANGUAGE_CONFIGS.get(file_spec.language, LANGUAGE_CONFIGS["python"])
    llm       = ChatOllama(model=ENGINEER_MODEL, temperature=0.2)
    workspace = Path(temp_dir) / f"eng_{instance_id}_{Path(file_spec.path).stem}"
    workspace.mkdir(parents=True, exist_ok=True)

    allowed_imports = SAFE_IMPORT_ALLOWLIST | frozenset(approved_packages)
    style_section   = style_guide.build()

    # ── One-time retrieval ────────────────────────────────────────────────────
    example_raw = await success_memory.get_example(
        f"{manifest.task_summary} {file_spec.description}",
        language=file_spec.language,
    )
    few_shot_section = (
        "Here is a similar successful implementation (style guidance only):\n"
        f"```python\n{example_raw}\n```\n"
        if example_raw else ""
    )
    user_constraints = await feedback_store.get_constraints(
        f"{file_spec.description} {manifest.task_summary}"
    )

    # ── Prompt templates ──────────────────────────────────────────────────────
    gen_prompt = ChatPromptTemplate.from_template("""
You are a Python engineer implementing one file in a larger project.

Project goal: {task_summary}
Your file   : {path}
Purpose     : {description}
Depends on  : {dependencies}
Target lines: ≤{target_lines}

API context of other project files:
{project_context}

Allowed imports: {allowlist}

{style_section}
{few_shot_section}
{user_constraints}
{failure_context}

Write the complete implementation.
Output ONLY a single markdown code block:
```python
# your code here
```
""")

    stub_prompt = ChatPromptTemplate.from_template("""
You are a Python engineer designing the public interface of one file.

Project goal: {task_summary}
Your file   : {path}
Purpose     : {description}
Depends on  : {dependencies}
Allowed imports: {allowlist}

API context:
{project_context}

Write ONLY function/class signatures and one-line docstrings. Use `...` for every body.
Output ONLY a single ```python``` code block.
""")

    fill_prompt = ChatPromptTemplate.from_template("""
Fill in the implementation bodies for these stubs.
Keep every signature and docstring exactly as written — only replace `...`.

```python
{stubs}
```

Project goal: {task_summary}
Allowed imports: {allowlist}

{few_shot_section}
{failure_context}

Output ONLY the complete filled-in file as a single ```python``` code block.
""")

    diff_prompt = ChatPromptTemplate.from_template("""
The following Python file fails its tests.

```python
{code}
```

Test failure (last 20 lines):
{summary}

Self-critique: {critique}

Output ONLY a unified diff patch that fixes the failure — nothing else.
""")

    critique_prompt = ChatPromptTemplate.from_template("""
The following Python code failed its tests.

```python
{code}
```

Failure:
{summary}

In ONE sentence, identify the single most likely bug.
""")

    reflection_tpl = ChatPromptTemplate.from_template("""
Your implementation of {path} failed:

```python
{code}
```

Test failure:
{summary}

In one sentence: what is the most likely cause?
""")

    # ── Initial state ─────────────────────────────────────────────────────────
    empty_result: Dict[str, Any] = {
        "path": file_spec.path, "code": "", "passed": False,
        "audit_approved": False, "security_score": 0,
        "score": -float("inf"), "attempts": 0, "output": "",
        "security_reports": {},
    }
    best              = dict(empty_result)
    failure_context   = ""
    passing_versions: List[str] = []
    current_stubs:    Optional[str] = None

    for attempt in range(1, MAX_ENGINEER_RETRIES + 1):
        logger.info(
            f"[Engineer {instance_id} | {file_spec.path}] "
            f"Attempt {attempt}/{MAX_ENGINEER_RETRIES}"
        )
        impl_name   = Path(file_spec.path).name
        impl_module = Path(impl_name).stem

        # ── Debate mode ───────────────────────────────────────────────────────
        if attempt == 1 and DEBATE_ENGINEERS > 1:
            logger.info(f"[{file_spec.path}] Running debate ({DEBATE_ENGINEERS} engineers)...")
            code = await run_debate(
                file_spec, manifest, project_context, approved_packages, DEBATE_ENGINEERS
            )
            if not code:
                failure_context = "Debate produced no valid code. Falling back to standard generation."
        elif STUB_FIRST and attempt == 1:
            # Stub-first phase 1: generate interface
            resp = await (stub_prompt | llm).ainvoke({
                "task_summary":    manifest.task_summary,
                "path":            file_spec.path,
                "description":     file_spec.description,
                "dependencies":    ", ".join(file_spec.dependencies) or "none",
                "project_context": project_context[:MAX_CONTEXT_CHARS],
                "allowlist":       ", ".join(sorted(allowed_imports)),
            })
            current_stubs = extract_python_code(resp.content)
            if not current_stubs:
                failure_context = "No stub block found. Try again."
                continue
            # Smoke-test stubs
            stub_path = workspace / impl_name
            stub_path.write_text(current_stubs, encoding="utf-8")
            smoke_ok = False
            try:
                r = subprocess.run(
                    [sys.executable, "-c",
                     f"import importlib.util; "
                     f"spec=importlib.util.spec_from_file_location('m',r'{stub_path}'); "
                     f"mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)"],
                    capture_output=True, text=True, timeout=10,
                )
                smoke_ok = r.returncode == 0
            except Exception:
                pass
            if not smoke_ok:
                failure_context = "Stubs did not import cleanly. Fix syntax and resubmit."
                current_stubs = None
                continue
            logger.info(f"[{file_spec.path}] Stubs generated — filling bodies next.")
            failure_context = ""
            continue
        else:
            # Diff-based editing from attempt DIFF_EDIT_FROM onward
            code = best["code"]
            used_diff = False
            if attempt >= DIFF_EDIT_FROM and code and failure_context:
                summary = extract_failure_summary(best.get("output", ""))

                # Check error cluster for a known recipe
                cluster_recipe = error_cluster.recipe_for(summary)

                crit_resp = await (critique_prompt | llm).ainvoke({
                    "code": code, "summary": summary or "(no output)",
                })
                critique_text = crit_resp.content.strip()

                if cluster_recipe:
                    critique_text = f"{critique_text} (Known recipe: {cluster_recipe})"

                patch_resp = await (diff_prompt | llm).ainvoke({
                    "code": code, "summary": summary or "(no output)", "critique": critique_text,
                })
                patch_text = re.sub(r"^```[a-z]*\n?", "", patch_resp.content.strip(), flags=re.MULTILINE)
                patch_text = re.sub(r"```$", "", patch_text, flags=re.MULTILINE)
                patched = apply_unified_diff(code, patch_text)
                if patched:
                    code = patched
                    used_diff = True
                    logger.debug(f"[{file_spec.path}] Diff patch applied on attempt {attempt}.")
                else:
                    logger.debug(f"[{file_spec.path}] Patch malformed — full regen.")

            if not used_diff:
                base_vars = {
                    "task_summary":    manifest.task_summary,
                    "path":            file_spec.path,
                    "description":     file_spec.description,
                    "dependencies":    ", ".join(file_spec.dependencies) or "none",
                    "target_lines":    file_spec.target_lines,
                    "project_context": project_context[:MAX_CONTEXT_CHARS],
                    "allowlist":       ", ".join(sorted(allowed_imports)),
                    "style_section":   style_section,
                    "few_shot_section": few_shot_section,
                    "user_constraints": user_constraints,
                    "failure_context": failure_context,
                }
                if STUB_FIRST and current_stubs:
                    candidates = []
                    for _ in range(max(1, VOTE_RUNS)):
                        r = await (fill_prompt | llm).ainvoke({**base_vars, "stubs": current_stubs})
                        candidates.append(extract_python_code(r.content))
                else:
                    candidates = []
                    for _ in range(max(1, VOTE_RUNS)):
                        r = await (gen_prompt | llm).ainvoke(base_vars)
                        candidates.append(extract_python_code(r.content))

                valid_candidates = [c for c in candidates if c]
                if not valid_candidates:
                    failure_context = "No Python code block found. Try again."
                    continue
                code = vote_on_candidates(valid_candidates)
                if len(valid_candidates) > 1:
                    logger.debug(f"[{file_spec.path}] Voted among {len(valid_candidates)} candidates.")

        # ── Write workspace files ─────────────────────────────────────────────
        impl_path = workspace / impl_name
        test_path = workspace / "test_generated.py"
        impl_path.write_text(code, encoding="utf-8")

        test_spec = next(
            (f for f in manifest.files if f.is_test and file_spec.path in f.dependencies),
            None,
        )
        test_body = (test_spec.test_content or "") if test_spec else ""
        if not test_body:
            test_body = "def test_importable():\n    pass\n"

        test_path.write_text(
            "import sys, os\n"
            "sys.path.insert(0, os.path.dirname(__file__))\n"
            f"from {impl_module} import *\n\n" + test_body,
            encoding="utf-8",
        )

        # ── Run tests ─────────────────────────────────────────────────────────
        test_cmd = lang_cfg["test_runner"]
        if use_docker:
            passed, output, runtime = await run_tests_in_docker(workspace, sandbox_image, test_cmd)
        else:
            local_cmd = (
                [sys.executable, "-m", "pytest", "-q", "test_generated.py"]
                if file_spec.language == "python"
                else test_cmd
            )
            passed, output, runtime = await run_tests_local(workspace, local_cmd)

        logger.info(
            f"[{file_spec.path}] Tests {'PASSED' if passed else 'FAILED'} ({runtime:.1f}s)"
        )

        # ── Per-file security (Python + Docker only) ──────────────────────────
        security_reports: Dict[str, str] = {}
        if passed and use_docker and file_spec.language == "python":
            sec_ws  = workspace / "_sec"
            src_sub = sec_ws / "src"
            src_sub.mkdir(parents=True, exist_ok=True)
            (src_sub / impl_name).write_text(code, encoding="utf-8")
            _, security_reports = await run_security_pipeline(sec_ws, sandbox_image, "src")

        # ── Audit ─────────────────────────────────────────────────────────────
        audit = await executive_audit(
            code, file_spec.path,
            test_output=output, passed=passed,
            security_reports=security_reports,
            approved_packages=approved_packages,
        )
        logger.info(
            f"[{file_spec.path}] Audit {'OK' if audit.is_approved else 'REJECTED'} "
            f"security={audit.security_score}/10  {audit.feedback[:80]}"
        )

        # ── Rollback buffer ───────────────────────────────────────────────────
        if passed:
            passing_versions.append(code)
            if len(passing_versions) > ROLLBACK_DEPTH:
                passing_versions.pop(0)

        # ── Score ─────────────────────────────────────────────────────────────
        sc = score_result(passed, audit.is_approved, code, runtime, attempt, audit.security_score)
        prev_best_code = best["code"]
        if sc > best["score"]:
            best = {
                "path": file_spec.path, "code": code, "passed": passed,
                "audit_approved": audit.is_approved, "security_score": audit.security_score,
                "score": sc, "attempts": attempt, "output": output,
                "security_reports": security_reports,
            }

        if passed and audit.is_approved:
            logger.info(f"[{file_spec.path}] ✓ Success on attempt {attempt}.")
            await success_memory.add(file_spec, code, manifest.task_summary)
            # v5.1: add to regression suite
            if test_spec and test_spec.test_content:
                regression_suite.add(file_spec, test_spec.test_content)
            break

        # ── Self-critique + reflection + failure context ───────────────────────
        if not passed:
            summary = extract_failure_summary(output)
            try:
                crit_resp = await (critique_prompt | llm).ainvoke({
                    "code": code, "summary": summary or "(no output)",
                })
                critique_text = crit_resp.content.strip()
            except Exception:
                critique_text = ""

            past_hints = await failure_memory.get_hints(summary)
            cluster_recipe = error_cluster.recipe_for(summary)

            ref_resp = await (reflection_tpl | llm).ainvoke({
                "path": file_spec.path, "code": code, "summary": summary,
            })
            diagnosis = ref_resp.content.strip()

            fix_hint = critique_text or diagnosis
            await failure_memory.add(summary, fix_hint)
            error_cluster.record(summary, fix_hint)

            failure_context = (
                f"Previous attempt FAILED tests.\n\n"
                f"Failure summary:\n{summary}\n\n"
                f"Your self-critique: {critique_text}\n\n"
                f"Your diagnosis: {diagnosis}\n\n"
                + (f"Known recipe for this error type: {cluster_recipe}\n\n" if cluster_recipe else "")
                + (f"{past_hints}\n\n" if past_hints else "")
                + "Fix the code based on the above analysis."
            )

            if attempt == MAX_ENGINEER_RETRIES and passing_versions and not best["passed"]:
                logger.warning(f"[{file_spec.path}] All retries exhausted — rolling back.")
                best["code"] = passing_versions[-1]
        else:
            failure_context = (
                f"Previous attempt passed tests but was REJECTED by the auditor.\n"
                f"Feedback: {audit.feedback}\n\n"
                f"Fix the issue and resubmit ONLY the corrected code."
            )

        # ── Interactive diff review ───────────────────────────────────────────
        if interactive and prev_best_code and prev_best_code != code:
            diff = show_unified_diff(prev_best_code, code, file_spec.path)
            if diff:
                print(f"\n── Diff for {file_spec.path} (attempt {attempt}) ──\n{diff}")
            raw = await asyncio.to_thread(input, "Accept? [y/N/quit/reason=...] ")
            answer = raw.strip().lower()
            if answer == "y":
                best["code"] = code
                break
            elif answer.startswith("quit"):
                break
            elif answer.startswith("reason="):
                reason = raw.split("reason=", 1)[1].strip()
                await feedback_store.record(file_spec.description, reason)
                logger.info(f"Feedback recorded: {reason}")

    return best


# ──────────────────────────────────────────────────────────────────────────────
# v6.0 WORKFLOW ENGINE
# ──────────────────────────────────────────────────────────────────────────────
class WorkflowEngine:
    """
    asyncio.Queue-based task dispatcher.
    Unlocks dependent tasks as each task completes.
    Supports milestone checkpoints and session resumption.
    """

    def __init__(
        self,
        tasks: List[TaskNode],
        state: ProjectState,
        interactive: bool = False,
    ) -> None:
        self.task_map: Dict[str, TaskNode] = {t.id: t for t in tasks}
        self.state      = state
        self.interactive = interactive
        self.queue: asyncio.Queue[TaskNode] = asyncio.Queue()
        self.done:  Set[str] = set()
        self.failed: Set[str] = set()
        self._restore_session()

    def _restore_session(self) -> None:
        session = self.state.load_session()
        self.done   = set(session.get("done", []))
        self.failed = set(session.get("failed", []))
        # Mark previously completed tasks in task_map
        for tid in self.done:
            if tid in self.task_map:
                self.task_map[tid].status = "done"
        logger.info(
            f"Session restored: {len(self.done)} done, {len(self.failed)} failed."
        )

    def _save_session(self) -> None:
        self.state.save_session({
            "done":   list(self.done),
            "failed": list(self.failed),
        })

    def _enqueue_ready(self) -> None:
        """Enqueue all tasks whose dependencies are satisfied."""
        for task in self.task_map.values():
            if task.status != "pending":
                continue
            if all(dep in self.done for dep in task.depends_on):
                task.status = "queued"
                self.queue.put_nowait(task)

    async def _milestone_checkpoint(self, milestone: str) -> bool:
        """
        Pause at a milestone and ask the user whether to continue.
        Returns True to proceed, False to abort.
        """
        print(f"\n{'─' * 60}")
        print(f"  ✅ Milestone reached: {milestone}")
        print(f"{'─' * 60}")
        if not self.interactive:
            return True
        answer = await asyncio.to_thread(input, "Continue to next milestone? [Y/n] ")
        return answer.strip().lower() != "n"

    async def run(
        self,
        processor: Any,   # async callable(task) → bool (success)
        manifest: ProjectManifest,
    ) -> Dict[str, bool]:
        """
        Run all tasks through `processor`.
        Returns {task_id: success}.
        """
        # Seed initial tasks
        self._enqueue_ready()

        completed_per_milestone: Dict[str, int] = collections.defaultdict(int)
        milestone_sizes: Dict[str, int] = collections.defaultdict(int)
        for task in self.task_map.values():
            if task.milestone:
                milestone_sizes[task.milestone] += 1

        results: Dict[str, bool] = {}

        while not self.queue.empty() or any(
            t.status in ("pending", "queued") for t in self.task_map.values()
        ):
            try:
                task = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                # Nothing ready yet but tasks remain — likely waiting on running tasks
                await asyncio.sleep(0.1)
                continue

            if task.id in self.done:
                self.queue.task_done()
                continue

            task.status = "running"
            logger.info(f"[Workflow] Running task: {task.id} — {task.description}")

            try:
                success = await processor(task)
            except Exception as exc:
                logger.error(f"[Workflow] Task {task.id} raised: {exc}")
                success = False

            if success:
                task.status = "done"
                self.done.add(task.id)
                results[task.id] = True

                if task.milestone:
                    completed_per_milestone[task.milestone] += 1
                    if (
                        completed_per_milestone[task.milestone]
                        >= milestone_sizes[task.milestone]
                    ):
                        proceed = await self._milestone_checkpoint(task.milestone)
                        if not proceed:
                            logger.info("[Workflow] User aborted at milestone checkpoint.")
                            self._save_session()
                            return results
            else:
                task.status = "failed"
                self.failed.add(task.id)
                results[task.id] = False
                # Mark dependent tasks as blocked
                for other in self.task_map.values():
                    if task.id in other.depends_on:
                        other.status = "failed"
                        self.failed.add(other.id)

            self._save_session()
            self._enqueue_ready()
            self.queue.task_done()

        return results


# ──────────────────────────────────────────────────────────────────────────────
# HARNESS
# ──────────────────────────────────────────────────────────────────────────────
async def run_harness(
    user_prompt:  str,
    use_docker:   bool = True,
    interactive:  bool = False,
    modify_file:  Optional[str] = None,
    modify_instr: Optional[str] = None,
    docs_mode:    bool = False,
    commit_mode:  bool = False,
    autonomous:   bool = False,
) -> None:
    base, src, apollo = setup_workspace()
    temp_dir = os.path.join(apollo, "tmp")
    os.makedirs(temp_dir, exist_ok=True)

    state = ProjectState(Path(apollo) / "state.json")

    # ── Shared v5.x memory objects ────────────────────────────────────────────
    mem_dir         = os.path.join(apollo, "memory")
    success_memory  = SuccessMemory(os.path.join(mem_dir, "successes"))
    failure_memory  = FailureMemory(os.path.join(mem_dir, "failures"))
    error_cluster   = ErrorCluster()
    style_guide     = StyleGuide(src)
    feedback_store  = UserFeedbackStore(os.path.join(mem_dir, "feedback"))
    regression_suite = RegressionSuite(base)

    # ── Context ───────────────────────────────────────────────────────────────
    logger.info("Loading project context from vector memory...")
    vs_dir  = os.path.join(apollo, "chroma")
    context = await load_context(src, vs_dir, user_prompt)

    # ── v5.2: --modify mode ───────────────────────────────────────────────────
    if modify_file and modify_instr:
        target = Path(src) / modify_file
        if not target.exists():
            logger.error(f"--modify: file not found: {target}")
            return
        current_code = target.read_text(encoding="utf-8")
        logger.info(f"Generating modification plan for {modify_file}...")
        plan = await executive_plan_modification(modify_file, current_code, modify_instr)
        patched = apply_unified_diff(current_code, plan.unified_diff)
        if patched:
            if interactive:
                diff = show_unified_diff(current_code, patched, modify_file)
                if diff:
                    print(f"\n── Proposed change to {modify_file} ──\n{diff}")
                answer = await asyncio.to_thread(input, "Apply? [Y/n] ")
                if answer.strip().lower() == "n":
                    logger.info("Modification rejected.")
                    return
            target.write_text(patched, encoding="utf-8")
            state.mark_file(modify_file, patched, False, False)
            logger.info(f"Modification applied: {plan.reason}")
        else:
            logger.error("Modification patch did not apply cleanly.")
        return

    # ── v6.0: --autonomous mode ───────────────────────────────────────────────
    if autonomous:
        logger.info("Decomposing goal into epics and tasks (autonomous mode)...")
        project_goal = await executive_decompose_goal(user_prompt, context)
        manifest     = project_goal.manifest
        state.save_manifest(manifest)
        all_tasks    = [t for epic in project_goal.epics for t in epic.tasks]

        logger.info(
            f"Goal decomposed: {len(project_goal.epics)} epics, "
            f"{len(all_tasks)} tasks"
        )

        # Resolve dependencies & sandbox once
        approved_packages = await resolve_dependencies(
            manifest.external_packages, interactive=interactive
        )
        state.set_packages(approved_packages)
        sandbox_image = await prepare_sandbox(approved_packages) if use_docker else ""

        all_results: Dict[str, Any] = {}

        async def task_processor(task: TaskNode) -> bool:
            if task.file_spec is None:
                return True  # no-op task
            result = await run_engineer_on_file(
                instance_id=0,
                file_spec=task.file_spec,
                manifest=manifest,
                project_context="\n\n".join(
                    f"### {p}\n```python\n{extract_api_surface(r['code'])}\n```"
                    for p, r in all_results.items() if r.get("code")
                ),
                temp_dir=temp_dir,
                sandbox_image=sandbox_image,
                approved_packages=approved_packages,
                success_memory=success_memory,
                failure_memory=failure_memory,
                error_cluster=error_cluster,
                style_guide=style_guide,
                feedback_store=feedback_store,
                regression_suite=regression_suite,
                use_docker=use_docker,
                interactive=interactive,
            )
            all_results[task.file_spec.path] = result
            return result.get("passed", False)

        engine = WorkflowEngine(all_tasks, state, interactive=interactive)
        workflow_results = await engine.run(task_processor, manifest)
        logger.info(
            f"Workflow complete: {sum(workflow_results.values())}/{len(workflow_results)} tasks succeeded."
        )

    else:
        # ── Standard planning flow ────────────────────────────────────────────
        logger.info("Executive planning project manifest...")
        manifest = await executive_plan_project(user_prompt, context)
        state.save_manifest(manifest)

        src_files  = [f for f in manifest.files if not f.is_test]
        test_files = [f for f in manifest.files if f.is_test]
        logger.info(
            f"Manifest: {len(src_files)} source files, "
            f"{len(test_files)} test files, "
            f"{len(manifest.external_packages)} package(s)"
        )

        if interactive:
            print("\n" + "─" * 56)
            print(f"  Task   : {manifest.task_summary}")
            for f in manifest.files:
                tag  = "  [TEST]" if f.is_test else "  [SRC] "
                deps = f"  ← {', '.join(f.dependencies)}" if f.dependencies else ""
                print(f"{tag}  {f.path}{deps}")
            if manifest.external_packages:
                print(f"  Pkgs   : {', '.join(manifest.external_packages)}")
            if manifest.milestones:
                print(f"  Milestones: {' → '.join(manifest.milestones)}")
            print("─" * 56)
            answer = await asyncio.to_thread(input, "Proceed? [Y/n] ")
            if answer.strip().lower() == "n":
                logger.info("User aborted at plan review.")
                return

        approved_packages = await resolve_dependencies(
            manifest.external_packages, interactive=interactive
        )
        state.set_packages(approved_packages)

        sandbox_image = await prepare_sandbox(approved_packages) if use_docker else ""
        if not use_docker:
            logger.warning("--no-docker: running tests locally with NO sandbox isolation.")

        pending = state.pending_files(manifest, src)
        if not pending:
            logger.info("All source files are up-to-date. Nothing to do.")
        else:
            logger.info(f"{len(pending)} file(s) need generation.")
            waves      = dependency_waves(pending)
            all_results: Dict[str, Dict[str, Any]] = {}

            for wave_idx, wave in enumerate(waves):
                logger.info(
                    f"Wave {wave_idx + 1}/{len(waves)}: "
                    + ", ".join(f.path for f in wave)
                )
                ctx_parts = [
                    f"### {path}\n```python\n{extract_api_surface(res['code'])}\n```"
                    for path, res in all_results.items() if res.get("code")
                ]
                project_context = "\n\n".join(ctx_parts)

                tasks = [
                    run_engineer_on_file(
                        i, spec, manifest, project_context,
                        temp_dir, sandbox_image, approved_packages,
                        success_memory=success_memory,
                        failure_memory=failure_memory,
                        error_cluster=error_cluster,
                        style_guide=style_guide,
                        feedback_store=feedback_store,
                        regression_suite=regression_suite,
                        use_docker=use_docker,
                        interactive=interactive,
                    )
                    for i, spec in enumerate(wave)
                ]
                wave_results = await asyncio.gather(*tasks, return_exceptions=True)

                for spec, result in zip(wave, wave_results):
                    if isinstance(result, Exception):
                        logger.error(f"Engineer for {spec.path} raised: {result}")
                    else:
                        all_results[spec.path] = result

    # ── Integrator ────────────────────────────────────────────────────────────
    if len(all_results) > 1:
        logger.info("Running integrator phase...")
        generated_code = {p: r["code"] for p, r in all_results.items() if r.get("code")}
        integration    = await integrate_project(manifest, generated_code)
        if integration.corrections:
            logger.info(
                f"Integrator made {len(integration.corrections)} correction(s): "
                + integration.overall_notes
            )
            for fix in integration.corrections:
                if fix.path in all_results:
                    all_results[fix.path]["code"]           = fix.content
                    all_results[fix.path]["passed"]         = False
                    all_results[fix.path]["audit_approved"] = False
                    logger.info(f"  ↳ {fix.path}: {fix.change_summary}")
        else:
            logger.info(f"Integrator: {integration.overall_notes}")
        integration_notes = integration.overall_notes
    else:
        integration_notes = ""

    # ── Write source files ────────────────────────────────────────────────────
    for rel_path, result in all_results.items():
        if not result.get("code"):
            logger.warning(f"No code produced for {rel_path}, skipping.")
            continue
        target = safe_target_path(src, rel_path)
        if target is None:
            continue
        if interactive and target.exists():
            diff = show_unified_diff(
                target.read_text(encoding="utf-8"), result["code"], rel_path
            )
            if diff:
                print(f"\n── Changes to {rel_path} ──\n{diff}")
                answer = await asyncio.to_thread(input, "Apply changes? [Y/n] ")
                if answer.strip().lower() == "n":
                    continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result["code"], encoding="utf-8")
        state.mark_file(
            rel_path, result["code"],
            result.get("passed", False),
            result.get("audit_approved", False),
            result.get("security_score", 0),
        )
        logger.info(f"Written: {target}")

    # ── Write test files ──────────────────────────────────────────────────────
    if not autonomous:
        test_files = [f for f in manifest.files if f.is_test]
    for spec in (test_files if not autonomous else []):
        if not spec.test_content:
            continue
        test_target = Path(base) / spec.path
        test_target.parent.mkdir(parents=True, exist_ok=True)
        test_target.write_text(spec.test_content, encoding="utf-8")
        logger.info(f"Written test: {test_target}")

    # ── v5.2: --docs mode ─────────────────────────────────────────────────────
    if docs_mode:
        logger.info("Generating documentation (--docs mode)...")
        generated_code = {p: r["code"] for p, r in all_results.items() if r.get("code")}
        pages = await generate_docs(manifest, generated_code)
        docs_dir = Path(base) / "docs"
        docs_dir.mkdir(exist_ok=True)
        for page in pages:
            doc_path = docs_dir / page.filename
            doc_path.write_text(page.content, encoding="utf-8")
            logger.info(f"Doc written: {doc_path}")

    # ── v5.2: --commit mode ───────────────────────────────────────────────────
    if commit_mode:
        logger.info("Composing commit message (--commit mode)...")
        commit_info = await compose_commit(manifest, all_results, integration_notes)
        commit_path = Path(apollo) / "commit_message.txt"
        commit_path.write_text(
            f"{commit_info.subject}\n\n{commit_info.body}", encoding="utf-8"
        )
        print(f"\n── Suggested commit ──\n{commit_info.subject}\n")
        logger.info(f"Commit message saved: {commit_path}")

    # ── Full project security sweep ───────────────────────────────────────────
    project_security_reports: Dict[str, str] = {}
    if use_docker and all_results:
        logger.info("Running full project security pipeline...")
        full_ws = Path(temp_dir) / "full_sweep"
        full_ws.mkdir(exist_ok=True)
        shutil.copytree(src, full_ws / "src", dirs_exist_ok=True)
        _, project_security_reports = await run_security_pipeline(
            full_ws, sandbox_image, "src"
        )
        coverage = project_security_reports.get("coverage_pct", "?")
        logger.info(f"Project coverage: {coverage}%")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "prompt":            user_prompt,
        "manifest":          manifest.model_dump(),
        "approved_packages": approved_packages if not autonomous else [],
        "sandbox_image":     sandbox_image if not autonomous else "",
        "results": {
            p: {k: v for k, v in r.items() if k not in ("code", "output")}
            for p, r in all_results.items()
        },
        "project_security":  project_security_reports,
        "error_cluster":     error_cluster.summary(),
    }
    report_path = Path(apollo) / "last_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Report saved: {report_path}")

    if use_docker:
        await cleanup_containers()

    # ── Summary ───────────────────────────────────────────────────────────────
    total    = len(all_results)
    n_passed = sum(1 for r in all_results.values() if r.get("passed"))
    n_audit  = sum(1 for r in all_results.values() if r.get("audit_approved"))
    coverage = project_security_reports.get("coverage_pct", "n/a")

    print("\n" + "=" * 60)
    print(f"  Files generated  : {total}")
    print(f"  Tests passing    : {n_passed}/{total}")
    print(f"  Audit approved   : {n_audit}/{total}")
    if use_docker:
        print(f"  Coverage         : {coverage}%")
        print(f"  Sandbox image    : {sandbox_image if not autonomous else 'n/a'}")
    print(f"  Report           : {report_path}")
    print(f"  Error clusters   :\n{error_cluster.summary()}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL HANDLING
# ──────────────────────────────────────────────────────────────────────────────
def register_signals(loop: asyncio.AbstractEventLoop) -> None:
    async def _shutdown() -> None:
        logger.info("Signal received – cleaning up containers...")
        await cleanup_containers()
        loop.stop()

    def _request_shutdown() -> None:
        loop.create_task(_shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_shutdown)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apollo 13 v6.0 – Autonomous Software Engineer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Standard multi-file build
          apollo13.py "build a CSV parser with filtering and aggregation"

          # Review plan without generating code
          apollo13.py "REST API client for GitHub" --dry-run

          # Interactive plan + diff review
          apollo13.py "data pipeline with pandas" --interactive

          # Self-consistency voting (3 candidates per attempt)
          apollo13.py "binary search tree" --vote-runs 3

          # Stub-first development for complex files
          apollo13.py "async HTTP client" --stub-first

          # Multi-agent debate (2 engineers, judge picks winner)
          apollo13.py "LRU cache implementation" --debate 2

          # Modify an existing file
          apollo13.py "add retry logic" --modify src/client.py --instruction "add exponential backoff"

          # Generate API docs after build
          apollo13.py "my project" --docs

          # Write a conventional commit message after build
          apollo13.py "my project" --commit

          # Fully autonomous from a single sentence
          apollo13.py "Build a REST API for a todo list" --autonomous

          # No Docker, more retries, verbose
          apollo13.py "binary search tree" --no-docker --max-retries 5 --verbose
        """),
    )

    parser.add_argument("prompt", nargs="+", help="Project description or goal")

    # Execution modes
    mode_group = parser.add_argument_group("Execution modes")
    mode_group.add_argument(
        "--dry-run", action="store_true",
        help="Print manifest and exit without generating code",
    )
    mode_group.add_argument(
        "--interactive", "-i", action="store_true",
        help="Human-in-the-loop: plan approval, package approval, per-file diffs",
    )
    mode_group.add_argument(
        "--autonomous", action="store_true",
        help="v6.0: Decompose goal into epics → tasks → workflow engine build",
    )
    mode_group.add_argument(
        "--modify", metavar="FILE",
        help="v5.2: Path (relative to src/) of an existing file to modify",
    )
    mode_group.add_argument(
        "--instruction", metavar="TEXT",
        help="v5.2: Natural-language instruction for --modify mode",
    )
    mode_group.add_argument(
        "--docs", action="store_true",
        help="v5.2: Generate Markdown API documentation after building",
    )
    mode_group.add_argument(
        "--commit", action="store_true",
        help="v5.2: Compose a conventional commit message + PR body after building",
    )

    # Small-model tuning
    sm_group = parser.add_argument_group("Small-model tuning (v5.0)")
    sm_group.add_argument(
        "--vote-runs", type=int, default=None, metavar="N",
        help=f"Run engineer N times per attempt; majority-vote selects output. Default: {VOTE_RUNS}",
    )
    sm_group.add_argument(
        "--stub-first", action="store_true",
        help="Attempt 1 generates stubs; attempt 2+ fills bodies",
    )
    sm_group.add_argument(
        "--debate", type=int, default=None, metavar="N",
        help="Run N engineers in parallel; judge picks the winner",
    )

    # Model / infra overrides
    inf_group = parser.add_argument_group("Infrastructure overrides")
    inf_group.add_argument(
        "--no-docker", action="store_true",
        help="Run tests locally (no sandbox; faster, less secure)",
    )
    inf_group.add_argument(
        "--executive-model", default=None, metavar="MODEL",
        help=f"Override executive model (default: {EXECUTIVE_MODEL})",
    )
    inf_group.add_argument(
        "--engineer-model", default=None, metavar="MODEL",
        help=f"Override engineer model (default: {ENGINEER_MODEL})",
    )
    inf_group.add_argument(
        "--max-retries", type=int, default=None, metavar="N",
        help=f"Engineer retry limit (default: {MAX_ENGINEER_RETRIES})",
    )
    inf_group.add_argument(
        "--docker-image", default=None, metavar="IMAGE",
        help=f"Base Docker image for sandbox (default: {DOCKER_BASE_IMAGE})",
    )
    inf_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="DEBUG-level logging",
    )

    args = parser.parse_args()

    # Apply globals
    global EXECUTIVE_MODEL, ENGINEER_MODEL, MAX_ENGINEER_RETRIES
    global DOCKER_BASE_IMAGE, VOTE_RUNS, STUB_FIRST, DEBATE_ENGINEERS

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")
    if args.vote_runs is not None:
        VOTE_RUNS = max(1, args.vote_runs)
        logger.info(f"Vote runs        → {VOTE_RUNS}")
    if args.stub_first:
        STUB_FIRST = True
        logger.info("Stub-first mode  → enabled")
    if args.debate is not None:
        DEBATE_ENGINEERS = max(1, args.debate)
        logger.info(f"Debate engineers → {DEBATE_ENGINEERS}")
    if args.executive_model:
        EXECUTIVE_MODEL = args.executive_model
        logger.info(f"Executive model  → {EXECUTIVE_MODEL}")
    if args.engineer_model:
        ENGINEER_MODEL = args.engineer_model
        logger.info(f"Engineer model   → {ENGINEER_MODEL}")
    if args.max_retries:
        MAX_ENGINEER_RETRIES = args.max_retries
        logger.info(f"Max retries      → {MAX_ENGINEER_RETRIES}")
    if args.docker_image:
        DOCKER_BASE_IMAGE = args.docker_image
        logger.info(f"Docker base image → {DOCKER_BASE_IMAGE}")

    use_docker  = not args.no_docker
    user_prompt = " ".join(args.prompt)

    if args.dry_run:
        base, src, apollo = setup_workspace()
        vs_dir   = os.path.join(apollo, "chroma")
        context  = await load_context(src, vs_dir, user_prompt)
        manifest = await executive_plan_project(user_prompt, context)
        print(f"\n── Dry Run: {manifest.task_summary} ─────────────────────")
        for f in manifest.files:
            tag  = "[TEST]" if f.is_test else "[SRC] "
            lang = f"  [{f.language}]" if f.language != "python" else ""
            deps = f"  ← {', '.join(f.dependencies)}" if f.dependencies else ""
            print(f"  {tag}{lang}  {f.path}{deps}")
            print(f"           {f.description}")
        if manifest.external_packages:
            print(f"\n  Packages   : {', '.join(manifest.external_packages)}")
        if manifest.milestones:
            print(f"  Milestones : {' → '.join(manifest.milestones)}")
        print(f"\n  Executive  : {EXECUTIVE_MODEL}")
        print(f"  Engineer   : {ENGINEER_MODEL}")
        print(f"  Docker     : {use_docker}")
        print(f"  Vote runs  : {VOTE_RUNS}")
        print(f"  Stub-first : {STUB_FIRST}")
        print(f"  Debate     : {DEBATE_ENGINEERS}")
        return

    await run_harness(
        user_prompt,
        use_docker=use_docker,
        interactive=args.interactive,
        modify_file=args.modify,
        modify_instr=args.instruction,
        docs_mode=args.docs,
        commit_mode=args.commit,
        autonomous=args.autonomous,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    register_signals(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
