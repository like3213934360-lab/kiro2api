# -*- coding: utf-8 -*-

"""
Invariants every content rewrite must satisfy.

These exist because spot checks kept missing cases: a timestamped log parsed as
grep, a traceback regrouped as grep hits, a source file mistaken for git status.
Each one silently removed text the model needed, and the model answered by
re-running the tool. The invariants below are checked against a corpus that
covers every format rule, so a new regression fails here instead of in a session.
"""

import re

import pytest

from kiro.tool_compression import DEDUPE_MARKER, MARKER, compress_text

ERROR_RE = re.compile(
    r"(error|failed|failure|exception|traceback|fatal|panic|not found|exit=|✗|FAILED|ERROR)",
    re.IGNORECASE,
)

# Anything a rule may add to say "content was shortened". If a rewrite has none
# of these, the model cannot tell that something is missing.
MARKERS = (MARKER.strip(), DEDUPE_MARKER, "more]", "matches in", "lines omitted")


def _corpus() -> dict:
    """Realistic tool outputs, one per format the compressor knows about."""
    return {
        "timestamped_build_log": "\n".join(
            f"2026-10-31 12:30:{i % 60:02d}.123 [INFO] hvigor task {i} done" for i in range(300)
        ) + "\n2026-10-31 12:35:00.000 [ERROR] StarShell.ets:42 undefined symbol cameraDegree",
        "python_traceback": "Traceback (most recent call last):\n" + "\n".join(
            f'  File "/tmp/x.py", line {i}, in <module>\n    run({i})' for i in range(60)
        ) + "\nValueError: boom",
        "plain_grep": "\n".join(f"src/file{i % 5}.ets:{i}:match text" for i in range(800)),
        "grep_with_error_hits": "\n".join(
            f"src/f{i % 4}.ets:{i}:ERROR something broke {i}" for i in range(300)
        ),
        "unified_diff": "\n".join(
            ["diff --git a/a.ets b/a.ets", "--- a/a.ets", "+++ b/a.ets", "@@ -1,3 +1,3 @@"]
            + [f"+const added{i} = {i};" for i in range(600)]
        ),
        "git_status": (
            "On branch main\n\nChanges not staged for commit:\n"
            + "\n".join(f"\tmodified:   src/f{i}.py" for i in range(40))
            + "\n\nUntracked files:\n\tnew.txt"
        ),
        "directory_tree": "src\n" + "\n".join(f"├── file{i}.ets" for i in range(300)),
        "path_listing": "\n".join(f"src/main/ets/mod{i}/Index.ets" for i in range(300)),
        "file_read": "\n".join(f"export function fn{i}(): number {{ return {i}; }}" for i in range(1500)),
        "json_payload": "[\n" + ",\n".join(
            f'  {{"id": {i}, "path": "/a/b/{i}", "ok": true}}' for i in range(400)
        ) + "\n]",
        "source_with_branch_text": "// On branch main the router restarts\n" + "\n".join(
            f"const v{i} = compute({i});" for i in range(300)
        ),
        "source_with_tree_char": "\n".join(
            ("// ├─ step" if i % 10 == 0 else f"const v{i} = {i};") for i in range(400)
        ),
        "unicode_output": "\n".join(f"第 {i} 行：相机方向已更新" for i in range(400)),
    }


@pytest.fixture(scope="module")
def corpus():
    return _corpus()


class TestNoSilentLoss:
    """A rewrite must either keep the text or tell the model it was shortened."""

    def test_every_rewrite_is_labelled(self, corpus):
        """
        What it does: Runs every corpus item through the compressor.
        Purpose: If bytes change without a marker, the model sees a truncated
                 answer with no way to know something was removed.
        """
        unlabelled = []
        for name, text in corpus.items():
            out, kind = compress_text(text)
            if out == text:
                continue
            if not any(marker in out for marker in MARKERS):
                unlabelled.append((name, kind, len(text), len(out)))

        assert unlabelled == [], f"rewrites without a marker: {unlabelled}"

    def test_error_lines_survive(self, corpus):
        """
        What it does: Counts error-looking lines before and after compression.
        Purpose: The verdict is what the model acts on; losing it sends the agent
                 back to re-run the command.
        """
        lost = []
        for name, text in corpus.items():
            before = len(ERROR_RE.findall(text))
            out, kind = compress_text(text)
            after = len(ERROR_RE.findall(out))
            if after < before and not any(marker in out for marker in MARKERS):
                lost.append((name, kind, before, after))

        assert lost == [], f"error lines dropped without a marker: {lost}"

    def test_compression_is_idempotent(self, corpus):
        """
        What it does: Compresses every item twice.
        Purpose: Kiro's prefix cache compares bytes across turns, so a second
                 pass must not change the text again.
        """
        unstable = []
        for name, text in corpus.items():
            once, _ = compress_text(text)
            twice, _ = compress_text(once)
            if twice != once:
                unstable.append((name, len(once), len(twice)))

        assert unstable == [], f"second pass changed the bytes: {unstable}"


class TestFormatSafety:
    """Content that only resembles a format must not be rewritten as that format."""

    def test_traceback_is_never_grep(self, corpus):
        """
        What it does: Compresses a stack trace.
        Purpose: 'File "x.py", line 51' fits path:line:content; regrouping it
                 dropped the exception text in a real session.
        """
        out, kind = compress_text(corpus["python_traceback"])

        assert kind != "grep"
        assert "ValueError: boom" in out

    def test_grep_keeps_error_hits(self, corpus):
        """
        What it does: Compresses grep output where every hit mentions an error.
        Purpose: The cap must not eat the matches the model asked for; at 100 per
                 file this corpus fits entirely.
        """
        text = corpus["grep_with_error_hits"]
        out, kind = compress_text(text)

        assert kind in ("grep", None)
        assert out.count("ERROR something broke") == text.count("ERROR something broke")

    def test_source_files_are_untouched(self, corpus):
        """
        What it does: Compresses the two code samples.
        Purpose: A quoted branch name or one box-drawing comment is not a format.
        """
        for name in ("source_with_branch_text", "source_with_tree_char"):
            out, kind = compress_text(corpus[name])
            assert out == corpus[name], f"{name} was rewritten as {kind}"
