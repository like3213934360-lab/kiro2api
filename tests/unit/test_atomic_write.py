# -*- coding: utf-8 -*-

"""Atomic state writes, including two processes writing at the same time."""

import json
import os
import subprocess
import sys
import threading

import pytest

from kiro.utils import atomic_write_text


class TestAtomicWriteText:
    def test_writes_and_replaces(self, tmp_path):
        """
        What it does: Writes a file twice.
        Purpose: The second write must replace the first, not append.
        """
        target = tmp_path / "state.json"
        atomic_write_text(target, "first")
        atomic_write_text(target, "second")

        assert target.read_text(encoding="utf-8") == "second"

    def test_creates_missing_directories(self, tmp_path):
        """
        What it does: Writes into a nested directory that does not exist.
        Purpose: First run on a fresh install must not fail.
        """
        target = tmp_path / "a" / "b" / "state.json"
        atomic_write_text(target, "x")

        assert target.read_text(encoding="utf-8") == "x"

    def test_leaves_no_temp_files(self, tmp_path):
        """
        What it does: Writes repeatedly and lists the directory.
        Purpose: A leaked temp file next to state.json is confusing and grows
                 across restarts.
        """
        target = tmp_path / "state.json"
        for i in range(20):
            atomic_write_text(target, str(i))

        assert [p.name for p in tmp_path.iterdir()] == ["state.json"]

    def test_failed_replace_keeps_the_original(self, tmp_path, monkeypatch):
        """
        What it does: Makes the final rename fail.
        Purpose: A failed save must not truncate the existing state file.
        """
        target = tmp_path / "state.json"
        atomic_write_text(target, "original")

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)

        with pytest.raises(OSError):
            atomic_write_text(target, "replacement")

        assert target.read_text(encoding="utf-8") == "original"
        assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


class TestConcurrentWriters:
    def test_threads_never_expose_a_partial_document(self, tmp_path):
        """
        What it does: Four threads write JSON while another reads and parses it.
        Purpose: A restart leaves two processes saving the same file; a reader
                 (or the next process) must never see half a document.
        """
        target = tmp_path / "cache_meter.json"
        atomic_write_text(target, json.dumps({"peaks": {}}))
        stop = threading.Event()
        parse_errors = []

        def writer(tag):
            for i in range(60):
                atomic_write_text(
                    target,
                    json.dumps({"peaks": {f"{tag}{i}": 0.5}, "pad": "x" * 4000}),
                )

        def reader():
            while not stop.is_set():
                try:
                    json.loads(target.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, FileNotFoundError) as exc:
                    parse_errors.append(repr(exc))

        reading = threading.Thread(target=reader)
        reading.start()
        writers = [threading.Thread(target=writer, args=(f"w{n}",)) for n in range(4)]
        for w in writers:
            w.start()
        for w in writers:
            w.join()
        stop.set()
        reading.join()

        assert parse_errors == []
        json.loads(target.read_text(encoding="utf-8"))
        assert [p.name for p in tmp_path.iterdir()] == ["cache_meter.json"]

    def test_two_processes_share_one_file(self, tmp_path):
        """
        What it does: Runs two real processes writing the same file.
        Purpose: The actual restart scenario — separate PIDs, separate memory,
                 one file. With a fixed temp name this corrupted the file.
        """
        target = tmp_path / "state.json"
        script = (
            "import json, sys\n"
            "from kiro.utils import atomic_write_text\n"
            "target, tag = sys.argv[1], sys.argv[2]\n"
            "for i in range(80):\n"
            "    atomic_write_text(target, json.dumps({'writer': tag, 'i': i, 'pad': 'x' * 2000}))\n"
        )
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(target), f"p{n}"],
                cwd=str(tmp_path.parent.parent),
                env={**os.environ, "PYTHONPATH": os.getcwd()},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            for n in range(2)
        ]
        for proc in procs:
            _, err = proc.communicate(timeout=120)
            assert proc.returncode == 0, err.decode()[-500:]

        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["writer"] in ("p0", "p1")
        assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
