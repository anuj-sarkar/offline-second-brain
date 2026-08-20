"""
tests/test_manifest.py

Tests for app.ingestion.manifest (SHA-256 hash tracking and incremental caching).
"""

import tempfile
from pathlib import Path

from app.ingestion.manifest import IngestionManifest, compute_file_sha256


def test_compute_file_sha256_and_manifest():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_file = tmp_path / "notes.md"
        test_file.write_text("# Test Notes\nSome content here.", encoding="utf-8")

        manifest_file = tmp_path / "manifest.json"
        manifest = IngestionManifest(manifest_path=str(manifest_file))

        assert manifest.is_file_changed(test_file) is True

        sha = compute_file_sha256(test_file)
        manifest.update_entry(
            filename="notes.md",
            sha256=sha,
            file_type=".md",
            char_count=len(test_file.read_text(encoding="utf-8")),
            page_count=1,
            chunk_count=1,
        )

        assert manifest.is_file_changed(test_file) is False

        # Modify file
        test_file.write_text("# Test Notes\nModified content.", encoding="utf-8")
        assert manifest.is_file_changed(test_file) is True
