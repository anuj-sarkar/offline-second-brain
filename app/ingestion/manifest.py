"""
app/ingestion/manifest.py

Manages document SHA-256 hash tracking to enable fast incremental indexing.
Only new or modified documents need to be processed; unchanged documents are skipped,
and deleted documents can be identified and purged from indices.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = "data/manifest.json"


@dataclass
class DocumentManifestEntry:
    filename: str
    sha256: str
    file_type: str
    char_count: int
    page_count: int
    chunk_count: int
    last_indexed: str


def compute_file_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hex digest for a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


class IngestionManifest:
    """Tracks indexed document hashes to avoid redundant re-indexing."""

    def __init__(self, manifest_path: str = DEFAULT_MANIFEST_PATH):
        self.manifest_path = Path(manifest_path)
        self.entries: dict[str, DocumentManifestEntry] = {}
        self.load()

    def load(self) -> None:
        """Load manifest from disk if it exists."""
        if not self.manifest_path.exists():
            self.entries = {}
            return

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.entries = {
                    filename: DocumentManifestEntry(**info)
                    for filename, info in data.items()
                }
        except Exception as e:
            logger.warning(f"Failed to load manifest at {self.manifest_path}: {e}")
            self.entries = {}

    def save(self) -> None:
        """Save current manifest state to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                data = {
                    filename: asdict(entry)
                    for filename, entry in self.entries.items()
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save manifest to {self.manifest_path}: {e}")

    def is_file_changed(self, file_path: Path) -> bool:
        """Return True if the file is new or its SHA-256 hash has changed."""
        filename = file_path.name
        if filename not in self.entries:
            return True
        current_hash = compute_file_sha256(file_path)
        return self.entries[filename].sha256 != current_hash

    def update_entry(
        self,
        filename: str,
        sha256: str,
        file_type: str,
        char_count: int,
        page_count: int,
        chunk_count: int,
    ) -> None:
        """Add or update an entry in the manifest."""
        self.entries[filename] = DocumentManifestEntry(
            filename=filename,
            sha256=sha256,
            file_type=file_type,
            char_count=char_count,
            page_count=page_count,
            chunk_count=chunk_count,
            last_indexed=datetime.now(timezone.utc).isoformat(),
        )
        self.save()

    def remove_entry(self, filename: str) -> Optional[DocumentManifestEntry]:
        """Remove an entry from the manifest."""
        entry = self.entries.pop(filename, None)
        if entry:
            self.save()
        return entry

    def get_stale_filenames(self, current_files: list[Path]) -> list[str]:
        """Find filenames recorded in the manifest that no longer exist on disk."""
        current_names = {p.name for p in current_files}
        return [fname for fname in self.entries if fname not in current_names]
