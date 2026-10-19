
from pathlib import Path

import pytest

from src.ingestion.loader import load_document, load_directory


def test_load_markdown(tmp_path):
    file_path = tmp_path / "sample.md"
    file_path.write_text(
        "# API Security\nTokens expire after 15 minutes.",
        encoding="utf-8",
    )

    document = load_document(file_path)

    assert document["document_id"] == "sample"
    assert document["file_type"] == "md"
    assert "Tokens expire after 15 minutes" in document["text"]


def test_empty_document_is_rejected(tmp_path):
    file_path = tmp_path / "empty.md"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(file_path)


def test_directory_continues_after_bad_file(tmp_path):
    (tmp_path / "good.md").write_text(
        "Valid document text.",
        encoding="utf-8",
    )
    (tmp_path / "empty.md").write_text("", encoding="utf-8")

    result = load_directory(tmp_path)

    assert len(result["documents"]) == 1
    assert len(result["failures"]) == 1
    assert result["documents"][0]["document_id"] == "good"
