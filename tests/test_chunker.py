
import pytest

from src.chunking.chunker import chunk_text


def test_short_text_returns_one_chunk():
    result = chunk_text("Hello world", chunk_size=100, overlap=10)

    assert result == ["Hello world"]


def test_long_text_is_split():
    text = "abcdefghij"

    result = chunk_text(text, chunk_size=4, overlap=1)

    assert result == ["abcd", "defg", "ghij"]


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ") == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=10)


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)
