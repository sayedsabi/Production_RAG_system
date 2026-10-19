
from pathlib import Path
import pymupdf


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_document(file_path: Path) -> dict:
    """Load a text, Markdown, or PDF document."""

    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        with pymupdf.open(file_path) as pdf:
            text = "\n".join(
                page.get_text() for page in pdf
            )
    else:
        text = file_path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(
            f"Document contains no extractable text: {file_path.name}"
        )

    return {
        "document_id": file_path.stem,
        "source": file_path.as_posix(),
        "file_type": extension.lstrip("."),
        "text": text,
    }


def load_directory(directory_path: Path) -> dict:
    """Load supported documents from a directory.

    Returns successfully loaded documents and a list of failures.
    """

    directory_path = Path(directory_path)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory_path}"
        )

    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {directory_path}"
        )

    documents = []
    failures = []

    # Process files in a predictable order.
    for file_path in sorted(directory_path.iterdir()):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            document = load_document(file_path)
            documents.append(document)

        except (OSError, UnicodeError, ValueError) as error:
            failures.append({
                "source": file_path.as_posix(),
                "error": str(error),
            })

    return {
        "documents": documents,
        "failures": failures,
    }
