from pathlib import Path

from pypdf import PdfReader


def extract_resume_text(file_path: str | Path) -> str:
    reader = PdfReader(str(file_path))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return "\n\n".join(pages).strip()
