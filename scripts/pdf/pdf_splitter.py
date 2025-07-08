import os
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from scripts.utils.logger import setup_logger

logger = setup_logger("pdf_splitter")


def run_pdf_split(
    input_pdf: Path, output_dir: Path, pages_per_chunk: int = 10
) -> list[Path]:
    """
    Splits a PDF into smaller PDFs with a fixed number of pages per chunk.
    Returns list of generated chunk paths.
    """
    logger.info(f"Loading PDF: {input_pdf}")
    reader = PdfReader(str(input_pdf))
    total_pages = len(reader.pages)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Splitting into chunks of {pages_per_chunk} pages each")

    chunk_paths = []
    for i in range(0, total_pages, pages_per_chunk):
        writer = PdfWriter()
        for j in range(i, min(i + pages_per_chunk, total_pages)):
            writer.add_page(reader.pages[j])

        chunk_path = (
            output_dir / f"{input_pdf.stem}_part_{i // pages_per_chunk + 1}.pdf"
        )
        with open(chunk_path, "wb") as f:
            writer.write(f)
        logger.info(f"Written chunk: {chunk_path}")
        chunk_paths.append(chunk_path)

    logger.info("PDF splitting complete.")
    return chunk_paths
