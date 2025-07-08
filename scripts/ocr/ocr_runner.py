import os
from pathlib import Path
from scripts.utils.logger import setup_logger
from config.config import MISTRAL_API_KEY
from mistralai import Mistral

logger = setup_logger("ocr_runner")


def load_mistral_ocr_model():
    """
    Loads the Mistral OCR model.
    Replace with your actual model loading logic.
    """
    logger.info("Loading Mistral OCR model...")
    if not MISTRAL_API_KEY:
        raise ValueError("Mistral API key is missing or invalid.")
    try:
        mistral_client = Mistral(
            api_key=MISTRAL_API_KEY
        )  # Mistral using mistral models
    except Exception as e:
        raise ValueError(f"Error configuring Mistral client: {e}")

    print("Mistral client configured successfully")
    return mistral_client


def run_mistral_ocr(model, pdf_path: Path, debug: bool) -> str:
    """
    Runs OCR on a single PDF file using Mistral OCR model.
    Replace this with the actual call to the model.
    """
    logger.info(f"Running OCR on: {pdf_path}")
    print(f"Debug mode: {debug}")
    if debug:
        logger.debug(f"Debug mode is enabled. Processing file: {pdf_path}")
        return "Debug mode is enabled. No OCR performed."
    uploaded_pdf = model.files.upload(
        file={
            "file_name": pdf_path,
            "content": open(pdf_path, "rb"),
        },
        purpose="ocr",
    )
    signed_url = model.files.get_signed_url(file_id=uploaded_pdf.id)
    ocr_response = model.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
        include_image_base64=True,
    )

    return ocr_response


def ocr_pdf_file(
    pdf_path: Path, output_dir: Path, model=None, debug: bool = False
) -> Path:
    """
    OCRs a single PDF file and writes the output text to a .txt file.
    """
    logger.info(f"Starting OCR for {pdf_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if model is None:
        model = load_mistral_ocr_model()

    extracted_text = run_mistral_ocr(model, str(pdf_path), debug)
    output_txt_path = output_dir / f"{pdf_path.stem}.txt"

    with open(output_txt_path, "w", encoding="utf-8") as f:
        if debug:
            f.write("Debug mode is enabled. No OCR performed.\n")
            logger.debug("Debug mode is enabled. No OCR performed.")
            return output_txt_path
        for i, page in enumerate(extracted_text.pages):
            page_text = str(page.markdown)
            f.write(f"\n\n# Page {i + 1}\n{page_text}")

    logger.info(f"OCR complete for {pdf_path}. Output saved to {output_txt_path}")
    return output_txt_path


def batch_ocr(
    pdf_paths: list[Path], output_dir: Path, model=None, debug: bool = False
) -> list[Path]:
    """
    OCRs a list of PDF files and returns paths to generated .txt files.
    """
    logger.info(f"OCR batch processing started for {len(pdf_paths)} files.")
    model = model or load_mistral_ocr_model()
    output_txt_paths = []

    for pdf_path in pdf_paths:
        txt_path = ocr_pdf_file(pdf_path, output_dir, model, debug)
        output_txt_paths.append(txt_path)

    logger.info("OCR batch processing complete.")
    return output_txt_paths
