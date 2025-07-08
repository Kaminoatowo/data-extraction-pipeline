from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PDF_DIR = BASE_DIR / "data" / "pdfs"
OCR_OUTPUT_DIR = BASE_DIR / "data" / "ocr_output"
TEXT_INPUT_DIR = BASE_DIR / "data" / "input_texts"
OUTPUT_DIR = BASE_DIR / "data"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
