from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DEBUG_MODE = False

RAW_PDF_DIR = BASE_DIR / "data" / "pdfs"
OCR_OUTPUT_DIR = BASE_DIR / "data" / "ocr_output"
TEXT_INPUT_DIR = BASE_DIR / "data" / "input_texts"
OUTPUT_DIR = BASE_DIR / "data"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")  # Default to gpt-4o-mini if not set

# Initialize OpenAI client (replace with your method of auth)
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing or invalid.")

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)  # OpenAI using ollama models
# openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    raise ValueError(f"Error configuring OpenAI client: {e}")
