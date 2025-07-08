import argparse
from pathlib import Path

from scripts.utils.logger import setup_logger
from scripts.pdf.pdf_splitter import run_pdf_split

from scripts.ocr.ocr_runner import batch_ocr

# from scripts.splitter.splitter import split_text
# from scripts.rag.rag_builder import build_rag_data
# from scripts.cpt.cpt_generator import generate_cpt
# from scripts.equations.extract_equations import extract_equations
# from scripts.qa.qa_generator import generate_qa_pairs

logger = setup_logger("pipeline")


def run_pipeline(args):
    work_dir = Path(args.work_dir)
    split_output_dir = work_dir / "pdfs"
    split_output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_pdf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting pipeline with input PDF: %s", input_path)

    if args.split_only:
        logger.info("Running split-only mode.")
        run_pdf_split(input_path, split_output_dir, args.pages_per_chunk)
        return

    ocr_output_dir = work_dir / "ocr_output"

    # Step 1: OCR / PDF Text Extraction
    if args.run_ocr:
        logger.info("Running OCR...")
        pdf_chunks = sorted(split_output_dir.glob("*.pdf"))

        if not pdf_chunks:
            logger.warning(
                "No PDF chunks found in %s. Make sure splitting has been done.",
                split_output_dir,
            )
            return

        ocr_txt_paths = batch_ocr(pdf_chunks, ocr_output_dir)

        if ocr_txt_paths:
            logger.info("OCR completed. Text files saved to: %s", ocr_output_dir)
        else:
            logger.error("OCR returned no output. Check model or PDF inputs.")
            return
    else:
        ocr_txt_files = sorted(ocr_output_dir.glob("*.txt"))

        if ocr_txt_files:
            logger.info("Loading existing OCR outputs from %s", ocr_output_dir)
            combined_text = "\n\n".join(
                f"# {txt_file.name}\n{txt_file.read_text(encoding='utf-8')}"
                for txt_file in ocr_txt_files
            )
            text_path = output_dir / "extracted_text.txt"
            text_path.write_text(combined_text, encoding="utf-8")
            logger.info("Combined text saved to %s", text_path)
        else:
            logger.error(
                "No existing OCR text files found in %s. Run OCR first or provide input.",
                ocr_output_dir,
            )
            return

    # # Step 2: Split Text
    # if args.run_split:
    #     logger.info("Splitting text...")
    #     splits = split_text(text)
    #     split_path = output_dir / "splits.json"
    #     import json

    #     split_path.write_text(json.dumps(splits, indent=2, ensure_ascii=False))
    #     logger.info("Text splits saved to %s", split_path)
    # else:
    #     split_path = output_dir / "splits.json"
    #     if split_path.exists():
    #         import json

    #         splits = json.loads(split_path.read_text(encoding="utf-8"))
    #         logger.info("Loaded existing splits from %s", split_path)
    #     else:
    #         logger.error("Splits not found. Run text splitter or provide manually.")
    #         return

    # # Step 3: RAG Data
    # if args.run_rag:
    #     logger.info("Generating RAG data...")
    #     rag_data = build_rag_data(splits, output_dir)
    #     logger.info("RAG data generated.")
    # else:
    #     logger.info("Skipping RAG generation.")
    #     rag_data = None

    # # Step 4: CPT
    # if args.run_cpt:
    #     logger.info("Generating CPT...")
    #     generate_cpt(splits, output_dir)
    #     logger.info("CPT generation complete.")
    # else:
    #     logger.info("Skipping CPT generation.")

    # # Step 5: Equation Extraction
    # if args.run_equations:
    #     logger.info("Extracting equations...")
    #     extract_equations(splits, output_dir)
    #     logger.info("Equation extraction complete.")
    # else:
    #     logger.info("Skipping equation extraction.")

    # # Step 6: QA Generation
    # if args.run_qa:
    #     logger.info("Generating Q&A...")
    #     generate_qa_pairs(splits, output_dir)
    #     logger.info("Q&A generation complete.")
    # else:
    #     logger.info("Skipping Q&A generation.")

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Pipeline from PDF to QA.")
    parser.add_argument(
        "--input_pdf", type=str, required=True, help="Path to the input PDF file."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to store outputs."
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        default="data",
        help="Base directory for all output files",
    )

    # Flags for partial execution
    parser.add_argument("--split_only", action="store_true", help="Only split PDF")
    parser.add_argument(
        "--run_split", action="store_true", help="Run PDF splitting step before OCR"
    )
    parser.add_argument(
        "--pages_per_chunk",
        type=int,
        default=10,
        help="Number of pages per split chunk",
    )

    parser.add_argument(
        "--run_ocr", action="store_true", help="Run OCR to extract text from PDF."
    )
    #    parser.add_argument("--run_split", action="store_true", help="Run text splitting.")
    parser.add_argument(
        "--run_rag", action="store_true", help="Run RAG data generation."
    )
    parser.add_argument("--run_cpt", action="store_true", help="Run CPT generation.")
    parser.add_argument(
        "--run_equations", action="store_true", help="Run equation extraction."
    )
    parser.add_argument(
        "--run_qa", action="store_true", help="Run question generation."
    )

    parser.add_argument("--run_all", action="store_true", help="Run the full pipeline.")

    args = parser.parse_args()

    if args.run_all:
        args.run_ocr = True
        args.run_split = True
        args.run_rag = True
        args.run_cpt = True
        args.run_equations = True
        args.run_qa = True

    run_pipeline(args)
