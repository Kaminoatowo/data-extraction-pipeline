"""
Utility functions for batch processing operations.
"""

from tqdm import tqdm
from typing import Generator, List, Tuple, Any, Callable
from scripts.utils.logger import setup_logger

logger = setup_logger("batch_processor")


def batch_files(files: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """
    Generator that yields batches of files.

    Args:
        files: List of files or data to batch
        batch_size: Size of each batch

    Yields:
        Batches of files
    """
    for i in range(0, len(files), batch_size):
        yield files[i : i + batch_size]


def process_batches(
    all_data: List[Any],
    batch_size: int,
    process_function: Callable,
    description: str = "Processing batches",
    **kwargs,
) -> Generator[Any, None, None]:
    """
    Process all batches and yield results.

    Args:
        all_data: List of data to process in batches
        batch_size: Size of each batch
        process_function: Function to process each batch
        description: Description for progress bar
        **kwargs: Additional arguments to pass to process_function

    Yields:
        Results from each batch processing
    """
    total_batches = (len(all_data) + batch_size - 1) // batch_size

    logger.info(f"Found {len(all_data)} items to process in {total_batches} batches")

    for batch_idx, batch in tqdm(
        enumerate(batch_files(all_data, batch_size), start=0),
        total=total_batches,
        desc=description,
    ):
        result = process_function(batch, batch_idx, **kwargs)
        if result:
            yield result


def process_batch_with_validation(
    batch: List[Any],
    batch_idx: int,
    process_function: Callable,
    validation_function: Callable = None,
    max_retries: int = 3,
    **kwargs,
) -> Any:
    """
    Process a single batch with optional validation and retry logic.

    Args:
        batch: The batch to process
        batch_idx: Index of the batch
        process_function: Function to process the batch
        validation_function: Optional function to validate results
        max_retries: Maximum number of retries on failure
        **kwargs: Additional arguments to pass to process_function

    Returns:
        Result from processing the batch
    """
    for attempt in range(max_retries):
        try:
            result = process_function(batch, batch_idx, **kwargs)

            if validation_function and not validation_function(result):
                logger.warning(
                    f"Batch {batch_idx + 1} validation failed on attempt {attempt + 1}"
                )
                if attempt < max_retries - 1:
                    continue
                else:
                    logger.error(
                        f"Batch {batch_idx + 1} failed validation after {max_retries} attempts"
                    )
                    return None

            return result

        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logger.error(
                    f"Batch {batch_idx + 1} failed after {max_retries} attempts"
                )
                return None

    return None
