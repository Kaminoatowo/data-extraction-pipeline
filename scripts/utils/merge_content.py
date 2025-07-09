def merge_files(input_paths, output_path):
    """
    Merges the contents of two or more files into a single output file.

    Args:
        input_paths (list of str): List of file paths to merge.
        output_path (str): Path to the output file.
    """
    with open(output_path, "w", encoding="utf-8") as outfile:
        for path in input_paths:
            with open(path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())


# Example usage:
# merge_files(['file1.txt', 'file2.txt', 'file3.txt'], 'merged.txt')
