import os
import re
from collections import Counter

def extract_vocabularies():
    # Define the path to the processed text directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    text_dir = os.path.join(root_dir, "data", "dataset_processed", "text")
    output_file = os.path.join(root_dir, "vocabularies.txt")
    
    if not os.path.exists(text_dir):
        print(f"Directory not found: {text_dir}")
        return
    
    # Initialize a Counter to keep track of vocabulary occurrences
    vocab_counter = Counter()

    # Loop through all files in the directory
    for file_name in os.listdir(text_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(text_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read content and normalize by lowercasing
                content = file.read().lower()
                # Extract words (splitting by whitespace or parentheses)
                words = re.findall(r'\b[a-z]+(?:\'[a-z]+)?\b', content)
                vocab_counter.update(words)
    
    # Write the vocabulary and their occurrences to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for vocab, count in vocab_counter.most_common():
            file.write(f"{vocab} {count}\n")
    
    print(f"Vocabulary extraction complete. Output written to {output_file}")

if __name__ == "__main__":
    extract_vocabularies()
