import json
from transformers import AutoTokenizer

def decode_ids_to_text(ids_list):
    """
    Decodes a list of token IDs back into readable text.
    """
    try:
        # Load the base tokenizer used in DiffuSeq
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        print("\nDecoding Sequence...")
        print("-" * 50)
        
        # Check for floats (embeddings) instead of integers (token IDs)
        sample_element = ids_list
        while isinstance(sample_element, list) and len(sample_element) > 0:
            sample_element = sample_element[0]
            
        if isinstance(sample_element, float):
            print("ERROR: You pasted a list of floating-point numbers (embeddings) instead of integer token IDs.")
            print("Older logs at the top of NaNLossInput.txt might contain these floats.")
            print("Please scroll to the BOTTOM of NaNLossInput.txt and copy a newer 'Raw IDs' array that contains integers like [101, 159, ...].")
            print("-" * 50)
            return
        
        # If it's a 2D list (batch of sequences)
        if isinstance(ids_list[0], list):
            for i, seq in enumerate(ids_list):
                text = tokenizer.decode(seq, skip_special_tokens=False)
                print(f"Sequence {i+1}:")
                print(text)
                print("-" * 50)
        else:
            # Single sequence
            text = tokenizer.decode(ids_list, skip_special_tokens=False)
            print("Sequence:")
            print(text)
            print("-" * 50)
            
    except Exception as e:
        print(f"Error decoding: {e}")

if __name__ == "__main__":
    # Paste your integer IDs here from NaNLossInput.txt (once you run the updated script)
    # Make sure to copy from the BOTTOM of the file where the IDs are integers (e.g. [[101, 159, ...]])
    # Example:
    # raw_ids = [[101, 1234, 5678, 102], [101, 4321, 8765, 102]]
    raw_ids = []
    
    if raw_ids:
        decode_ids_to_text(raw_ids)
    else:
        print("Please paste the integer IDs into the 'raw_ids' variable in this script.")
