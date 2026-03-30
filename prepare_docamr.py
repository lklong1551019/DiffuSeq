import json
import re
import os

def mask_amr_relations(amr_string):
    # Regex to match AMR relation labels like :ARG0, :snt1, :ARG0-of, :time, :name
    # It catches a colon followed by word characters and optional hyphens/numbers
    relation_pattern = re.compile(r':[a-zA-Z0-9\-]+')
    
    # Replace all matches with the [MASK] token (standard for BERT vocab)
    masked_amr = relation_pattern.sub('[MASK]', amr_string)
    
    # Optional: clean up extra whitespaces/newlines to keep sequence length smaller
    masked_amr = re.sub(r'\s+', ' ', masked_amr).strip()
    return masked_amr

def process_docamr_dataset(input_file, output_file):
    """
    Reads a file containing raw DocAMR strings (where documents span multiple lines
    and are separated by an empty line) and writes the src/trg pairs into a jsonl file for DiffuSeq.
    """
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        current_doc_lines = []
        
        def process_doc(lines):
            comments = []
            graph_lines = []
            
            # Separate the metadata/comment lines from the actual graph lines
            for l in lines:
                if l.startswith('#'):
                    comments.append(l)
                else:
                    graph_lines.append(l)
            
            comment_text = " ".join(comments)
            graph_text = " ".join(graph_lines)
            
            if not graph_text:
                return

            # Only mask relations inside the actual AMR graph, leaving the metadata intact
            src_masked_graph = mask_amr_relations(graph_text)
            
            # Combine metadata and graph back together
            src_masked = f"{comment_text} {src_masked_graph}".strip()
            trg_original = f"{comment_text} {graph_text}".strip()
            
            # Clean up extra spaces to save sequence length
            src_masked = re.sub(r'\s+', ' ', src_masked).strip()
            trg_original = re.sub(r'\s+', ' ', trg_original).strip()
            
            json_record = {
                "src": src_masked,
                "trg": trg_original
            }
            fout.write(json.dumps(json_record) + '\n')

        # Read line by line. An empty line signifies the end of one document.
        for line in fin:
            clean_line = line.strip()
            if clean_line == "":
                if current_doc_lines:
                    process_doc(current_doc_lines)
                    current_doc_lines = []
            else:
                current_doc_lines.append(clean_line)
        
        # Make sure to process the final document if the file doesn't end with a blank line
        if current_doc_lines:
            process_doc(current_doc_lines)

if __name__ == "__main__":
    os.makedirs('datasets/docamr', exist_ok=True)
    
    # You would pass your processed txt/lines of DocAMR here:
    # process_docamr_dataset('raw_docamr_train.txt', 'datasets/docamr/train.jsonl')
    # process_docamr_dataset('raw_docamr_valid.txt', 'datasets/docamr/valid.jsonl')
    print("Dataset generation script ready to run.")
