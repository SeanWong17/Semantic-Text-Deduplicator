import argparse
from text_deduplicator import TextDeduplicatorWithFAISS

def main():
    parser = argparse.ArgumentParser(description="Deduplicate large text files based on semantic similarity using Transformers and FAISS.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input .jsonl file.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the deduplicated .jsonl file.")
    parser.add_argument("--text-key", type=str, default="output", help="The key in the JSON object that contains the text to be processed.")
    parser.add_argument("--model-name", type=str, default="bert-base-chinese", help="Name of the pre-trained model from Hugging Face Hub.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Similarity threshold for deduplication (e.g., 0.95).")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu'). Autodetects if not specified.")
    
    args = parser.parse_args()

    print("Initializing deduplicator...")
    deduplicator = TextDeduplicatorWithFAISS(model_name=args.model_name, device=args.device)
    
    deduplicator.process_and_save(
        input_path=args.input_file,
        output_path=args.output_file,
        text_key=args.text_key,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()
