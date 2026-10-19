import argparse
import pandas as pd
import torch
import numpy as np
import random

from transformers import AutoModelForCausalLM, AutoTokenizer

from pathlib import Path
import pandas as pd
from summarizer import Summarizer
from explainer import Explainer
from contra2rubric import Contra2Rubric
from utils import get_checkpoint

TEMP = 0.7
LENGTH = 500
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_and_tokenizer(checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if torch.cuda.is_available():
        print("Running on GPU")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="balanced",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    else:
        print("Running on CPU")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float32,
        )
        
    #print(f"Device: {model.hf_device_map}")

    return model, tokenizer
  
def main():
    parser = argparse.ArgumentParser(description="Clustering features")
    parser.add_argument('--llm', type=str, default="llama3_8b", help="Large language model type")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--temp', type=float, default=0.7, help="Select temp for model")
    parser.add_argument('--length', type=int, default=500, help="Output length")
    parser.add_argument('--nsamples', type=int, default=10, help="Sample size")
    
    args = parser.parse_args()
    return args.llm, args.seed, args.temp, args.length, args.nsamples

if __name__ == '__main__':
    model_args, seed, temp, length, nsamples = main()
    set_seed(seed)
    
    #2. Load tokenizer and model
    checkpoint = get_checkpoint(model_args)
    model, tokenizer = load_model_and_tokenizer(checkpoint)
    
   # Format model identifier for file paths
    model_name = getattr(model.config, "_name_or_path", "model")
    model_name = model_name.replace(".", "-").replace("/", "-")
    #formatted_temp = str(temp).replace(".", "_")
   
    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
   
    # 1. Load or generate explanations
    v_path = output_dir / f"explanations-by-{model_name}-{seed}-{temp}-{length}-{nsamples}.csv"
    out_path = output_dir / f"rubrics-by-{model_name}-{seed}-{temp}-{length}-{nsamples}-cobra.csv"
    
    print(f"v_path: {v_path}")
          
    if v_path.exists():
           print("Loading explanations...")
    else:
           print("Generating explanations...")
           Explainer(model, tokenizer, seed, temp, length, nsamples)
   
    df = pd.read_csv(v_path)
    explanations = df["explanation"].tolist()
    
    # 2. Summarize explanations
    summary = Summarizer(model, tokenizer, temp, length, explanations)
    print(summary)
    print("-" * 20)
    
    # 3. Construct rubric
    global_rubric = ""
    rubric_result = Contra2Rubric(model, tokenizer, temp, length, summary, global_rubric)["rubric"]
    print(rubric_result)
    print("-" * 20)
    
    # Ensure rubric is iterable for DataFrame creation if returned as a string
    rubric_data = rubric_result if isinstance(rubric_result, list) else [rubric_result]
    
    # 4. Save generated rubric
    pd.DataFrame({"rubric": rubric_data}).to_csv(out_path, index=False)
    print("Rubric is created using CoBRA!")
    print("-" * 20)