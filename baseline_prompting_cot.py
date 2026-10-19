import torch
import pandas as pd
from datasets import Dataset
from utils import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_checkpoint
from evaluation import Evaluation
from prompts import get_example, get_system_prompt_template_for_feedback_generation, get_user_prompt_template_for_feedback_generation
import argparse

def compose_prompt(samples):
    system_prompt = f"""
    You are an expert writing evaluator specializing in creative assessment.
        
    Your task is to analyze an essay to determine whether the essay is creative, conventional, or non-creative.
        
    Structure your response exactly as follows:
        
    <thinking>
        1. Generate essay creativity evaluation elements.
        2. Evaluate the essay against the elements.
        3. Classify the essay to creative, conventional or non-creativite.
    </thinking>
        
    <answer>
        [creative OR conventional]
    </answer>
    """
    
    prompts = []
    
    for row in zip(samples['essay']):
        essay = row[0]
            
        user_prompt = f"""
            Essay: {essay}
            """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
            ]
        prompts.append(messages)
            
    return prompts

def call_model(samples, temp, length, model, tokenizer):
    prompts = compose_prompt(samples)
    
    # Configure padding for batched generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = model.device if hasattr(model, "device") else "cuda"
        
    # Format chat template with padding enabled for batches
    inputs = tokenizer.apply_chat_template(
        prompts, 
        tokenize=True,
        add_generation_prompt=True, 
        padding=True,
        return_tensors="pt", 
        return_dict=True,
    ).to(device)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=length,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # Slice out newly generated tokens for each sequence in the batch
    feedbacks = []
    for output in outputs:
        generated_tokens = output[input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        feedback = response.split("assistant\n\n")[-1].strip()
        feedbacks.append(feedback)
    
    return {"gen_feedback": feedbacks}

def get_feedback(model, tokenizer, seed, temp, length, out_path):
    batch = 1  # Can now be increased safely for faster processing
    
    # 1. Load dataset
    ds_filepath = "dataset/dev.csv" 
    test_nsamples = 25        
    
    dataset_df = get_dataset(ds_filepath, seed, test_nsamples)
    dataset_df = Dataset.from_pandas(dataset_df)

    # 4. Generate feedback and append new column
    feedback = dataset_df.map(
            call_model,
            batched=True,
            batch_size=batch,
            fn_kwargs={
                'temp': temp,
                'length': length,
                'model': model,
                'tokenizer': tokenizer,
            }
        )
        
    # 4. Save updated dataset to file
    feedback.to_pandas().to_csv(out_path, index=False)
    print(f"Feedback is created using!")
    print("-" * 20)
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
         
def parse_args():
    parser = argparse.ArgumentParser(description="Score2Feedback: Essay -> Feedback")
    parser.add_argument("--llm", type=str, default="llama3_8b", help="LLM type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temp", type=float, default=0.7, help="LLM temperature")
    parser.add_argument("--length", type=int, default=500, help="Output length")
    
    
    args = parser.parse_args()
    return args.llm, args.seed, args.temp, args.length
        
if __name__ == '__main__':
    llm, seed, temp, length = parse_args()
    
    checkpoint = get_checkpoint(llm)
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
        
    model_name = getattr(model.config, "_name_or_path", "model")
    model_name = model_name.replace('.', '-').replace('/', '-')
      
        
    file_suffix = f"{model_name}-{seed}-{temp}-{length}"
    out_path = f"output/feedback-by-{file_suffix}-prompt-cot.csv"
        
    get_feedback(model, tokenizer, seed, temp, length, out_path)
    
    print(f"Feedback is saved to {out_path}")
   
