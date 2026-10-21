import argparse
import torch
import numpy as np
import random
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_checkpoint, get_dataset
from prompts import get_user_prompt_template_for_feedback_generation, get_system_prompt_template_for_feedback_generation, get_example

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compose_prompt(sample, rubric):
    system_prompt = ""
    user_prompt = ""
    prompts = []
    
    system_prompt = get_system_prompt_template_for_feedback_generation()
    for row in zip(sample['essay']):
        essay = row[0]
            
        user_prompt = get_user_prompt_template_for_feedback_generation(essay, rubric)
        messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        prompts.append(messages)
        
    return prompts

def call_model(model, tokenizer, samples, rubric, temp, length):
    prompts = compose_prompt(samples, rubric)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    inputs = tokenizer.apply_chat_template(
        prompts, 
        tokenize=True,
        add_generation_prompt=True, 
        return_tensors="pt", 
        return_dict=True,
        ).to(device)
    
    output = model.generate(
            **inputs,
            max_new_tokens=length,
            do_sample=True,
            temperature=temp, # Change temp to 0.7 or 0.3
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_token = output[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_token, skip_special_tokens=True)
    torch.cuda.empty_cache()
    feedback = response.split("assistant\n\n")[-1]
    
    return {"gen_feedback": [feedback]}

def main():
    parser = argparse.ArgumentParser(description="Feedback generator")
    parser.add_argument('--model', type=str, default="llama3_8b", help="Large language model type")
    parser.add_argument('--batch', type=int, default=1, help="Batch size")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--temp', type=float, default=0.7, help="Select temp for model")
    parser.add_argument('--length', type=int, default=500, help="Output length")
    parser.add_argument('--nsamples', type=int, default=10, help="Sample size")
    parser.add_argument('--rubric', type=str, default="cobra", help="Select rubric type cobra (batch aggregator)")
    
    args = parser.parse_args()
    return args.model, args.batch, args.seed, args.temp, args.length, args.nsamples, args.rubric

if __name__ == "__main__":
       # 0. Load args
    model_args, batch, seed, temp, length, nsamples, rub_type = main()
    set_seed(seed)
    
    # 1. load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = get_checkpoint(model_args)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="balanced")
    
    model_name = model.config._name_or_path
    model_name = model_name.replace('.', '-').replace('/', '-')

    # 2. Load dataset
    ds_filepath = "dataset/dev.csv"
    model_args_rub = "llama3_8b"
    test_nsamples = 125
    dataset = get_dataset(ds_filepath, seed, test_nsamples)
    dataset = Dataset.from_pandas(dataset)
    
    rub_filepath = f"output/rubrics-by-{model_name}-{seed}-{temp}-{length}-{nsamples}-{rub_type}.csv"
    rubric_set = pd.read_csv(rub_filepath)
    rubric = rubric_set.iloc[-1]['rubric']
    print(rubric)
    
    # 3. Rubric construction
    dataset_rubric = dataset.map(call_model, batched=True, batch_size=batch, fn_kwargs={'model': model, 'tokenizer': tokenizer, 'rubric': rubric, 'temp': temp, 'length': length})
    
    # 4. Save to file
    dataset_df = dataset_rubric.to_pandas()
    dataset_df.to_csv(f"output/feedback-by-{model_name}-{seed}-{temp}-{length}-{nsamples}-{rub_type}.csv", index=False)
