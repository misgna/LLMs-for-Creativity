import torch
import numpy as np
import pandas as pd
import random
from utils import get_pairs
from prompts import get_user_prompt_template_for_essay_scores

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def call_model(prompt, model, tokenizer):   
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    device = model.device if hasattr(model, "device") else "cuda"
    
    inputs = tokenizer.apply_chat_template(
        prompt, 
        tokenize=True,
        add_generation_prompt=True, 
        return_tensors="pt", 
        return_dict=True,
        ).to(device)
    
    output = model.generate(
            **inputs,
            max_new_tokens=LENGTH,
            do_sample=True,
            temperature=TEMP, # Change temp to 0.7 or 0.3
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_token = output[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_token, skip_special_tokens=True)
    torch.cuda.empty_cache()
    explanation = response.split("assistant\n\n")[-1]
    
    return explanation

def compose_prompt(positive, negative):
    system_prompt = f"""
    You are an expert in essay creative writing assessment.
    
    ***Your task***:
    To extract the difference between two essay regarding their creativity.
    
    ***Instruction***:
    - Ignore content related information.
    - Use the trait scores as hint.
    - A bulleted list of differences between the two essays.
    """
    
    creative_essay = get_user_prompt_template_for_essay_scores(
        positive['essay'], 
        [
            positive['novelty'], 
            positive['richness'], 
            positive['logic'], 
            positive['originality'], 
            positive['coherence'], 
            positive['freshness'], 
            positive['appropriateness'], 
            positive['individuality'], 
            positive['persuasiveness']
        ]
    )
    
    noncreative_essay = get_user_prompt_template_for_essay_scores(
        negative['essay'], 
        [
            negative['novelty'], 
            negative['richness'], 
            negative['logic'], 
            negative['originality'], 
            negative['coherence'], 
            negative['freshness'], 
            negative['appropriateness'], 
            negative['individuality'], 
            negative['persuasiveness']
        ]
    )
    
    user_prompt = f"""
    Creative Essay:
    {creative_essay}
    
    Non-creative Essay:
    {noncreative_essay}

    """
    
    return [
        {'role': 'system', 'content': system_prompt}, 
        {'role': 'user', 'content': user_prompt}
        ]

def Explainer(model, tokenizer, seed, temp, length, nsamples):
    global TEMP
    global LENGTH
    TEMP = temp
    LENGTH = length
    
    #1. Load dataset
    train_path = "dataset/train.csv"
    dataset, sample = get_pairs(train_path, seed, nsamples)
    
    #3. Call compose prompt
    explanations = []
    n_iter = 1
    for idx, row in dataset.iterrows():
        for sidx, srow in sample.iterrows():
            prompt = compose_prompt(row, srow)
            explanation = call_model(prompt, model, tokenizer)
            explanations.append(explanation)
            
            #print(f"Explanation generation from pairs (%): {100*(n_iter/nsamples)}")
        print(f"Explanation generation from pairs (%): {100 * ((n_iter) / (nsamples)):.2f}")
        n_iter += 1
        

    #5. Save results
    output = pd.DataFrame({
            'explanation': explanations
        })
    model_name = model.config._name_or_path
    model_name = model_name.replace('.', '-').replace('/', '-')
    output.to_csv(f"output/explanations-by-{model_name}-{seed}-{temp}-{length}-{nsamples}.csv", index=False)
