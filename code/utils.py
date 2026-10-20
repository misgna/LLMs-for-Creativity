import pandas as pd

def get_checkpoint(model_id):
    checkpoints = {
        "gpt": "openai/gpt-oss-20b",
        'gemma_7b':  "google/gemma-7b-it",
        'llama3_1b':  "meta-llama/Llama-3.2-1B-Instruct",
        'llama3_8b':  "meta-llama/Llama-3.1-8B-Instruct",
        'llama3_70b':  "meta-llama/Llama-3.3-70B-Instruct",
        'mistral_7b':  "mistralai/Mistral-7B-Instruct-v0.1",
        'mistral_8x7b':  "mistralai/Mixtral-8x7B-Instruct-v0.1", #Fails while loading checkpoint shards
        'falcon3_1b':  "tiiuae/Falcon3-1B-Instruct",
        'falcon3_7b':  "tiiuae/Falcon3-7B-Instruct",
        'falcon3_3b':  "tiiuae/Falcon3-3B-Instruct",
        'falcon3_10b':  "tiiuae/Falcon3-10B-Instruct",
        'deepseek_7b':  "deepseek-ai/deepseek-llm-7b-chat",
        'deepseek_67b':  "deepseek-ai/deepseek-llm-67b-chat",
        'deepseek_moe_16b':  "deepseek-ai/deepseek-moe-16b-chat",
        'phi3.5_mini':  "microsoft/Phi-3.5-mini-instruct",
        'phi3.5_moe':  "microsoft/Phi-3.5-MoE-instruct",
        'phi3_mini_4k':  "microsoft/Phi-3-mini-4k-instruct",
        'phi3_mini_medium_128k':  "microsoft/Phi-3-medium-128k-instruct",
        'phi3_small_128k':  "microsoft/Phi-3-small-128k-instruct",
        'phi3_small_8k':  "microsoft/Phi-3-small-8k-instruct",
        'qwen3_8b': "Qwen/Qwen3-8B", 
        
    }
    return checkpoints.get(model_id, f'{model_id} is not in the list!')

def get_pairs(filepath, seed, nsamples):
    
    df = pd.read_csv(filepath)
    
    highest = df[df['class'] == 1] # creative
    middle = df[df['class'] == 0]  # conventional
    lowest = df[df['class'] == -1] # non-creative
    
    positive_sample = highest.sample(n=nsamples, random_state=seed)
    middle_sample = middle.sample(n=nsamples, random_state=seed)
    negative_sample = lowest.sample(n=nsamples, random_state=seed)
    negative_sample_ext = pd.concat([middle_sample, negative_sample])
    
    print(f"#Positive sample: {len(positive_sample['total'].tolist())}")
    print(f"#Negative sample: {len(negative_sample_ext['total'].tolist())}")
    
    return positive_sample, negative_sample_ext
    
def get_dataset(filepath, seed, nsamples):
    df = pd.read_csv(filepath)
    sample = df.sample(n=nsamples, random_state=seed)
    return sample
