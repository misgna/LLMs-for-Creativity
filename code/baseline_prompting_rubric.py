import torch
import pandas as pd
from datasets import Dataset
from utils import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_checkpoint
from evaluation import Evaluation
from prompts import get_example, get_system_prompt_template_for_feedback_generation, get_user_prompt_template_for_feedback_generation
import argparse

def get_rubric_r2c():
    return f"""
        # Content
        ## Novelty of ideas and content
        - The idea is ingenious.
        - The argument or evidence is fresh and novel.
        
        ## Richness of content
        - The content is varied, including diverse evidence and examples
        - The content is specific.
        
        ## Logicality of content
        - The evidence is valid and reasonable.
        - The author responds appropriately, taking into account expected counter arguments.
        
        # Organization
        ## Orginality of structure
        - The narative approach is unique.
        - The introduction and conclusion have been impressively structured.
        
        ## Cohesiveness of structure
        - Paragraphs are well separated, and the structure is systematic.
        - The text flows smoothly and is cohesive.
        
        # Expression
        ## Originality of expression
        - The expression is original and not clicheed, featuring creative metaphors, witty quotations, and literary expressions.
        
        ## Appropriateness of expression
        - The author uses accurate and objective words that fit the grammar.
        
        # Author's voice
        ## Perspective and personality
        - The author's new perspective on the topic is revealed.
        - The author's personality is revealed through his/her writing style and other elements.
        
        # Readers' response
        ## Fun and persuasiveness
        - The writing is fun and interesting.
        - The author's argument is persuasive and the reader is impressed.
"""

def get_rubric():
    return f"""
        ## Originality:
        - [strict]: The essay must present novel or uncommon ideas that go beyond conventional or predictable responses to the topic.
        - [strict]: The essay must demonstrate originality in its ideas or interpretations, not merely through unusual wording or stylistic variations.
        - [weak]: The essay may challenge common assumptions or introduce unconventional interpretations.
        - [weak]: The essay may approach familiar topics from a distinctive or unexpected edge.
        
        ## Idea diversity
        - [strict]: The essay must explore multiple distinct ideas, perspectives, or approaches relevant to the topic.
        - [strict]: the essay must demonstrate meaningful variations between ideas, rather than repetition, paraphrasing, or minor variations of the same idea.
        - [weak]: The essay may consider opposing viewpoints or alternative explanations.
        - [weak]: The essay may explore different dimensions of the topic, such as social, psychological, cultural, or ethical implications.
        
        ## Associative thinking
        - [strict]: The essay must establish meaningful connections between different concepts, domains, or experiences.
        - [strict]: The connections must be logically meaningful rather than arbitrary associations or superficial comparisons.
        - [weak]: The essay may connect concepts from seemingly unrelated fields or disciplines.
        - [weak]: The essay must use analogies, metaphors, or cross-domain examples to communicate complex ideas.
        
        ## Insightfulness
        - [strict]: The essay must present meaningful observations, interpretations, or conclusions that deepen understanding of the topic.
        - [strict]: The insights must be relevant and logically supported rather than generic observations or unsupported claims.
        - [weak]: The essays may reveal unexpected patterns, hidden assumptions, or overlooked relationships.
        - [weak]: The essay may reframe a familiar problem in a way that encourages readers to reconsider their existing understanding.
        
        ## Elaboration and development
        - [strict]: The essay must develop its creative ideas through substantive reasoning, explanations, examples, or implications.
        - [strict]: The development must add meaningful depth rather that merely increasing length through repetition or irrelevant details.
        - [weak]: The essay may explore the potential consequences or broader implications of its ideas.
        - [weak]: The essay may use illustrative scenarios, thought experiments, or concrete examples to develop its ideas.
        
"""

def compose_prompt(samples, criteria):
    system_prompt = f"""
    You are an expert in essay creative writing assessment.
        
    ***Your task***:
    To evaluate the given essay and provide detail feedback, scores and classify the essays into CREATIVE, CONVENTIONAL and NON-CREATIVE.
        
    ***Instructions***:
    - Support your feedback by using relevant examples from the essay.
    - The classification classes are CREATIVE, COVENTIONAL, or NON-CREATIVE.
    - Use the criteria for creativity as a guide for classification.
    - Do not include your reasoning process.
    """
    
    prompts = []
    
    for row in zip(samples['essay']):
        essay = row[0]
            
        user_prompt = f"""
            Essay: {essay}
            Criteria: {get_rubric() if criteria == 'generated' else get_rubric_r2c()}
            """
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
            ]
        prompts.append(messages)
            
    return prompts

def call_model(samples, temp, length, model, tokenizer, criteria):
    prompts = compose_prompt(samples, criteria)
    
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

def get_feedback(model, tokenizer, seed, temp, length, out_path, criteria):
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
                'criteria': criteria
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
    parser.add_argument("--criteria", type=str, default="generated", help="Select classification criteria: generated, or r2c")
    
    
    args = parser.parse_args()
    return args.llm, args.seed, args.temp, args.length, args.criteria
        
if __name__ == '__main__':
    llm, seed, temp, length, criteria = parse_args()
    
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
    out_path = f"output/feedback-by-{file_suffix}-{criteria}.csv"
        
    get_feedback(model, tokenizer, seed, temp, length, out_path, criteria)
    
    print(f"Feedback is saved to {out_path}")
   
