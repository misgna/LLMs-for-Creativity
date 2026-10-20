import torch

TEMP = 0.7
LENGTH = 1000

def compose_prompt(explanation, rubric):
    system_prompt = ""
    user_prompt = ""
    prompts = []
    
    system_prompt = f"""
    You are an expert on essay writing assessment.

    ***Your task***:
    Given explanations based on the difference between two essays and previously constructed rubric, update the rubric to accomodate the new explanation. 

    ***Instructions***:
    - Focus on essay quality traits.
    - Ignore content related information.
    - Update the rubric if there is a new criteria learned, otherwise do not change the rubric.
    - Make the rubric descriptive and short.
    - Use 1-5 score range, 1 for lower quality and 5 for high quality.
    """
    
            
    user_prompt = f"""
        ***Explanation***:
        {explanation}

        ***Rubric***:
        {rubric}
        """

    messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
    prompts.append(messages)

    return prompts

def call_model(model, tokenizer, explanation, rubric, temp, max_length):
    messages = compose_prompt(explanation, rubric)
    
    # Format chat prompt directly
    prompt_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and push directly to model device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # Run under inference_mode to disable autograd tracking completely
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=(temp > 0),
            temperature=temp if temp > 0 else None,
            top_p=0.9 if temp > 0 else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated continuation tokens
    prompt_len = inputs["input_ids"].shape[-1]
    generated_tokens = output[0][prompt_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return {"rubric": [response]}

def Contra2Rubric(model, tokenizer, temp: float, length: int, explanation: str, rubric: str):
    return call_model(
        model=model,
        tokenizer=tokenizer,
        explanation=explanation,
        rubric=rubric,
        temp=temp,
        max_length=length
    )
