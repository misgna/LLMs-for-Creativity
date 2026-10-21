def get_score_template(scores):
    return f"""
        **Novelty:** {scores[0]}, 
        **Richness:** {scores[1]}, 
        **Logic:** {scores[2]}, 
        **Originality:** {scores[3]},
        **Coherence:** {scores[4]},
        **Freshness:** {scores[5]},
        **Appropriateness:** {scores[6]}, 
        **Individuality:** {scores[7]}, 
        **Persuasiveness:** {scores[8]}
"""


def get_system_prompt_template_for_essay_scores():
    return f"""
    You are an expert in essay creative writing assessment.
    
    ***Your task***:
    Given essay, and scores, construct a generic rubric for essay assessment.
    
    ***Instruction***:
    - Focus on essay quality traits.
    - Ignore content related information.
    - Use 1-5 score range, 1 for lower quality and 5 for high quality.
    """
    
def get_user_prompt_template_for_essay_scores(essay, scores):
    return f"""
    ***Essay***: 
    {essay}
    
    ***Scores***: 
    {get_score_template(scores)}
    """
    
def get_system_prompt_template_for_feedback_generation():
    return f"""
    You are an expert in essay creative writing assessment.
    
    ***Your task***:
    To evaluate the given essay and provide detail feedback and scores based on the provided rubric.
    
    ***Instructions***:
    - Strictly follow the rubric when evaluating the essay.
    - Support your feedback by using relevant examples from the essay.
    - DO NOT include your reasoning process.
    """

def get_user_prompt_template_for_feedback_generation(essay, rubric):
    return f"""
    ***Essay***:
    {essay}
    
    ***Rubric***:
    {rubric}
    """
    

def get_chat_template(model_name, user_prompt, system_prompt):
    chat_template_models = ['meta-llama-Llama-3-1-8B-Instruct', 'mistral_7b', 'falcon3_7b', 'falcon3_10b', 'llama3_8b']
    if model_name in chat_template_models:
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    else:
        return f"""
        {system_prompt}
        \n\n
        {user_prompt}
        """