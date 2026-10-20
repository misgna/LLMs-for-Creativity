import inspect
from typing import Any, List, Union
import torch


def _dedent(text: str) -> str:
    """Helper to remove leading indentation and surrounding whitespace."""
    return inspect.cleandoc(text)


def compose_prompt(explanations: Union[str, List[str]]) -> List[dict[str, str]]:
    system_prompt = _dedent(
        """
        You are an expert on essay writing assessment.

        ***Your task***:
        Given explanations based on the difference between two essays, summarize the explanations.

        ***Instructions***:
        - Focus on essay quality traits.
        - Ignore content related information.
        - Capture the most number of explanation categories.
        - Generate a summary based on the explanations.
        """
    )

    if isinstance(explanations, (list, tuple)):
        explanations_text = "\n\n".join(f"- {exp.strip()}" for exp in explanations)
    else:
        explanations_text = str(explanations).strip()

    user_prompt = _dedent(
        f"""
        ***Explanations***:
        {explanations_text}
        """
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_model(
    model: Any,
    tokenizer: Any,
    explanations: Union[str, List[str]],
    temp: float,
    length: int,
) -> str:
    messages = compose_prompt(explanations)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = model.device if hasattr(model, "device") else "cuda"

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    generation_kwargs = {
        "max_new_tokens": length,
        "do_sample": temp > 0,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temp > 0:
        generation_kwargs.update({"temperature": temp, "top_p": 0.9})

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)

    prompt_len = inputs["input_ids"].shape[-1]
    generated_tokens = output[0][prompt_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    summary = response.split("assistant\n\n")[-1].strip()
    return summary


def Summarizer(
    model: Any,
    tokenizer: Any,
    temp: float,
    length: int,
    explanations: List[str],
    batch_size: int = 4,
) -> str:
    print(f"Number of explanations to summarize: {len(explanations)}")

    def recursive_summarize(texts: List[str]) -> str:
        if len(texts) <= batch_size:
            return call_model(
                model=model,
                tokenizer=tokenizer,
                explanations=texts,
                temp=temp,
                length=length,
            )

        summaries = []
        for idx in range(0, len(texts), batch_size):
            batch = texts[idx : idx + batch_size]
            summary = call_model(
                model=model,
                tokenizer=tokenizer,
                explanations=batch,
                temp=temp,
                length=length,
            )
            summaries.append(summary)

        return recursive_summarize(summaries)

    return recursive_summarize(explanations)