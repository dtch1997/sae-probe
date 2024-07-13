#%%

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t
from utils import load_json, get_a_b_probs
from typing import Dict, Optional, List


#%% 
def tokenize_question(tokenizer, question, model_output):
    """
    Returns the tokenized input for each eval question to be passed into the model.
    """
    template = [{"role":"user", "content": question}, {"role":"assistant", "content": model_output}, {"role":"assistant", "content": "("}]
    tokens = tokenizer.apply_template(template, tokenize = True, return_tensors = "pt", add_generation_prompt = False)
    return tokens


def get_logits_from_text(
    model,
    tokens, 
) -> t.Tensor:
    
    with t.no_grad():
        logits = model(tokens).logits.to(device)
        print(f"{logits.shape=}")
    return logits

def get_a_b_probs(logits, a_token_id, b_token_id):
    last_token_logits = logits[0, -1, :]
    last_token_probs = t.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob


def process_item_ab(
    item: Dict[str, str],
    model: AutoModelForCausalLM,
    # system_prompt: Optional[str],
    a_token_id: int,
    b_token_id: int,
) -> Dict[str, str]:
    """
    Process an item from the AB test dataset.
    """
    question: str = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]

    tokens = tokenize_question(tokenizer, question, "(")
    logits = get_logits_from_text(model=model,
                                        tokens=tokens)
    a_prob, b_prob = get_a_b_probs(logits, a_token_id, b_token_id)

    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "choice": "(A)" if a_prob > b_prob else "(B)",
    }


def evaluate_dataset(
    dataset: List[Dict[str, str]],
    model: AutoModelForCausalLM,
    a_token_id: int,
    b_token_id: int,
    # system_prompt: Optional[str],
) -> List[Dict[str, str]]:
    """
    Evaluate the AB test dataset.
    """
    results = []
    for item in dataset:
        result = process_item_ab(item, model, a_token_id, b_token_id)
        results.append(result)
    return results

def summarize_results(results: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Summarize the results of the evaluation.
    """
    correct = 0
    total = len(results)
    for result in results:
        if result["choice"] == result["answer_matching_behavior"]:
            correct += 1
    accuracy = correct / total
    return {"accuracy": accuracy}
# %%

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    torch_dtype=t.bfloat16
)
device = t.device("cuda" if t.cuda.is_available() else "cpu")
dataset = load_json("datasets/test/sycophancy/test_dataset_ab.json")

a_token = tokenizer.encode("A")[0]
b_token = tokenizer.encode("B")[0]

results = evaluate_dataset(dataset=dataset, model=model, a_token_id=1, b_token_id=2)