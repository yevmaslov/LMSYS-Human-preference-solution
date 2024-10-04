import vllm
import pandas as pd
from transformers import AutoTokenizer
from typing import List
from transformers import LogitsProcessor
import torch
import math
import numpy as np

model_path = '/kaggle/input/exp143-merged'

llm = vllm.LLM(
    model_path,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.95, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    max_model_len=2048,
)
tokenizer = llm.get_tokenizer()
tokenizer = AutoTokenizer.from_pretrained(model_path)

response_template = "Best response:"
def get_prompt(sample):
    sys_prompt = """Please read the following prompt and two responses. Determine which response is better.
If the responses are relatively the same, respond with 'T'. Otherwise respond with 'A' or 'B' to indicate which is better."""
    turns = []
    for i, (prompt, response_a, response_b) in enumerate(zip(sample['prompt'], sample['response_a'], sample['response_b'])):
        turn = ''
        turn += f'<|user|> {prompt}'
        turn += f'\n\n<|model_a|> {response_a}'
        turn += f'\n\n<|model_b|> {response_b}'
        turns.append(turn)
    turns = '\n\n'.join(turns)
    turns = turns[:1300*5]
    text = sys_prompt + '\n\n' + "#"*25 + "\n" + turns + '\n' + "#"*25 + f"\n\n{response_template}"
    return text


test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')

test["prompt"] = test.prompt.map(lambda x: eval(x))
test["response_a"] = test.response_a.map(lambda x: eval(x.replace("null", "''")))
test["response_b"] = test.response_b.map(lambda x: eval(x.replace("null", "''")))

test_temp = test.copy()
test['response_a'] = test_temp['response_b'].copy()
test['response_b'] = test_temp['response_a'].copy()

texts = []
for i in range(len(test)):
    row = test.iloc[i]
    text = get_prompt(row)
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    texts.append(text)
test['text'] = texts

choices = ["A", "B", "T"]

KEEP = []
for x in choices:
    c = tokenizer.encode(x, add_special_tokens=False)[0]
    KEEP.append(c)
print(f"Force predictions to be tokens {KEEP} which are {choices}.")

class DigitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.allowed_ids = KEEP
        
    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        scores[self.allowed_ids] += 100
        return scores
    
logits_processors = [DigitLogitsProcessor(tokenizer)]
responses = llm.generate(
    texts,
    vllm.SamplingParams(
        n=1,
        top_p=0.9,
        temperature=0,
        seed=777,
        skip_special_tokens=True,
        max_tokens=1,
        logits_processors=logits_processors,
        logprobs=5
    ),
    use_tqdm=True
)

results = []
errors = 0

for i,response in enumerate(responses):
    try:
        x = response.outputs[0].logprobs[0]
        logprobs = []
        for k in KEEP:
            if k in x:
                logprobs.append(x[k].logprob)
            else:
                logprobs.append(0)
                print(f"bad logits {i}")
        logprobs = np.array(logprobs)
        results.append(logprobs)
    except:
        results.append(np.array([1/3., 1/3., 1/3.]))
        errors += 1
        
print(f"There were {errors} inference errors out of {i+1} inferences")
results = np.vstack(results)

test['winner_model_a'] = results[:, 1]
test['winner_model_b'] = results[:, 0]
test['winner_tie'] = results[:, 2]
test = test[['id', 'winner_model_a', 'winner_model_b', 'winner_tie']]
test.to_csv('submission.csv', index=False)
