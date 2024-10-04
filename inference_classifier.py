import time
from concurrent.futures import ThreadPoolExecutor

import torch
import pandas as pd
from transformers import GemmaTokenizerFast, Gemma2ForSequenceClassification
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import PeftModel
from argparse import ArgumentParser

assert torch.cuda.device_count() == 2, "Sorry - multi-GPU required!"
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)


parser = ArgumentParser()
parser.add_argument(
    '--model_name',
    type=str,
)
parser.add_argument(
    '--model_path',
    type=str,
    default='None',
)
parser.add_argument(
    '--max_length',
    type=int,
    default=1700
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=4
)
cfg = parser.parse_args()
cfg.device = torch.device("cuda")
cfg.debug = False


if cfg.debug:
    test = pd.read_csv('/kaggle/input/lmsys-debug-set/test.csv')
else:
    test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')

test["prompt"] = test.prompt.map(lambda x: eval(x))
test["response_a"] = test.response_a.map(lambda x: eval(x.replace("null", "''")))
test["response_b"] = test.response_b.map(lambda x: eval(x.replace("null", "''")))


def get_text_multiturn(sample):
    turns = []
    for i, (prompt, response_a, response_b) in enumerate(zip(sample['prompt'], sample['response_a'], sample['response_b'])):
        turn = ''
        turn += f'<|user|> {prompt}'
        turn += f'\n\n<|model_a|> {response_a}'
        turn += f'\n\n<|model_b|> {response_b}'
        turns.append(turn)
    text = '\n\n'.join(turns)
    return text
    

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx].to_dict()
        text = get_text_multiturn(sample, self.tokenizer, self.max_length)
        tokenized = self.tokenizer(text, truncation=True, max_length=self.max_length)
        return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask']}
    

tokenizer = GemmaTokenizerFast.from_pretrained(cfg.model_name)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

    
dataset = CustomDataset(test, tokenizer, cfg.max_length)
input_ids, attention_mask = [], []
for i in range(len(dataset)):
    tokenized = dataset[i]
    input_ids.append(tokenized['input_ids'])
    attention_mask.append(tokenized['attention_mask'])
data = test[['id']].copy()
data['input_ids'] = input_ids
data['attention_mask'] = attention_mask
data["length"] = data["input_ids"].apply(len)
print(tokenizer.decode(data["input_ids"][0]))


device_0 = torch.device('cuda:0')
base_model_0 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.model_name,
    device_map=device_0,
    use_cache=False,
)
model_0 = PeftModel.from_pretrained(base_model_0, cfg.model_path).to(device_0)
model_0.eval()


device_1 = torch.device('cuda:1')
base_model_1 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.model_name,
    device_map=device_1,
    use_cache=False,
)
model_1 = PeftModel.from_pretrained(base_model_1, cfg.model_path).to(device_1)
model_1.eval()


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
    a_win, b_win, tie = [], [], []
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            max_length=max_length,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        outputs = model(**inputs.to(device))
        proba = outputs.logits.cpu() # .softmax(-1)
        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
        tie.extend(proba[:, 2].tolist())
    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    df["winner_tie"] = tie
    return df


st = time.time()

data = data.sort_values("length", ascending=False)
sub_1 = data.iloc[0::2].copy()
sub_2 = data.iloc[1::2].copy()

with ThreadPoolExecutor(max_workers=2) as executor:
    results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device_0, device_1))

result_df = pd.concat(list(results), axis=0)
submission_df = result_df[["id", 'winner_model_a', 'winner_model_b', 'winner_tie']]
submission_df.to_csv('submission.csv', index=False)
print(submission_df)