
import os
import random
import pandas as pd
import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers import (
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

import warnings
warnings.filterwarnings('ignore')

from types import SimpleNamespace
from copy import deepcopy
from sklearn.metrics import log_loss
from scipy.special import softmax
from pathlib import Path
import yaml
import wandb
wandb.login()

import huggingface_hub
huggingface_hub.login()

def namespace_to_dictionary(data):
    dictionary = vars(data)
    for k, v in dictionary.items():
        if type(v) is SimpleNamespace:
            v = namespace_to_dictionary(v)
        dictionary[k] = v
    return dictionary


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = softmax(logits, axis=1)
    loss = log_loss(labels, predictions)
    return {'log_loss': loss, }


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        
def get_config():
    config = SimpleNamespace()
    config.exp_name = 'exp207'
    config.exp_idx = int(config.exp_name[3:])
    config.job_type = 'train'
    config.project = 'lmsys-chat-arena'
    config.seed = 42
    config.out_dir = f'models/{config.exp_name}'

    config.max_length = 2000
    config.test_size = 0.05

    config.epochs = 1
    config.learning_rate = 2e-4
    config.batch_size = 1
    config.gradient_accumulation_steps = 16
    config.evaluate_n_times_per_epoch = 1
    config.weight_decay = 0.01
    config.max_grad_norm = 10

    config.model_name = "unsloth/gemma-2-9b-it-bnb-4bit"
    config.lora_dropout = 0.05
    config.lora_r = 16
    config.lora_alpha = 16
    config.freeze_layers = 0
    config.lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    config.additional_data_paths = [
        'data/additional/additional_data_v6.parquet',
    ]
    return config
    
         
def save_config(config, path):
    config_out = deepcopy(config)
    config_out.tokenizer = None
    config_out = namespace_to_dictionary(config_out)
    for key, value in config_out.items():
        if type(value) == type(Path()):
            config_out[key] = str(value)
    with open(path, 'w') as file:
        yaml.dump(config_out, file, default_flow_style=False)
             

def init_wandb(config):
    config_out = deepcopy(config)
    config_out = namespace_to_dictionary(config)
    run = wandb.init(
        project=config.project,
        job_type=config.job_type,
        name=config.exp_name,
        group=config.exp_name,
        config=config_out,
    )
    return run


def process_response_for_filtering(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return  ' '.join(sentences)


def get_text_multiturn(sample, tokenizer):
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
    def __init__(self, df, config, tokenizer, max_length, train):
        self.df = df
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx].to_dict()
        text = get_text_multiturn(sample, self.tokenizer, self.max_length, self.train)
        tokenized = self.tokenizer(text, truncation=True, max_length=self.max_length)
        label = np.argmax([sample['winner_model_a'], sample['winner_model_b'], sample['winner_tie']])
        return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask'], 'label': label}


if __name__ == '__main__':
    config = get_config()
    os.makedirs(config.out_dir, exist_ok=True)
    save_config(config, f'{config.out_dir}/config.yaml')
    set_seeds(seed=config.seed)

    run = init_wandb(config)
  
    train = pd.read_csv('data/raw/train.csv')
    train['response_a_temp'] = train['response_a'].apply(process_response_for_filtering)
    train['response_b_temp'] = train['response_b'].apply(process_response_for_filtering)
    train = train[~((train.response_a_temp == 'null') & (train.response_b_temp == 'null'))]

    train["prompt"] = train.prompt.map(lambda x: eval(x))
    train["response_a"] = train.response_a.map(lambda x: eval(x.replace("null", "''")))
    train["response_b"] = train.response_b.map(lambda x: eval(x.replace("null", "''")))

    train['rem'] = train['id'].apply(lambda x: x%5)
    train['valid'] = train['rem'] == 0
    train_df, valid_df = train.copy(), train[train['valid']].copy()

    for fp in config.additional_data_paths:
        before = train_df.shape[0]
        add = pd.read_parquet(fp)
        train_df = pd.concat([train_df, add])
        print(f'Adding data from: {fp}, shape before: {before}, shape after: {train_df.shape[0]}, additional samples: {add.shape[0]}')

    tokenizer = GemmaTokenizerFast.from_pretrained(config.model_name)
    tokenizer.add_eos_token = True
    tokenizer.padding_side = "right"

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = CustomDataset(train_df, config, tokenizer, config.max_length, train=True)
    valid_ds = CustomDataset(valid_df, config, tokenizer, config.max_length, train=False)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias='none',
        inference_mode=False,
        task_type='SEQ_CLS',
        target_modules=config.lora_modules, 
    )

    model = Gemma2ForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=3,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    eval_steps = len(train_ds) // config.batch_size // config.evaluate_n_times_per_epoch // config.gradient_accumulation_steps

    training_args = TrainingArguments(
        output_dir=config.out_dir,
        report_to="wandb",
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.batch_size,
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='steps',
        save_steps=50,
        optim="adamw_8bit",
        bf16=True,
        gradient_checkpointing=False,
        learning_rate = config.learning_rate,
        warmup_steps=20,
        save_total_limit=25,
        max_grad_norm=config.max_grad_norm,
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = valid_ds,
        tokenizer = tokenizer,
        data_collator = collate_fn,
        compute_metrics = compute_metrics,
    )

    trainer.train()
    trainer.save_model(config.out_dir)
