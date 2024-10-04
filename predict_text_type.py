import numpy as np
import os
import pandas as pd
import random
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

class config:
    APEX = True
    BATCH_SIZE_TEST = 16
    DEBUG = False
    GRADIENT_CHECKPOINTING = True
    MAX_LEN = 512
    MODEL = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-large"
    NUM_CLASSES = 10
    NUM_WORKERS = 0
    PRINT_FREQ = 20
    SEED = 20


class paths:
    BEST_MODEL_PATH = "/kaggle/input/multiclass-numclasses-10"
    OUTPUT_DIR = "/kaggle/input/multiclass-numclasses-10"
    SUBMISSION_CSV = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
    TEST_CSV = "/kaggle/input/lmsys-chatbot-arena/test.csv"
    TRAIN_CSV = "/kaggle/input/valid-set-11k/valid_set.csv"
    
def seed_everything(seed=20):
    """Seed everything to ensure reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sep():
    print("-"*100)

seed_everything(seed=config.SEED)

test_df = pd.read_csv(paths.TEST_CSV, sep=',')
print(f"Test dataframe has shape: {test_df.shape}"), sep()

test_df['prompt'] = test_df['prompt'].apply(lambda x: str(x))
test_df['response_a'] = test_df['response_a'].apply(lambda x: str(x))
test_df['response_b'] = test_df['response_b'].apply(lambda x: str(x))
test_df['length'] = (test_df['prompt'] + test_df['response_a'] + test_df['response_b']).apply(lambda x: len(x.split()))

tokenizer = AutoTokenizer.from_pretrained(config.MODEL)
vocabulary = tokenizer.get_vocab()
total_tokens = len(vocabulary)
print("Total number of tokens in the tokenizer:", total_tokens)
print(tokenizer)

def prepare_input(cfg, text, tokenizer):
    """
    This function tokenizes the input text with the configured padding and truncation. Then,
    returns the input dictionary, which contains the following keys: "input_ids",
    "token_type_ids" and "attention_mask". Each value is a torch.tensor.
    :param cfg: configuration class with a TOKENIZER attribute.
    :param text: a numpy array where each value is a text as string.
    :return inputs: python dictionary where values are torch tensors.
    """
    inputs = tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=cfg.MAX_LEN,
        padding='max_length',
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


def collate(inputs):
    """
    It truncates the inputs to the maximum sequence length in the batch.
    """
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs


class CustomDataset(Dataset):
    def __init__(self, cfg, df, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.texts = df['prompt'].values
        self.id = df['id'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        output = {}
        output["inputs"] = prepare_input(self.cfg, self.texts[item], self.tokenizer)
        output["id"] = self.id[item]
        return output
    
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.MODEL, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.MODEL, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        if self.cfg.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()

        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, config.NUM_CLASSES)

    def _init_weights(self, module):
        """
        This method initializes weights for different types of layers. The type of layers
        supported are nn.Linear, nn.Embedding and nn.LayerNorm.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        """
        This method makes a forward pass through the model, get the last hidden state (embedding)
        and pass it through the MeanPooling layer.
        """
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        """
        This method makes a forward pass through the model, the MeanPooling layer and finally
        then through the Linear layer to get a regression value.
        """
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output
    
def inference_fn(test_df, model, device):
    test_dataset = CustomDataset(config, test_df, tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    prob_preds = []
    idx = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, batch in enumerate(tqdm_test_loader):
            inputs = batch.pop("inputs")
            ids = batch.pop("id")
            inputs = collate(inputs) # collate inputs
            for k, v in inputs.items():
                inputs[k] = v.to(device) # send inputs to device
            with torch.no_grad():
                y_preds = model(inputs) # forward propagation pass
                _, y_preds = torch.max(softmax(torch.tensor(y_preds)), dim=1)
            preds.append(y_preds.to('cpu').numpy()) # save predictions
            prob_preds.append(_.to('cpu').numpy())
            idx.append(ids)

    prediction_dict["predictions"] = np.concatenate(preds)
    prediction_dict["predictions_prob"] = np.concatenate(prob_preds)
    prediction_dict["id"] = np.concatenate(idx)
    
    test_df['preds'] = prediction_dict['predictions']
    test_df['preds_prob'] = prediction_dict['predictions_prob']
    return test_df

data = test_df.sort_values("length", ascending=False)
sub_1 = data.iloc[0::2].copy()
sub_2 = data.iloc[1::2].copy()

device_0 = torch.device('cuda:0')
model_0 = CustomModel(config, config_path=paths.BEST_MODEL_PATH + "/config.pth", pretrained=False)
state = torch.load(paths.BEST_MODEL_PATH + "/microsoft_deberta-v3-large_fold_0_best.pth")
model_0.load_state_dict(state["model"])
model_0.to(device_0)
model_0.eval()

device_1 = torch.device('cuda:1')
model_1 = CustomModel(config, config_path=paths.BEST_MODEL_PATH + "/config.pth", pretrained=False)
state = torch.load(paths.BEST_MODEL_PATH + "/microsoft_deberta-v3-large_fold_0_best.pth")
model_1.load_state_dict(state["model"])
model_1.to(device_1)
model_1.eval()

with ThreadPoolExecutor(max_workers=2) as executor:
    results = executor.map(inference_fn, (sub_1, sub_2), (model_0, model_1), (device_0, device_1))
    
result_df = pd.concat(list(results), axis=0)
result_df.to_csv('test_df_with_task.csv', index=False)