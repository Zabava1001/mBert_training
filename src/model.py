from transformers import MarianMTModel, MarianTokenizer, BertModel, BertTokenizer
from config import MBERT_PATH, MARIAN_PATH, SAVE_PATH, device

import torch
import torch.nn as nn


class MBertToMarian(nn.Module):
    def __init__(self, bert_model, marian_model):
        super().__init__()
        self.bert = bert_model
        self.marian = marian_model

        self.projection = nn.Linear(self.bert.config.hidden_size, self.marian.config.d_model)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        encoder_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state

        projected_hidden = self.projection(hidden_states)

        marian_output = self.marian(
            encoder_outputs=(projected_hidden,),
            attention_mask=attention_mask,
            labels=labels
        )

        return marian_output


def load_model():
    bert_tokenizer = BertTokenizer.from_pretrained(MBERT_PATH)
    bert_model = BertModel.from_pretrained(MBERT_PATH).to(device)

    marian_tokenizer = MarianTokenizer.from_pretrained(MARIAN_PATH)
    marian_model = MarianMTModel.from_pretrained(MARIAN_PATH).to(device)

    model = MBertToMarian(bert_model, marian_model).to(device)

    print(f"Модели загружены на: {device}")

    return {
        "model": model,
        "bert_tokenizer": bert_tokenizer,
        "marian_tokenizer": marian_tokenizer,
    }


def save_model(model, tokenizer=None):
    model.save_pretrained(SAVE_PATH)
    if tokenizer:
        tokenizer.save_pretrained(SAVE_PATH)


if __name__=='__main__':
    load_model()
