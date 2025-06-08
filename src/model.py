from transformers import MarianMTModel, MarianTokenizer, BertModel, BertTokenizer

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MBERT_PATH, MARIAN_PATH, SAVE_PATH, DEVICE

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

    @classmethod
    def from_pretrained_custom(cls, bert_path, marian_path):
        bert = BertModel.from_pretrained(bert_path)
        marian = MarianMTModel.from_pretrained(marian_path)
        return cls(bert, marian)


def load_model():
    bert_tokenizer = BertTokenizer.from_pretrained(MBERT_PATH)
    bert_model = BertModel.from_pretrained(MBERT_PATH).to(DEVICE)

    marian_tokenizer = MarianTokenizer.from_pretrained(MARIAN_PATH)
    marian_model = MarianMTModel.from_pretrained(MARIAN_PATH).to(DEVICE)

    model = MBertToMarian(bert_model, marian_model).to(DEVICE)

    print(f"Модели загружены на: {DEVICE}")

    return model, bert_tokenizer, marian_tokenizer


def save_model(model, tokenizer1, tokenizer2):
    model.bert.save_pretrained(SAVE_PATH + "/bert", safe_serialization=False)
    model.marian.save_pretrained(SAVE_PATH + "/marian", safe_serialization=False)
    torch.save(model.projection.state_dict(), SAVE_PATH + "/projection.pt")

    tokenizer1.save_pretrained(SAVE_PATH + "/tokenizer_bert")
    tokenizer2.save_pretrained(SAVE_PATH + "/tokenizer_marian")


if __name__=='__main__':
    load_model()
