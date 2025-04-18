from transformers import  BertTokenizer, MarianTokenizer, BertModel, MarianMTModel
from transformers.modeling_outputs import BaseModelOutput
import torch

from src.model import MBertToMarian
from src.config import DEVICE, MAX_LENGTH, SAVE_PATH


def load_model_custom():
    bert = BertModel.from_pretrained(SAVE_PATH + "/bert")
    marian = MarianMTModel.from_pretrained(SAVE_PATH + "/marian")

    model = MBertToMarian(bert, marian)

    projection_weights = torch.load(SAVE_PATH + "/projection.pt", map_location=DEVICE)
    model.projection.load_state_dict(projection_weights)

    bert_tokenizer = BertTokenizer.from_pretrained(SAVE_PATH + "/tokenizer_bert")
    marian_tokenizer = MarianTokenizer.from_pretrained(SAVE_PATH + "/tokenizer_marian")

    model.to(DEVICE)
    model.eval()

    return model, bert_tokenizer, marian_tokenizer


def translate(text, model, bert_tokenizer, marian_tokenizer):
    text = f">>khk<< {text}"

    inputs = bert_tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        bert_outputs = model.bert(**inputs)

        projected = model.projection(bert_outputs.last_hidden_state)

        generated_ids = model.marian.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=projected),
            attention_mask=inputs['attention_mask'],
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True
        )

    translated = marian_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return translated


if __name__=="__main__":
    model, bert_tok, marian_tok = load_model_custom()

    russian_text = "слышите , что говорит бог ?"

    khakas_translation = translate(russian_text, model, bert_tok, marian_tok)
    print(f"Перевод: {khakas_translation}")
