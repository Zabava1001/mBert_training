from transformers import MarianMTModel, MarianTokenizer, BertModel, BertTokenizer
from config import MBERT_PATH, MARIAN_PATH, SAVE_PATH, device


def load_model():
    bert_tokenizer = BertTokenizer.from_pretrained(MBERT_PATH)
    bert_model = BertModel.from_pretrained(MBERT_PATH).to(device)

    marian_tokenizer = MarianTokenizer.from_pretrained(MARIAN_PATH)
    marian_model = MarianMTModel.from_pretrained(MARIAN_PATH).to(device)

    print(f"Модели загружены на: {device}")

    return {
        "bert_tokenizer": bert_tokenizer,
        "bert_model": bert_model,
        "marian_tokenizer": marian_tokenizer,
        "marian_model": marian_model
    }


def save_model(model, tokenizer):
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)



if __name__=='__main__':
    load_model()
