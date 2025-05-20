from transformers import BertModel, BertTokenizer, MarianMTModel, MarianTokenizer

import os


bert_model_name = "bert-base-multilingual-cased"
marian_model_name = "Helsinki-NLP/opus-mt-ru-en"

save_path_bert = "./mbert_encoder"
save_path_marian = "./marian_decoder"

bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

marian_tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
marian_model = MarianMTModel.from_pretrained(marian_model_name)

os.makedirs(save_path_bert, exist_ok=True)
os.makedirs(save_path_marian, exist_ok=True)

bert_tokenizer.save_pretrained(save_path_bert)
bert_model.save_pretrained(save_path_bert)
print(f"mBERT сохранен в {save_path_bert}")


marian_tokenizer.save_pretrained(save_path_marian)
marian_model.save_pretrained(save_path_marian)
print(f"MarianMT сохранена в {save_path_marian}")
