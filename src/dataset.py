from datasets import Dataset
from transformers import MarianTokenizer, BertTokenizer
from src.config import DATA_PATH, MBERT_PATH, MARIAN_PATH, MAX_LENGTH, TEST_SIZE

import pandas as pd


def load_data(path=DATA_PATH):
    df = pd.read_excel(path, header=None)
    df = df.iloc[:, :2]

    df.columns = ["khakas", "russian"]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df['khakas'] = df['khakas'].fillna('').astype(str)
    df['russian'] = df['russian'].fillna('').astype(str)

    df["russian"] = ">>khk<< " + df['russian']

    return Dataset.from_pandas(df).train_test_split(test_size=TEST_SIZE)


def tokenize_data(dataset):
    bert_tokenizer = BertTokenizer.from_pretrained(MBERT_PATH)
    marian_tokenizer = MarianTokenizer.from_pretrained(MARIAN_PATH)

    def preprocess_function(examples):
        model_inputs = bert_tokenizer(examples["russian"], max_length=MAX_LENGTH, truncation=True, padding="max_length")
        labels = marian_tokenizer(examples["khakas"], max_length=MAX_LENGTH, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True)


if __name__=='__main__':
    dataset = load_data()
    data = tokenize_data(dataset)
