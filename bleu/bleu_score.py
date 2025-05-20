from transformers import  BertTokenizer, MarianTokenizer, BertModel, MarianMTModel
from transformers.modeling_outputs import BaseModelOutput
from src.config import SAVE_PATH, MAX_LENGTH, SAMPLE_SIZE, DEVICE, BATCH_SIZE_TRAIN, BLUE_PATH
from src.dataset import load_data
from src.model import MBertToMarian

import evaluate
import torch


def generate_translations(inputs, model, tokenizer, batch_size=BATCH_SIZE_TRAIN):
    predictions = []
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    for i in range(0, len(input_ids), batch_size):
        input_ids_batch = input_ids[i:i+batch_size]
        mask_batch = attention_mask[i:i+batch_size]

        with torch.no_grad():
            bert_outputs = model.bert(input_ids=input_ids_batch, attention_mask=mask_batch)
            projected = model.projection(bert_outputs.last_hidden_state)

            generated_ids = model.marian.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=projected),
                attention_mask=mask_batch,
                max_length=MAX_LENGTH,
                num_beams=4,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


bert = BertModel.from_pretrained(SAVE_PATH + "/bert")
marian = MarianMTModel.from_pretrained(SAVE_PATH + "/marian")
model = MBertToMarian(bert, marian)

projection_weights = torch.load(SAVE_PATH + "/projection.pt", map_location=DEVICE)
model.projection.load_state_dict(projection_weights)

bert_tokenizer = BertTokenizer.from_pretrained(SAVE_PATH + "/tokenizer_bert")
marian_tokenizer = MarianTokenizer.from_pretrained(SAVE_PATH + "/tokenizer_marian")

model.to(DEVICE)
model.eval()

dataset = load_data(path=BLUE_PATH)

print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")

bleu = evaluate.load("bleu")

sample_size = SAMPLE_SIZE
random_sample = dataset['test'].shuffle(seed=42).select([i for i in range(sample_size)])
print(f"Sample size: {len(random_sample)}")


inputs = bert_tokenizer(
    list(random_sample['russian']), return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH
).to(DEVICE)


reference_sample = random_sample['khakas']

translations_decoded = generate_translations(inputs, model, marian_tokenizer)

for i in range(199):
    print(f"Source: {random_sample[i]['russian']}")
    print(f"Translate: {translations_decoded[i]}")
    print(f"Target: {random_sample[i]['khakas']}")
    print("-" * 50)

results_sample = bleu.compute(predictions=translations_decoded, references=[[ref] for ref in reference_sample])
print(f"BLEU score on sample: {results_sample['bleu']}")
