from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import torch
from torch import nn
import wandb
from torch.optim import AdamW
from torchmetrics.text import SacreBLEUScore
from torch.utils.data import DataLoader
from _modules import ByT5Wrapper, Pipeline

wandb.login()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

train_pipe = Pipeline("output.json", train=True)
train_loader = DataLoader(train_pipe, batch_size = 8, shuffle=True)

test_pipe = Pipeline("output.json", train=False)
test_loader = DataLoader(test_pipe, batch_size = 8, shuffle=False)

#decoded_text = tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True) # shape is (1, seq_len) so take first elem

#print(decoded_text)
#print(model_inputs)

wrapper = ByT5Wrapper(T5ForConditionalGeneration.from_pretrained("google/byt5-small")).to(device)
#output = wrapper(**model_inputs, labels=labels.input_ids)

#print(output)

optimizer = AdamW(model.parameters())

wandb.init(
      project="byt5-finetuning",
      name=f"experiment")

for batch in tqdm(train_loader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    optimizer.zero_grad()

    loss = wrapper(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    wandb.log({"loss": loss})

    loss.backward()
    optimizer.step()

wrapper.eval()
bleu = SacreBLEUScore()
all_preds = []
all_refs = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        generated_ids = wrapper.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4)

        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        all_preds.extend(decoded_preds)
        all_refs.extend([[ref] for ref in decoded_refs])

score = bleu(all_preds, all_refs)
print(f"SacreBLEU score: {score:.4f}")
wandb.log({"sacrebleu": score})

