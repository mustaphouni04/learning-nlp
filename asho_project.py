from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import torch
from torch import nn
import wandb
import json
from torch.optim import AdamW
from torchmetrics.text import SacreBLEUScore
from torch.utils.data import DataLoader
from _modules import ByT5Wrapper, Pipeline
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.environ.get('WANDB_API_KEY')
wandb.login(key=api_key)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

if os.path.exists("saved_models/mt5_synth_phase"):
    model = T5ForConditionalGeneration.from_pretrained("saved_models/mt5_synth_phase")
    tokenizer = AutoTokenizer.from_pretrained("saved_models/mt5_synth_phase")
    print("Loading from saved checkpoint!")
else:
    model = T5ForConditionalGeneration.from_pretrained("google/mt5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    print("Loading base model!")

train_loader = DataLoader(Pipeline("output.json", train=True), 
                          batch_size = 4, 
                          shuffle=True)

test_loader = DataLoader(Pipeline("output.json", train=False), 
                         batch_size = 1, 
                         shuffle=False)

wrapper = ByT5Wrapper(model).to(device)

optimizer = AdamW(wrapper.model.parameters())

num_epochs = 20

def train_loop(loader, lr:float, epochs:int, name:str):
    optimizer = AdamW(wrapper.model.parameters(), lr=lr)
    wandb.init(project="mt5-two-stage", name=name)
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"[{name}] Epoch {epoch+1}/{epochs}")
        for step,batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
    
            loss = wrapper(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
            loss.backward()
            optimizer.step()

            wandb.log({f"{name}_loss": loss.item()}, step=epoch*len(loader)+step)
            pbar.set_postfix(loss=loss.item())

#train_loop(train_loader, lr = 1e-3, epochs=num_epochs, name="synthetic")
   
#wrapper.model.save_pretrained("saved_models/mt5_synth_phase", safe_serialization=True)
#tokenizer.save_pretrained("saved_models/mt5_synth_phase")

with open("uab_summary_2024_all.json", "r") as f:
    real_data = json.load(f)

annotated = [real for real in real_data if real["Summary"] != ""]
print(f"There are {len(annotated)} annotated samples")

real_loader = DataLoader(Pipeline(annotated, train=True, test_size=0.1), 
                          batch_size = 4, 
                          shuffle=True)

train_loop(real_loader, lr=4e-5, epochs=num_epochs, name="real")

model.save_pretrained("saved_models/mt5_real_phase")
tokenizer.save_pretrained("saved_models/mt5_real_phase")

"""
dpo_data = []
for sample in Pipeline(annotated, train=True, test_size=0.1):
    # prepare single example
    enc = tokenizer(sample["Text"], return_tensors="pt",
                    truncation=True, padding="max_length", max_length=1024)
    in_ids, in_mask = enc.input_ids.to(device), enc.attention_mask.to(device)

    # generate two different summaries
    outs1 = wrapper.model.generate(**enc, max_length=1024, num_beams=4, top_k=50, do_sample=True)
    outs2 = wrapper.model.generate(**enc, max_length=1024, num_beams=4, top_k=50, do_sample=True)
    sum1 = tokenizer.decode(outs1[0], skip_special_tokens=True)
    sum2 = tokenizer.decode(outs2[0], skip_special_tokens=True)

    print("\n=== SOURCE TEXT ===")
    print(sample["Text"])
    print("\n[1]", sum1)
    print("[2]", sum2)
    choice = input("Which do you prefer? (1/2): ").strip()
    if choice not in ("1","2"):
        print("Skipping example.")
        continue

    good, bad = (outs1, outs2) if choice=="1" else (outs2, outs1)
    dpo_data.append({"input_ids":in_ids.cpu(),
                     "attention_mask": in_mask.cpu(), 
                     "good": good[0].cpu(), 
                     "bad": bad[0].cpu()
                     ))
torch.save(dpo_data, "dpo_training_data.pt")
"""

base_model = T5ForConditionalGeneration.from_pretrained("saved_models/mt5_real_phase")
tokenizer = AutoTokenizer.from_pretrained("saved_models/mt5_real_phase")
wrapper = ByT5Wrapper(base_model).to(device)
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
                max_length=256,
                num_beams=4,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2)

        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("Predicted: ", decoded_preds[0])
        decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print("Reference: ", decoded_refs[0])

        all_preds.extend(decoded_preds)
        all_refs.extend([[ref] for ref in decoded_refs])

score = bleu(all_preds, all_refs)
print(f"SacreBLEU score: {score:.4f}")
wandb.log({"sacrebleu": score})
