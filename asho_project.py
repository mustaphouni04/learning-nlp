from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import torch
from torch import nn
import wandb
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

num_epochs = 1

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

train_loop(train_loader, lr = 1e-3, epochs=num_epochs, name="synthetic")
    
wrapper.model.save_pretrained("saved_models/mt5_synth_phase", safe_serialization=True)
tokenizer.save_pretrained("saved_models/mt5_synth_phase")

base_model = T5ForConditionalGeneration.from_pretrained("saved_models/mt5_synth_phase")
tokenizer = AutoTokenizer.from_pretrained("saved_models/mt5_synth_phase")
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
