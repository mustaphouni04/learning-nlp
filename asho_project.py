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

model_inputs = tokenizer(
    train_pipe[1][0], padding="longest", return_tensors="pt"
)

labels = tokenizer(
    train_pipe[1][1], padding="longest", return_tensors="pt"
)

decoded_text = tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True) # shape is (1, seq_len) so take first elem

#print(decoded_text)
#print(model_inputs)

wrapper = ByT5Wrapper(model).to(device)
output = wrapper(**model_inputs, labels=labels.input_ids)

#print(output)

optimizer = AdamW(model.parameters())

wandb.init(
      project="byt5-finetuning",
      name=f"experiment")

for batch in tqdm(train_loader):
    model.train()
    texts, summaries = batch
    texts, summaries = texts.to(device), summaries.to(device)

    optimizer.zero_grad()

    loss = wrapper(**texts, labels=summaries)
    wandb.log({"loss": loss})

    loss.backward()
    optimizer.step()



