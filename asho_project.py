from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from torch import nn


class ByT5Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, 
                input_ids,
                attention_mask,
                labels):
        outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels)
        return outputs.loss


model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

model_inputs = tokenizer(
    "Life is like a box of chocolates.", padding="longest", return_tensors="pt"
)

labels = tokenizer(
    "La vida es como una caja de chocolatinas", padding="longest", return_tensors="pt"
)

decoded_text = tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True)

print(decoded_text)
print(model_inputs)

wrapper = ByT5Wrapper(model)
output = wrapper(**model_inputs, labels=labels.input_ids)

print(output)

