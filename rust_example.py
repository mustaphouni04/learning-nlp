from tokenizers import Tokenizer
from tokenizers.models import BPE
import pandas as pd
from transformers import GPT2Tokenizer
from tqdm import tqdm

# Load the data
df = pd.read_excel("ExercisesTest_filtered.xlsx")
summaries = df["summary"].values

tokenizer = Tokenizer.from_pretrained("gpt2")
for summary in tqdm(summaries):
    tokens = tokenizer.encode(str(summary)).tokens
