from transformers import WhisperProcessor, WhisperForConditionalGeneration 
from datasets import load_dataset
import aiohttp
import numpy as np 
import soundfile as sf
from dataclasses import dataclass
import ollama
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ollama.pull("qwen3:4b")

@dataclass
class AudioFile:
    path: str 
    array: np.ndarray
    sampling_rate: int

def save_audio_file(audio_sample: AudioFile):
    audio_file = AudioFile(path=audio_sample["path"], 
                           array=audio_sample["array"],
                           sampling_rate=audio_sample["sampling_rate"])
    if np.issubdtype(audio_file.array.dtype, np.floating):
        audio_array = (audio_file.array * 32767).astype(np.int16)  

    sf.write("output.wav", audio_array, sampling_rate, subtype='PCM_16')
    print("Saved as output.wav")
    return 0
     
class WhisperCaller:
    def __init__(self, model_name: str):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.config.forced_decoder_ids = None

    def generate_text_from_audio(self, sample: AudioFile):
        audio_file = AudioFile(path=sample["path"], 
                               array=sample["array"],
                               sampling_rate=sample["sampling_rate"])
        if len(audio_file.array.shape) > 1:
            audio_array = audio_file.array.mean(axis=0)
        else:
            audio_array = audio_file.array 


        input_features = self.processor(audio_array, 
                                        sampling_rate=audio_file.sampling_rate, 
                                        return_tensors="pt").input_features 
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, 
                                                    skip_special_tokens=True)

        return transcription

class AudioDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["audio"]


# load dataset and read audio files
ds = load_dataset("librispeech_asr", "clean", split="test",
                  storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
dataset = AudioDataset(ds)
print(f"[DEBUG] Length of librispeech dataset (test) is {len(dataset)}")


test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

#save_audio_file(sample) Usage: sample=ds[0]["audio"]

whisper = WhisperCaller("openai/whisper-large-v3")

for audio_sample in tqdm(test_loader, desc="Decoding and translating sentences!"):
    decoded = whisper.generate_text_from_audio(audio_sample)
    print(f"Decoded: {decoded}")

    response = ollama.chat(model='qwen3:4b', messages=[
        {
            'role': 'system',
            'content': 'You will always respond with German translations given user input. If the user speaks in English, you will translate everything he says in German. Exclusively limited to English to German translation. Do not use emojis or anything to answer user queries. If the user has a question in English respond with the same question translated back to German. Keep the translation natural to German language.',
        },
        {
            'role': 'user',
            'content': f'{decoded[0]}',
        },
    ])
    
    text = response['message']['content']
    tokens = text.split("\n")
    
    start_idx = 0
    for idx, tok in enumerate(tokens):
        if tok == "</think>":
            start_idx = idx
    
    translated = " ".join(tokens[start_idx+1:])
    print(f"Translated: {translated}")
    


