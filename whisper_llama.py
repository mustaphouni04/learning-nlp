from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import aiohttp
import numpy as np 
import soundfile as sf
from dataclasses import dataclass
import typing

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

        input_features = self.processor(audio_file.array, sampling_rate=audio_file.sampling_rate, return_tensors="pt").input_features 
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription

# load dataset and read audio files
ds = load_dataset("librispeech_asr", "clean", split="test",
                  storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
sample = ds[0]["audio"]
audio_array = sample['array']
sampling_rate = sample['sampling_rate']

save_audio_file(sample)

whisper = WhisperCaller("openai/whisper-large-v3")
print(whisper.generate_text_from_audio(sample))




