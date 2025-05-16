from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import aiohttp
import numpy as np 
import soundfile as sf

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model.config.forced_decoder_ids = None

# load dataset and read audio files
ds = load_dataset("librispeech_asr", "clean", split="test",
                  storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
sample = ds[0]["audio"]
audio_array = sample['array']
sampling_rate = sample['sampling_rate']
if np.issubdtype(audio_array.dtype, np.floating):
    audio_array = (audio_array * 32767).astype(np.int16)  

sf.write("output.wav", audio_array, sampling_rate, subtype='PCM_16')
print("Saved as output.wav")

print(sample)
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# generate token ids
predicted_ids = model.generate(input_features)

# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
#['<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.<|endoftext|>']

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
#[' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']
print(transcription)

