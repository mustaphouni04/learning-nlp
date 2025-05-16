from transformers import WhisperProcessor, WhisperForConditionalGeneration

from datasets import load_dataset

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
model.config.forced_decoder_ids = None

# load dataset and read audio files
ds = load_dataset("librispeech_asr", "clean", split="test")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# generate token ids
predicted_ids = model.generate(input_features)

# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
#['<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.<|endoftext|>']

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
#[' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']
print(transcription)

