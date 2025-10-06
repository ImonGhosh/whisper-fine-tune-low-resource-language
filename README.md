# Fine Tuning Whisper ASR Model on Low Resource Indian Languages (Bengali and Telugu)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![HF Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)]()

- Practical, reproducible pipeline to fine-tune OpenAI Whisper for any **low-resource language**.
- Demonstrates **significant WER reduction** and improved transcription quality **post fine-tuning** when fine-tuned for Bengali and Telugu.
- Comparative study of **four PEFT methods**: **LoRA**, **LoRA + SpecAugment**, **BitFit**, and **Adapter Layers**.
- **LoRA-based methods** achieve the **best WER** across both languages.
- For theory, setup, datasets, and full results, see the **Research Paper** (information given below).

## 📄Research Paper<br>
**Name** : BREAKING LANGUAGE BARRIERS: FINE-TUNING WHISPER FOR BENGALI AND TELUGU AUTOMATIC SPEECH RECOGNITION<br>
**Date** : April, 2025.<br>
**Authors** : Imon Kalyan Ghosh, Ishmita Basu, Bathula Veera Raghavulu<br>
**Check out the paper here** : [Research Paper Link](paper/Research_Paper.pdf)<br>
**Check out a short paper presentation here** : [Research Paper Link](paper/Paper_Presentation.pdf)

---

## ✨ Highlights
- **Problem**: Off-the-shelf Whisper underperforms on low resource languages like **Bengali**/**Telugu** due to limited labeled speech.
- **Approach**: Parameter-efficient fine-tuning (PEFT) variants + clean ASR dataset
- **Results**: **WER and transcription quality improved significantly** post FT; **LoRA > Adapters ≈ BitFit** in our setting.
- **Scale/Cost**: Runs on a single GPU with PEFT (fast, memory-efficient); mixed precision; configurable train/test splits.
- **Artifacts**: One-file CLI pipeline, HF-compatible checkpoints, example inference code.

---

## 🧠 Methods Compared
- **LoRA** (rank-limited adapters on attention/MLP)
- **LoRA + SpecAugment** (time/freq masking for robustness)
- **BitFit** (bias-only tuning)
- **Adapter Layers** (bottleneck adapters inserted into blocks)

> Conclusion: **LoRA-based** approaches consistently produced the **lowest WER** on Bengali/Telugu in our experiments.

---

## 🚀 Quickstart (Using our CLI Pipeline for fine-tuning any low resource language)
**Requirements**
```bash
# Create & activate env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip

# Core deps
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121   # pick CUDA/CPU as needed
pip install transformers datasets accelerate peft evaluate jiwer soundfile librosa tensorboard
```


### Download the pipeline file to your project root: (lora_finetuning_pipeline.py)

Run below CLI command from file location -

```bash
python lora_finetuning_pipeline.py \
  --dataset imonghose/bengali-asr-data \
  --language bengali \
  --username imonghose \
  --model_size small \
  --output_dir ./finetuned_whisper_bengali \
  --logging_dir ./tensorboard_logs \
  --train_frac 0.25 \
  --test_frac 0.75
```

--dataset, --language, --username, --output_dir, --train_frac, --test_frac are fully customizable.

Your custom fine-tuned model is pushed to your hugging face and saved under folder specified under --output_dir

Use a GPU for training: torch.cuda.is_available() should be True.


## ▶️ Using the Fine-Tuned Model (Inference)

```bash
# ----------- Load Audio File --------------
AUDIO_FILE = "bengali-convo-2.wav"  # replace with your file

# Preview audio
ipd.Audio(AUDIO_FILE)

# Load audio
waveform, sr = torchaudio.load(AUDIO_FILE)
waveform = waveform[0].numpy()  # mono
resampled = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
sr = 16000  # Whisper expects 16kHz


# -------------------Set model and tokenizer properties-----------------------------------

model_name_or_path = "openai/whisper-small"
language = "bengali"
task = "transcribe"

# -------------------Load Whisper tokenizer-----------------------------------

from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path,language=language,task=task)

#---------------------------------Load LORA model from Hugging Face Hub-----------------------------------

from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
config = LoraConfig(r=32, lora_alpha=64, target_modules=["k_proj", "v_proj", "q_proj", "out_proj"], lora_dropout=0.05, bias="none")

# Load base model
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
peft_model = get_peft_model(base_model, config)

# Load LoRA adapter
# Note : Replace our model with your own fine-tuned model if you use our CLI piepline to create one
fine_tuned_model = PeftModel.from_pretrained(peft_model, "imonghose/whisper-small-bengali-lora-final")

# Move model to GPU
fine_tuned_model = fine_tuned_model.to("cuda")

# ----------- Create Transciption from Audio using the fine-tuned whisper model --------------

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
model = fine_tuned_model
# Prepare input
inputs = processor(resampled, sampling_rate=sr, return_tensors="pt").input_features.to(device)
# Generate token ids
with torch.no_grad():
    op = model.generate(inputs, language='bengali', task='transcribe')
transcription = tokenizer.batch_decode(op, skip_special_tokens=True)[0]
print("Full Transcription:")
print(transcription)

```

## 🔭 Future Scope
- Replacing CLI pipeline with an easy-to-use public UI for fine tuning for any low resource language
- Speech diarization on top of the fine-tuned models
- Multimodal & real-time ASR extensions (streaming)
