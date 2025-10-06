# Fine Tuning Whisper ASR Model on Low Resource Indian Languages (Bengali and Telugu)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![HF Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)]()

- Practical, reproducible pipeline to fine-tune OpenAI Whisper for any **low-resource Indic languages**.
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

## 🚀 Quickstart (Fine-tuning in One Command)
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

--dataset, --language, --username, --train_frac, --test_frac are fully customizable.

The fine-tuned model is saved under --output_dir and pushed to user's hugging face.

Use a GPU for training: torch.cuda.is_available() should be True.


## ▶️ Using the Fine-Tuned Model (Inference)

```bash
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

lang = "bengali"
base_id = "openai/whisper-small"
adapter_id = "imonghose/whisper-small-bengali-lora-final"  # replace with your HF repo or local path

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Processor
processor = WhisperProcessor.from_pretrained(base_id, language=lang, task="transcribe")

# 2) Base model + LoRA adapter
base = WhisperForConditionalGeneration.from_pretrained(base_id)
base = PeftModel.from_pretrained(base, adapter_id).to(device).eval()

# 3) Transcribe (single audio)
audio: float32 16kHz mono numpy array; use librosa.load(path, sr=16000)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
with torch.no_grad():
pred_ids = base.generate(inputs, max_new_tokens=128)
text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
print(text)
```

## 🔭 Future Scope
- Replacing CLI pipeline with an easy-to-use public UI for fine tuning for any low resource language
- Speech diarization on top of the fine-tuned models
- Multimodal & real-time ASR extensions (streaming)
