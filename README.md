# Fine-Tuning Whisper ASR Model for Low-Resource Languages

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![HF Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)]()

This repository presents a compact and reproducible framework for adapting OpenAI Whisper to low-resource automatic speech recognition (ASR) settings, with a focus on two major Indian Languages, namely Bengali and Telugu. The project studies parameter-efficient fine-tuning (PEFT) strategies on the Whisper Small model and provides training, evaluation, visualization, and inference artifacts in a structure intended for straightforward reuse.

The accompanying study evaluates four adaptation settings: LoRA, LoRA with SpecAugment, BitFit, and Adapter Layers. Across the reported experiments, LoRA-based adaptation delivered the most reliable balance of word error rate (WER) reduction, training stability, and computational efficiency.

## Research Paper
**Title:** BREAKING LANGUAGE BARRIERS: FINE-TUNING WHISPER FOR BENGALI AND TELUGU AUTOMATIC SPEECH RECOGNITION  
**Date:** April 2025  
**Authors:** Imon Kalyan Ghosh, Ishmita Basu, Bathula Veera Raghavulu  
**Paper:** [Research_Paper.pdf](paper/Research_Paper.pdf)  
**Presentation:** [Paper_Presentation.pptx](paper/Paper_Presentation.pptx)

## Project Scope
- Fine-tuning Whisper Small for Bengali and Telugu ASR in low-resource conditions.
- Comparative evaluation of PEFT methods using WER, runtime efficiency, and training dynamics.
- Modular training pipeline for adapting Whisper to a custom dataset via a simple CLI interface.
- Supplementary notebooks for visualization, qualitative inspection, and downstream speech diarization exploration.

## Main Findings
- Fine-tuning substantially improves performance over zero-shot Whisper on both target languages.
- LoRA provides the strongest overall trade-off between accuracy, stability, and resource efficiency.
- LoRA with SpecAugment yields marginal gains in some settings, particularly for Bengali, but introduces additional evaluation overhead.
- BitFit is computationally lightweight but exhibits limited adaptation capacity.
- Adapter Layers show less stable optimization behavior and weaker WER gains in this study.

The paper further reports that the LoRA configuration achieved competitive adaptation while updating fewer than 3% of model parameters, making it particularly suitable for constrained training environments.

## Fine-Tuning Workflow
<img width="865" height="497" alt="Whisper fine-tuning workflow" src="https://github.com/user-attachments/assets/ae0efb67-4eb6-48e8-97ee-a10902b93936" />

## Results Snapshot
LoRA-based methods consistently produced the strongest empirical results across the Bengali and Telugu experiments conducted in the paper.

<img width="1367" height="318" alt="Comparative results across PEFT methods" src="https://github.com/user-attachments/assets/04548136-6b82-4142-b201-7e913ccd7c68" />

## Quickstart
Install the core dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -U pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft evaluate jiwer soundfile librosa tensorboard
```

Run the CLI pipeline from the project root:

```bash
python lora_finetuning_pipeline.py \
  --dataset imonghose/bengali-asr-data \
  --language bengali \
  --username imonghose \
  --model_size small \
  --output_dir ./model-tensors/finetuned_whisper_bengali \
  --logging_dir ./tensorboard/lora-tensorboard \
  --train_frac 0.25 \
  --test_frac 0.75
```

Key arguments such as `--dataset`, `--language`, `--username`, `--output_dir`, `--train_frac`, and `--test_frac` are fully configurable. The resulting adapter is saved locally under `--output_dir` and can also be pushed to the Hugging Face Hub.

## Inference Example
```python
AUDIO_FILE = "audio_samples/bengali-convo-2.wav"

waveform, sr = torchaudio.load(AUDIO_FILE)
waveform = waveform[0].numpy()
resampled = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
sr = 16000

model_name_or_path = "openai/whisper-small"
language = "bengali"
task = "transcribe"

tokenizer = WhisperTokenizer.from_pretrained(
    model_name_or_path,
    language=language,
    task=task,
)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["k_proj", "v_proj", "q_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
)

base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
peft_model = get_peft_model(base_model, config)
fine_tuned_model = PeftModel.from_pretrained(
    peft_model,
    "imonghose/whisper-small-bengali-lora-final",
).to("cuda")

processor = WhisperProcessor.from_pretrained(
    model_name_or_path,
    language=language,
    task=task,
)

inputs = processor(
    resampled,
    sampling_rate=sr,
    return_tensors="pt",
).input_features.to("cuda")

with torch.no_grad():
    generated = fine_tuned_model.generate(inputs, language="bengali", task="transcribe")

transcription = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(transcription)
```

## Future Directions
- Integrating speaker diarization with fine-tuned Whisper for multi-speaker transcription.
- Building a modular user-facing interface for dataset upload, training, and inference.
- Extending the pipeline toward dialect adaptation, larger datasets, and real-time or multimodal ASR workflows.
