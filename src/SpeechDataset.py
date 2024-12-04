import os
import random
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import wer
from pathlib import Path

class SpeechDataset(Dataset):
    def __init__(self, audio_dir: Path, text_dir: Path, processor):
        self.audio_files = sorted(audio_dir.glob("*.wav"))  # 오디오 파일 필터링 (*.wav)
        self.text_files = sorted(text_dir.glob("*.txt"))    # 텍스트 파일 필터링 (*.txt)
        self.processor = processor

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # 오디오 로드
        audio_path = self.audio_files[idx]
        speech, sr = librosa.load(audio_path, sr=16000)

        # 텍스트 로드
        text_path = self.text_files[idx]
        with text_path.open("r", encoding="utf-8") as f:
            transcript = f.read().strip()

        # 전처리
        inputs = self.processor(
            speech,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )

        return {
            "input_values": inputs.input_values.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": self.processor.tokenizer(transcript, return_tensors="pt").input_ids.squeeze(0),
            "txt_raw": transcript
        }