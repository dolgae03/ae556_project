from dataclasses import dataclass
from typing import Any, Dict, List, Union

import os, csv
from pathlib import Path

from tqdm import tqdm

import jiwer, random
from pydub import AudioSegment
import numpy as np

import torch
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)

from src.SpeechDataset import SpeechDataset
from src.load_dataset import download_file, extract_tgz
from src.utils import parse_xml_to_segments, clean_text, VocabularyMatcher
from openai import OpenAI
import time
import pickle

with open('./secure.txt', 'r') as f:
    api_key = f.read()


def prepare_data():
    url = "https://www.replaywell.com/atco2/download/ATCO2-ASRdataset-v1_beta.tgz"
    save_dir = Path('./data') 
    target_dir = save_dir / 'dataset_raw.tgz'
    unzip_dir = save_dir / 'dataset_unzip'
    audio_processed_dir = save_dir / 'dataset_processed/audio'
    text_processed_dir = save_dir / 'dataset_processed/text'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(unzip_dir, exist_ok=True)
    os.makedirs(audio_processed_dir, exist_ok=True)
    os.makedirs(text_processed_dir, exist_ok=True)

    if not os.path.exists(target_dir):
        download_file(url, target_dir)
    extract_tgz(target_dir, unzip_dir)

    base_dir = unzip_dir / 'ATCO2-ASRdataset-v1_beta/DATA'

    data_list = os.listdir(base_dir)
    data_list_without_ext = set([os.path.splitext(file_name)[0] for file_name in data_list])
    data_list_without_ext = list(data_list_without_ext)

    data_list_without_ext.sort()

    target_path = Path('./logs/')
    target_data = {}

    for idx, each_file_name in enumerate(data_list_without_ext, 0):
        text_file = base_dir / f'{each_file_name}.xml'
        audio_file = base_dir / f'{each_file_name}.wav'

        with open(text_file) as f:
            trasnscription = f.read()

        audio = AudioSegment.from_wav(audio_file)
        segments = parse_xml_to_segments(trasnscription)
        
        for segment_idx, segment in enumerate(segments):
            start, end = segment['start'], segment['end']

            segment_text = clean_text(segment['text'])
            segment_audio = audio[start * 1000:end * 1000]

            segment_audio_file = audio_processed_dir / f"data_{idx}_{segment_idx}.wav"
            segment_audio.export(segment_audio_file, format="wav")
        
            segment_transcription_file = text_processed_dir / f"data_{idx}_{segment_idx}.txt"
            with open(segment_transcription_file, 'w') as f:
                f.write(segment_text)


            segment_audio_fd = open(segment_audio_file, 'rb')
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=segment_audio_fd, 
                response_format="text",
                language = 'en'
            )

            print(transcription)

            target_data[f'data_{idx}_{segment_idx}'] = [segment_text, transcription]
            time.sleep(0.25)

        with open(target_path / 'result/whisper.pkl', 'wb') as f:
            pickle.dump(target_data, f)

client = OpenAI(api_key = api_key)
prepare_data()

with open('./logs/result/whisper.pkl','rb') as f:
    print(pickle.load(f))