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

DEBUG = False

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # 텍스트 라벨 패딩
        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(
                {"input_ids": label_features},
                padding=self.padding,
                return_tensors="pt",
            )

        # 패딩 과정에서 생긴 -100 처리
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        batch["labels"] = labels
        return batch

class ModelRunner:
    trained_model_path = "./results/trained_model"

    def __init__(self, use_pretrained = False, use_processor = False):
        self.use_pretrained = use_pretrained
        self.use_processor = use_processor

        self.fix_seed()
        self.load_model(use_pretrained)
        self.prepare_data()

    def fix_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
    
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU의 경우
        
        # PyTorch에서 Reproducibility 보장
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def prepare_data(self):
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
        
        self.dataset = SpeechDataset(audio_processed_dir, text_processed_dir, self.processor)
        # labels = self.processor.tokenizer(self.dataset[0]['txt_raw'], return_tensors="pt").input_ids
        # print("Tokenized labels:", labels)

        # print(self.dataset[0])
        # exit(0)
        print('Length of Dataset' , len(self.dataset))

        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])

    def load_model(self, use_pretrained: bool):
        if use_pretrained:
            model_name = self.trained_model_path
        else:
            model_name = "facebook/wav2vec2-base-960h"

        print(f"Used Model : {model_name}")

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
        predictions = self.processor.batch_decode(pred_ids)
        labels = pred.label_ids

        labels[labels == -100] = self.processor.tokenizer.pad_token_id
        references = self.processor.batch_decode(labels, group_tokens=False)

        filtered_references = []
        filtered_predictions = []

        for ref, pred_text in zip(references, predictions):
            if ref.strip() != "":
                filtered_references.append(ref)
                filtered_predictions.append(pred_text)

        if len(filtered_references) == 0:
            return {"cer": 1.0} 

        cer = jiwer.wer(filtered_references, filtered_predictions)
        return {"cer": cer}

    def train(self):
        data_collator = DataCollatorCTCWithPadding(processor=self.processor)

        training_args = TrainingArguments(
            output_dir="./results",               
            evaluation_strategy="epoch",        
            learning_rate=3e-5,                  
            per_device_train_batch_size=8,        
            per_device_eval_batch_size=8,        
            num_train_epochs=8,                
            save_steps=500,                    
            save_total_limit=10,                 
            logging_dir="./logs",               
            logging_steps=50,                  
            fp16=torch.cuda.is_available(),       
        )

        trainer = Trainer(
            model=self.model,                      
            args=training_args,                 
            train_dataset=self.train_dataset,     
            eval_dataset=self.test_dataset,          
            data_collator=data_collator,       
            tokenizer=self.processor.feature_extractor, 
            compute_metrics=self.compute_metrics,    
        )

        trainer.train()
        trainer.save_model(self.trained_model_path)
        self.processor.save_pretrained(self.trained_model_path)

        print(f"Model and processor saved to {self.trained_model_path}")
    
    def get_test_dataset_size(self):
        return len(self.test_dataset)

    def test(self, sample_index, post_processor=None):
        sample = self.test_dataset[sample_index]

        # 모델 입력 준비
        input_values = sample["input_values"].unsqueeze(0).to(self.model.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(self.model.device)
        labels = sample["labels"].unsqueeze(0)
        raw_transcription = sample['txt_raw']

        # 모델 추론
        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attention_mask).logits

        # 예측된 토큰 ID
        pred_ids = torch.argmax(logits, dim=-1)

        # 예측 문자열 디코딩
        pred_str = self.processor.batch_decode(pred_ids)[0]
        
        if post_processor is not None: pred_str = post_processor(pred_str)

        if DEBUG:
            print("예측 결과:", pred_str)
            print("실제 라벨:", raw_transcription)

        return raw_transcription.lower(), pred_str.lower()
    
    def test_bulk(self, post_processor=None):
        test_dataset_size = self.get_test_dataset_size()

        save_dir = Path('./logs/result')
        os.makedirs(save_dir, exist_ok=True)
        prediction_result_file = save_dir / f"predictions_{self.use_pretrained}_{self.use_processor}.csv"
        prediction_summary_file = save_dir /  f"prediction-summary_{self.use_pretrained}_{self.use_processor}.txt"
        columns = ['label', 'prediction', 'WER']

        wers = []

        with open(prediction_result_file, mode="w", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(columns)

            for i in tqdm(range(test_dataset_size), desc="Prediction in progress"):
                label, prediction = self.test(i, post_processor)

                wer = jiwer.wer(label.lower(), prediction.lower()) if label else (0.0 if prediction else 1.0)
                
                writer.writerow([label, prediction, wer])
                wers.append(wer)
        
        with open(prediction_summary_file, mode="w", encoding="utf-8") as file:
            file.writelines([
                "test dataset size: " + str(test_dataset_size) + '\n',
                "average WER: " + str(sum(wers) / len(wers))
            ])

import argparse

if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("--train", action="store_true", help="Include this option to train the model. Otherwise model inference will be run.")
    argumentParser.add_argument("--bulk", action="store_true", help="When this is set, test will run in bulk on every test cases.")
    argumentParser.add_argument("--use_post_processor", action="store_true", help="When this is set, test will run in bulk on every test cases.")
    argumentParser.add_argument("--use_pretrained_model", action="store_true", help="When this is set, test will run in bulk on every test cases.")
    args = argumentParser.parse_args()

    runner = ModelRunner(use_pretrained = args.use_pretrained_model, use_processor=args.use_post_processor)
    vocabularyMatcher = VocabularyMatcher('vocabularies.txt')

    if args.train:
        runner.train()
    else:
        if args.bulk: 
            post_processor = vocabularyMatcher.get_closest_words if args.use_post_processor else None
            runner.test_bulk(post_processor = post_processor)
        else: 
            runner.test(0)
