from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from src.utils import parse_xml_to_segments, clean_text
from src.load_dataset import download_file, extract_tgz
import torch, os
from pydub import AudioSegment
from pathlib import Path
import librosa
from pathlib import Path

from src.SpeechDataset import SpeechDataset

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
import torch

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 오디오 부분 추출
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        # 라벨 부분 추출
        label_features = [feature["labels"] for feature in features]

        # 오디오 입력 패딩
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

from jiwer import wer

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
    predictions = processor.batch_decode(pred_ids)
    labels = pred.label_ids

    labels[labels == -100] = processor.tokenizer.pad_token_id
    references = processor.batch_decode(labels, group_tokens=False)

    # 빈 문자열 참조 제거
    filtered_references = []
    filtered_predictions = []

    for ref, pred_text in zip(references, predictions):
        if ref.strip() != "":
            filtered_references.append(ref)
            filtered_predictions.append(pred_text)
        # ref가 빈 문자열이면 해당 pair는 무시

    if len(filtered_references) == 0:
        # 모든 참조가 빈 문자열일 경우 WER 계산 불가
        # 적절히 처리하거나 디폴트값 반환
        return {"cer": 1.0}  # 혹은 {"cer": float('nan')}

    cer = wer(filtered_references, filtered_predictions)
    return {"cer": cer}

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


# Pretrained 모델과 토크나이저 불러오기
model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# 데이터셋 생성
dataset = SpeechDataset(audio_processed_dir, text_processed_dir, processor)

print('Length of Dataset' , len(dataset))

# 데이터셋 나누기 (80% Train, 20% Test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 2. DataCollator 정의
# DataCollator는 배치 데이터를 패딩하여 모델 입력에 맞게 만듭니다.
data_collator = DataCollatorCTCWithPadding(processor=processor)

# 3. CER 계산 함수 정의

# 4. TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",                # 결과 저장 디렉토리
    evaluation_strategy="epoch",          # 매 에포크마다 평가
    learning_rate=3e-5,                   # 학습률
    per_device_train_batch_size=16,        # 학습 배치 크기
    per_device_eval_batch_size=16,         # 평가 배치 크기
    num_train_epochs=10,                   # 학습 에포크 수
    save_steps=500,                       # 모델 저장 빈도
    save_total_limit=2,                   # 저장할 체크포인트 최대 개수
    logging_dir="./logs",                 # 로그 디렉토리
    logging_steps=50,                     # 로그 출력 간격
    fp16=torch.cuda.is_available(),       # FP16 사용 (GPU 환경에서)
)

# 5. Trainer 생성
trainer = Trainer(
    model=model,                         # 학습할 모델
    args=training_args,                  # 학습 설정
    train_dataset=train_dataset,         # 학습 데이터셋
    eval_dataset=test_dataset,           # 평가 데이터셋
    data_collator=data_collator,         # 데이터 전처리(collation)
    tokenizer=processor.feature_extractor,  # 토크나이저
    compute_metrics=compute_metrics,     # 평가 지표
)

bef_results = trainer.evaluate()


trainer.train()
aft_results = trainer.evaluate()

print(f"Evaluation results before train: {bef_results}")
print(f"Evaluation results after train: {aft_results}")

# sample = test_dataset[0]

# # 모델 입력 준비
# input_values = sample["input_values"].unsqueeze(0).to(model.device)
# attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)
# labels = sample["labels"].unsqueeze(0)
# raw_transcription = sample['txt_raw']

# # 모델 추론
# with torch.no_grad():
#     logits = model(input_values, attention_mask=attention_mask).logits

# # 예측된 토큰 ID
# pred_ids = torch.argmax(logits, dim=-1)

# # 예측 문자열 디코딩
# pred_str = processor.batch_decode(pred_ids)[0]

# # 실제 라벨 디코딩
# # 라벨에서 -100 부분을 pad 토큰 id로 대체
# labels[labels == -100] = processor.tokenizer.pad_token_id
# ref_str = processor.batch_decode(labels, group_tokens=False)[0]

# print("예측 결과:", pred_str)
# print("실제 라벨:", raw_transcription)