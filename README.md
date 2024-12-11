
# AE556_project

## Overview
This project focuses on **training** and **evaluating** a speech recognition model using Facebook's Wav2Vec2 model for CTC (Connectionist Temporal Classification). It includes tools for data preparation, model training, evaluation, and integration with external outputs like Whisper.

---

## Directory Structure
```
project/
│-- main.py                   # Main script to run the project
│-- requirements.txt          # Python dependencies
│-- extract_vocabularies.py   # Script to extract vocabularies
│-- get_whisper_api_results.py# Script to fetch and process API results
│-- vocabularies.txt          # Extracted vocabularies (output)
│-- secure.txt                # Secure configuration or keys
│-- src/
│   │-- load_dataset.py       # Module to load datasets
│   │-- process_label.py      # Module to process labels
│-- .gitignore                # Git ignore file
```


### Description
The `main.py` file is the main script that:

1. Downloads and preprocesses a speech dataset.  
2. Trains and evaluates the Wav2Vec2 model for Automatic Speech Recognition (ASR).  
3. Supports optional post-processing to clean transcriptions using a vocabulary matcher.  
4. Includes integration with Whisper outputs for comparison.  

---

### Features
- **Data Preparation**:
   - Downloads ATCO2-ASR dataset.
   - Splits audio and transcription into segments.
   - Prepares training and test datasets.

- **Model Training**:
   - Supports training Facebook's Wav2Vec2 models (`base` or `large`).
   - Implements proper data collators for padding and batching.

- **Testing and Evaluation**:
   - Supports individual and bulk inference on test datasets.
   - Calculates Word Error Rate (WER) for accuracy assessment.
   - Summarizes test results into CSV and TXT files.

- **Post-processing**:
   - Cleans and aligns model outputs using a predefined vocabulary.

- **Integration with Whisper**:
   - Loads pre-generated Whisper outputs for comparison.

---

### Requirements
1. **Python Packages**:
   - `torch`
   - `transformers`
   - `pydub`
   - `jiwer`
   - `numpy`
   - `tqdm`
2. **External Dependencies**:
   - ATCO2-ASR dataset.
3. **Hardware**:
   - GPU support is recommended.

---

### Installation
It is highly recommended to run this on a GPU rather than a CPU for better performance. This setup is designed for Linux servers. While it has not been fully tested, it may not work properly on macOS.


For the newer nvidia driver version
```bash
git clone https://github.com/dolgae03/ae556_project.git
cd ae556_project

conda create -n ae556_project python=3.9.13
conda activate ae556_project
pip install -r requirements.txt
```

For the older nvidia driver version(like 470.256.02, CUDA Version: 11.4)
```bash
git clone https://github.com/dolgae03/ae556_project.git
cd ae556_project

conda create -n ae556_project python=3.9.13
conda activate ae556_project

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers pydub librosa Levenshtein jiwer transformers[torch] tqdm
```
---

### Usage

#### Training the Model
To train the large model:
```bash
python main.py --train --use_large
```

#### Testing the Model
**Bulk Testing**:
```bash
python main.py --bulk --use_post_processor
```

**Testing with Whisper**:
```bash
python main.py --bulk --whisper --use_post_processor
```

---

### Key Functions
- **`ModelRunner` Class**:
   - Handles model loading, dataset preparation, training, and testing.

- **`DataCollatorCTCWithPadding`**:
   - Pads input and target sequences for Wav2Vec2 training.

- **`clean_text()`**:
   - Cleans text transcriptions for training and testing.

- **`compute_metrics()`**:
   - Computes WER for predictions.

---

### Outputs
- **Model Files**: Saved to `./results/trained_model/`.
- **Logs**: WER summaries and prediction outputs are stored in `./logs/result/`.

---

### Notes
- Adjust training parameters like `learning_rate`, `num_train_epochs`, and batch sizes within the `TrainingArguments` class.
- Add a file `vocabularies.txt` for vocabulary-based post-processing.

---

### Example Workflow
1. Train the model:
   ```bash
   python main.py --train
   ```
2. Test the trained model:
   ```bash
   python main.py --bulk
   ```
3. Evaluate against Whisper outputs with post-processor:
   ```bash
   python main.py --whisper --use_post_processor
   ```
---
