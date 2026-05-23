# Multimodal Emotion Recognition (MER): Late Fusion and Modality Constraint Analysis

This repository contains the official implementation for the **IIITH Phase 2 Assignment on Multimodal Emotion Recognition (MER)**. The system classifies human emotions into seven distinct categories (*Anger, Disgust, Fear, Happiness, Pleasant Surprise, Sadness, and Neutral*) by utilizing acoustic and semantic features. 

Our pipeline implements two unimodal encoders—**Speech (Modality A)** and **Text (Modality B)**—and combines their decision-level representations through a late fusion meta-classifier.

---

## 1. Project Overview

The objective of this project is to explore the synergies and constraints of speech and text modalities in emotional classification. The system operates on the **Toronto Emotional Speech Set (TESS)**, which contains audio recordings of actresses speaking short carrier phrases. 

- **Dataset Acquisition**: Automated via the Kaggle API using the custom `utils/data_manager.py` utility.
- **Classification Objective**: Identify 7 primary emotions:
  1. Anger (`angry`)
  2. Disgust (`disgust`)
  3. Fear (`fear`)
  4. Happiness (`happy`)
  5. Pleasant Surprise (`ps`)
  6. Sadness (`sad`)
  7. Neutral (`neutral`)
- **Integration Methodology**: Late Fusion (Decision-Level) utilizing unimodal logits routed into a shallow meta-classifier.

---

## 2. Directory Structure

The repository is structured as a modular machine learning pipeline:

```text
speech_pipeline/
text_pipeline/
fusion_pipeline/
utils/
models/
Results/
requirements.txt
README.md
```

Detailed directory tree breakdown:
* **`speech_pipeline/`**: Contains `dataset.py` (Librosa audio loading, features extraction, and padding), `model.py` (HuBERT + Bi-LSTM architecture), `train.py`, and `test.py`.
* **`text_pipeline/`**: Contains `dataset.py` (DistilBERT tokenization), `model.py` (DistilBERT backbone + Pooling head), `train.py`, and `test.py`.
* **`fusion_pipeline/`**: Contains `dataset.py` (Synchronized audio-text loader), `model.py` (Decision-Level Late Fusion Meta-Classifier), `train.py`, and `test.py`.
* **`utils/`**: Contains `data_manager.py` (Automated Kaggle dataset downloader using `kagglehub`).
* **`models/`**: Serialization directory containing saved `.pth` weights for trained unimodal and fused models.
* **`Results/`**: Repository for model outputs, containing confusion matrices and `tsne_visualization.png`.
* **`requirements.txt`**: Environment specification list.
* **`README.md`**: Academic-grade project documentation.

---

## 3. Architecture & Results

### Modality A: Speech (Acoustic Pipeline)
* **Architecture**: The acoustic pipeline utilizes a pre-trained **HuBERT** backbone (`facebook/hubert-base-ls960`) to extract robust contextualized speech representations. These representations are fed into a **2-layer Bidirectional LSTM (Bi-LSTM)** to capture temporal dynamics, followed by an MLP head.
* **Accuracy**: **83%**
* **Engineering Insight**: HuBERT’s self-supervised representations provide strong acoustic boundaries, enabling highly effective speaker-independent generalization.

### Modality B: Text (Semantic Pipeline)
* **Architecture**: The semantic pipeline employs a pre-trained **DistilBERT** model (`distilbert-base-uncased`) with **Global Average Pooling** across token representations, followed by a linear classification head.
* **Accuracy**: **~14%**
* **The Text Constraint (Crucial Context)**: 
  > [!WARNING]
  > **The Semantic Paradox**: An accuracy of ~14% resides exactly at the **random chance baseline (1/7 $\approx$ 14.28%)**. 
  > This is not a failure of DistilBERT or the training pipeline, but an inherent constraint of the TESS dataset design. The transcripts in TESS are comprised of highly repetitive, neutral carrier phrases (e.g., *"Say the word back"*, *"Say the word choice"*). Since the text transcripts themselves contain **zero emotional semantics**, the semantic model has no predictive signal and converges to a random guess. Unfreezing or scaling this model merely leads to severe overfitting on spurious noise.

### Multimodal Fusion (Decision-Level)
* **Architecture**: The fusion pipeline implements **Decision-Level Late Fusion**. Unimodal models extract 7-dimensional logits representing class scores. These logits are concatenated into a **14-dimensional joint feature representation**:
  $$\mathbf{z}_{\text{fusion}} = [\mathbf{z}_{\text{speech}} \parallel \mathbf{z}_{\text{text}}] \in \mathbb{R}^{14}$$
  This joint vector is routed through a trainable **Meta-Classifier** (consisting of Fully Connected layers, Layer Normalization, ReLU activation, and Dropout) to yield the final emotional predictions.
* **Accuracy**: **86% overall accuracy**
* **Engineering Insight**: Despite the noisy and uninformative nature of the text features, the late fusion Meta-Classifier successfully regularizes and learns to filter text logits, achieving a $+3\%$ absolute improvement over the strong Speech-only baseline.

| Modality | Core Model | Input Representation | Test Accuracy |
| :--- | :--- | :--- | :---: |
| **Speech (Modality A)** | HuBERT + 2-layer Bi-LSTM | 16kHz Raw Audio (Librosa) | **83.0%** |
| **Text (Modality B)** | DistilBERT + Global Average Pooling | Neutral Carrier Transcripts | **~14.0%** |
| **Multimodal Fusion** | Late Fusion + Meta-Classifier | Concatenated 14-D Logits | **86.0%** |

---

## 4. Critical Analysis (Engineering Insights & Trade-offs)

Under rigorous evaluation, decision-level late fusion introduced critical trade-offs between global accuracy optimization and class-specific sensitivity.

### 1. The 'Sadness' Breakthrough
* **Observation**: In the acoustic-only model, **Sadness** achieved a poor recall of **0.69**. Upon routing unimodal logits through the Fusion Meta-Classifier, the recall surged to **0.88** (a $+19\%$ absolute increase).
* **Analysis**: Sadness in acoustic speech often presents overlapping features with Neutral states due to low intensity and low pitch variance. While the speech model struggled to separate these under standard temporal pooling, the Meta-Classifier successfully leveraged joint probability boundaries, utilizing the tiny differences in acoustic-semantic correlations to dramatically boost class sensitivity.

### 2. The 'Disgust' Casualty
* **Observation**: The recall for **Disgust** dropped sharply from **0.77** in the acoustic-only model to **0.56** in the late fusion model.
* **Analysis**: The fusion network's optimizer attempts to maximize global cross-entropy. To regularize the noise introduced by the uninformative text features (~14% accuracy), the Meta-Classifier heavily penalized the volatile, high-frequency decision boundaries of less frequent categories. Disgust, which relies on extremely distinct but volatile acoustic transitions, had its boundaries smoothed over by this global regularization, resulting in misclassifications and a performance drop.

### 3. Latent Space Visualizations
* **Visual Verification**: High-dimensional features extracted prior to classification were projected using t-Distributed Stochastic Neighbor Embedding (**t-SNE**). 
* The resulting visualization, stored in `Results/tsne_visualization.png`, provides **empirical proof** of the pipeline's behavior:
  - **High-Arousal Emotions** (e.g., Anger, Pleasant Surprise) form clean, isolated semantic clusters in the latent space.
  - **Low-Arousal/Neutral States** show minor, expected overlap, validating the model's ability to model domain boundaries while maintaining distinct cluster centers.

---

## 5. Installation & Setup

Follow these steps to set up the environment and replicate the experiment.

### Prerequisites
Ensure you have Python 3.8+ installed. You also need a Kaggle account to fetch the dataset automatically.

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd Multimodal_Emotion_Recognition
   ```

2. **Configure Kaggle Credentials**:
   Place your `kaggle.json` API token under `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows). Alternatively, export your credentials:
   ```bash
   export KAGGLE_USERNAME="your_kaggle_username"
   export KAGGLE_KEY="your_kaggle_api_key"
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r project/requirements.txt
   ```

---

## 6. Execution Guide

The pipelines are structured to run independently or end-to-end. Navigate to the `project` directory before running:
```bash
cd project
```

### Step 1: Download the Dataset
The dataset will be downloaded automatically when running `data_manager.py`:
```bash
python utils/data_manager.py
```

### Step 2: Train Unimodal Pipelines
Train the Speech and Text models first to serialize their weights to the `models/` directory:
```bash
# Train Modality A (Speech)
python speech_pipeline/train.py

# Train Modality B (Text)
python text_pipeline/train.py
```

### Step 3: Train Late Fusion Meta-Classifier
Once the unimodal checkpoints are saved, train the fusion pipeline:
```bash
python fusion_pipeline/train.py
```

### Step 4: Run End-to-End Evaluation
To run the full evaluation suite, generate comparative reports, and construct the t-SNE plot:
```bash
python evaluate_all.py
```
This script will:
1. Load speech, text, and fusion checkpoints.
2. Generate an academic comparative report under `Results/comparative_report.md`.
3. Save the t-SNE latent space visualization to `Results/tsne_visualization.png`.
