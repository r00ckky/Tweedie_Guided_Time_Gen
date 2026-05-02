# Tweedie-Guided Diffusion for Imbalanced Financial Time-Series

## 📌 Overview
This project explores the use of **diffusion-based generative models** to synthesize high-quality **tabular financial time-series data**, particularly under **extreme class imbalance** conditions.

We propose a hybrid architecture combining:
- Transformer-based latent representation learning  
- Noise Conditional Score Networks (NCSN)  
- Tweedie-guided conditional generation  

The goal is to generate realistic synthetic credit default data while preserving minority class characteristics.

---

##  Methodology

### 1. Representation Learning
- Transformer-based encoder processes time-series sequences  
- Initially used VQ-VAE → faced **codebook collapse**  
- Final approach: **continuous latent space (no quantization)**  

### 2. Score-Based Generation
- Used **Noise Conditional Score Network (NCSN)**  
- Backbone: **DiT-1D (Diffusion Transformer)**  
- Models latent distribution using a **variance exploding noise schedule**

### 3. Tweedie Guidance
- Applies **post-hoc class conditioning** without training a classifier  
- Uses Tweedie’s formula to estimate clean data from noisy samples  
- Adds a gradient-based correction toward target class  

### 4. Sampling
- Reverse diffusion via **Euler-Maruyama update rule**  
- Generates synthetic sequences conditioned on class labels

---

## 📊 Dataset
- **American Express Default Prediction Dataset**
- Features:
  - 190 anonymized financial variables  
  - Time-series (up to 13 steps per user)  

### Preprocessing
- Missing value imputation (median)  
- Quantile normalization (Gaussian transformation)  
- Handling extreme class imbalance

---

##  Evaluation Strategy

### Metrics
- Balanced Accuracy  
- Confusion Matrix  
- Minority Class Recall  

### Framework
- **TRTR (Train Real → Test Real)** → baseline  
- **TSTR (Train Synthetic → Test Real)** → synthetic data evaluation

---

## 🔍 Results

| Setup | Balanced Accuracy |
|------|------------------|
| TRTR | **89%** |
| TSTR | **59.6%** |

### Key Insight
- Synthetic data fails to capture minority class patterns  
- Minority class recall drops to ~28.6%  
- High false negatives dominate predictions

---

## 🧪 Ablation Study

| Model | Performance |
|------|------------|
| Conv-ResNet | Poor |
| Seq-ResNet | Moderate |
| ConvNeXt-1D | Best loss but unstable |
| **DiT-1D** | Stable and best overall |

---

##  Authors
- Nakul Gupta  
- Dev Patel  
- Chaitanya Kohli  
- Ananya Rai 
