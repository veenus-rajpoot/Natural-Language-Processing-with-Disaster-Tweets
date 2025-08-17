# 🚨 Disaster Tweets Classification (JAX/Flax + BERT)

This project classifies tweets into **disaster-related (1)** or **not disaster-related (0)** using **BERT embeddings** and the **JAX/Flax deep learning framework**.  
It is based on the [Kaggle NLP Disaster Tweets dataset](https://www.kaggle.com/c/nlp-getting-started).

---

## 📌 Project Workflow

### 1. Setup
- Install dependencies: `jax`, `flax`, `optax`, `torch`, `transformers`, `pandas`, `numpy`
- Configure JAX backend (GPU/TPU if available)

### 2. Dataset
- Dataset files (`train.csv`, `test.csv`, `sample_submission.csv`) are loaded from Kaggle input.
- `train.csv` → contains tweet text + labels (0 or 1)  
- `test.csv` → contains tweet text (without labels)

### 3. Preprocessing
- HuggingFace `AutoTokenizer` with **BERT (bert-base-uncased or bert-large-uncased)**
- Sentences tokenized into:
  - `input_ids`
  - `attention_mask`
  - `token_type_ids`
- All sequences padded to the **maximum sentence length** in dataset.

### 4. DataLoader
- Convert processed data into tensors (`torch.utils.data`)
- Create training & validation loaders

### 5. Model
- Load pretrained BERT with classification head:
  ```python
  FlaxAutoModelForSequenceClassification.from_pretrained(
      model_name, num_labels=2
  )
