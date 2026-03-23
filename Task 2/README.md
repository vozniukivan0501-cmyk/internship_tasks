# Internship Task 2: Animals Entity Recognition and Image Classification

## Project Overview

This repository contains a solution for Internship Task 2. It includes:

* Two models for NER and image classification
* A unified inference pipeline
* Generated data for NER training
* A synthetic text request generator
* A translator module for the Animals10 dataset
* A trained MobileNetV2-based model

---

## Model and Dataset Setup

DistilBERT-based model and Animals10 dataset are too large to upload to GitHub.

### Steps to prepare data and train the NER model:

* Download dataset: https://www.kaggle.com/datasets/alessiocorrado99/animals10
* Unpack all directories from `raw-img` into `data/animals10`
* Run translator:

  ```bash
  python src/translator.py
  ```
* Train NER model:

  ```bash
  python src/train_ner.py
  ```
* Wait until the model is saved

---

## Project Structure

```
base_dir/

├── data/
│   ├── animals10/              # dataset (user-provided)
│   └── ner_data/
│       └── ner_dataset.json   # generated NER dataset

├── models/
│   ├── ner_checkpoints/       # temporary training files
│   ├── ner_model/             # saved NER model
│   └── cv_model.keras         # trained CV model

├── notebooks/
│   ├── EDA.ipynb              # dataset analysis
│   └── DEMO.ipynb             # pipeline demo

├── src/
│   ├── cv_model_module.py
│   ├── ner_model_module.py
│   ├── train_cv.py
│   ├── train_ner.py
│   ├── inference_cv.py
│   ├── inference_ner.py
│   ├── pipeline.py
│   ├── translator.py
│   └── request_generator.py
```

---

## Installation and Setup

* Clone the repository:

  ```bash
  git clone https://github.com/vozniukivan0501-cmyk/internship_task2
  ```

* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

* Complete all steps from **Model and Dataset Setup**

* Run the pipeline:

  ```bash
  python src/pipeline.py --request "..." --image_path "..."
  ```

  
  
  
