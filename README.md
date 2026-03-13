# News Article Classification using Machine Learning

##  Project Overview
This project implements a **machine learning pipeline for classifying news articles into different categories** such as Business, Sports, Technology, Entertainment, and Politics.  

The pipeline performs **text preprocessing, feature extraction using TF-IDF, model training, and evaluation** using Python scripts only. The entire workflow can be executed from the terminal using a single entry script.

---

##  Objective
The objective of this project is to build a **News Article Classification system** that automatically predicts the category of a news article based on its textual content.

The project demonstrates the **complete machine learning workflow**:
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation

---

## Dataset

**Dataset Source:**  
BBC News Dataset
https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category

The dataset contains news articles categorized into different topics such as:

- Business
- Sport
- Tech
- Entertainment
- Politics

Each record in the dataset contains:

| Column | Description |
|--------|-------------|
| category | Category of the news article |
| text | News article content |

Example:
```
category,text
tech,TV future in the hands of viewers...
sport,Tigers wary of Farrell gamble...
business,Worldcom boss left books alone...
```

---

## Project Folder Structure

```
news_classification_project/
│
├── data/
│   ├── raw/
│   │   └── news_dataset.csv
│   └── processed/
│       └── cleaned_news.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── config.py
│
├── models/
│   └── news_classifier.pkl
│
├── results/
│   └── metrics.txt
│
├── requirements.txt
├── README.md
└── main.py
```

---

## Machine Learning Pipeline

###  Data Preprocessing
**Script:** `data_preprocessing.py`

Steps performed:
- Load dataset
- Remove missing values
- Convert text to lowercase
- Remove special characters
- Remove stopwords
- Save cleaned dataset

###  Feature Engineering
**Script:** `feature_engineering.py`

**Technique used:** **TF-IDF Vectorization**

**Purpose:** Convert text data into numerical features that can be used by machine learning models.

###  Model Training
**Script:** `train.py`

**Model used:** **Multinomial Naive Bayes**

Steps:
- Split dataset into **training and testing sets**
- Train the model on training data
- Save trained model in `models/`

###  Model Evaluation
**Script:** `evaluate.py`

**Evaluation metrics used:**
- **Accuracy Score**
- **Confusion Matrix**

Results are saved in: `results/metrics.txt`

###  Entry Point
**Script:** `main.py`

Running this script executes the entire pipeline:
1. Data preprocessing  
2. Feature engineering  
3. Model training  
4. Model evaluation  

---

## Steps to Run the Project

###  Clone Repository
```bash
git clone <repository_link>
cd news_classification_project
```

###  Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Pipeline
```bash
python main.py
```

---

##  Output

After execution, the following files are generated:

```
data/processed/cleaned_news.csv
models/news_classifier.pkl
results/metrics.txt
```

**Example output:**
```
Final Accuracy: 0.95
```

---

## Model Used

**Multinomial Naive Bayes**

**Reasons for choosing this model:**
- Works well for text classification
- Fast training
- Effective with TF-IDF features

---

## Final Results

**Example evaluation result:**

```
Accuracy: 0.95
Confusion Matrix:
[[40 1 0 0 0]
 [0 38 1 0 0]
 [0 0 41 0 1]
 [0 0 1 39 0]
 [0 0 0 0 40]]
```

---

##  Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Joblib

---

##  Requirements

```bash
pandas
numpy
scikit-learn
nltk
joblib
```

---

##  Explanation Video

A 2–3 minute explanation video describing:
- Dataset
- Project architecture
- Machine learning model
- Results
- Key learnings

**Video Link:**  
`<Add video link here>`

---

## 🔗 GitHub Repository

**Repository Link:**  
`<Add your GitHub repository link here>`

---

## Key Learnings

- Text preprocessing techniques in NLP
- Feature extraction using TF-IDF
- Building an end-to-end machine learning pipeline
- Training and evaluating classification models
- Structuring ML projects for production environments

---

##  Author

**Nandini Giri**  
AIML Engineering Student
```
