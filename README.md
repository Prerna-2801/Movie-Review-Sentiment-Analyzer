# Movie Sentiment Analyzer

A Machine Learning project that classifies movie reviews as positive or negative using text preprocessing, TF-IDF vectorization, and Logistic Regression.

## Dataset

The project uses `IMDB Dataset.csv`, which contains two columns:

- `review`: Movie review text.
- `sentiment`: Sentiment label (`positive` or `negative`).

The original dataset contains 50,000 reviews. After removing 418 duplicate rows, 49,582 reviews remain:

- Positive: 24,884
- Negative: 24,698

## Project Workflow

1. **Exploratory Data Analysis (EDA):** Checked missing values, identified and removed duplicate rows, and examined sentiment distribution.
2. **Data Splitting:** Split the data into training and testing sets using an 80/20 stratified split.
3. **Text Preprocessing:** Converted text to lowercase, removed HTML break tags and non-alphabetic characters, tokenized text, removed English stop words, and applied WordNet lemmatization using NLTK.
4. **TF-IDF Vectorization:** Converted cleaned reviews into numerical features using `TfidfVectorizer`, fitting it on the training data.
5. **Model Training:** Trained a Logistic Regression classifier with `max_iter=1000`.
6. **Model Evaluation:** Evaluated the model using accuracy and a classification report.
7. **Custom Review Prediction:** Implemented `predict_my_review()` to predict sentiment and display confidence for a custom review.

## Model Performance

The model was evaluated on 9,917 test reviews.

| Metric | Negative | Positive |
|---|---:|---:|
| Precision | 0.90 | 0.88 |
| Recall | 0.87 | 0.90 |
| F1-score | 0.88 | 0.89 |

**Overall Accuracy:** 88.61%

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- Jupyter Notebook

## Installation and Usage

### 1. Clone or download the project
Download the project files and place them in the same working directory.

### 2. Install dependencies
Make sure `requirements.txt` is present in the project directory, then run:

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset
Place `IMDB Dataset.csv` in the directory used by the notebook.

### 4. Open the Jupyter Notebook
Run:

```bash
jupyter notebook movie_sentiment_analyzer.ipynb
```

Run the notebook cells in order to preprocess the reviews, train the model, and evaluate its performance.

