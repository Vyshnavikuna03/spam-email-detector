# Spam Email Detection

A machine learning project that classifies emails as spam or not spam.

## Tools used
Python, scikit-learn, NLTK, pandas, Streamlit

## Dataset
Kaggle "Spam Email Classification Dataset" (83,446 emails).
Download combined_data.csv from Kaggle and save it as data/spam.csv.

## Method
1. Clean the text (lowercase, remove HTML, remove stopwords, stemming)
2. Convert text to numbers using TF-IDF
3. Train 3 models: Naive Bayes, Logistic Regression, SVM
4. Test on 20% of the data that the model has not seen

## Results
| Model | Accuracy |
|---|---|
| Naive Bayes | 96.36% |
| Logistic Regression | 98.24% |
| SVM (best) | 98.64% |

## How to run
1. python -m venv venv
2. venv\Scripts\activate
3. pip install pandas scikit-learn nltk joblib streamlit
4. python train.py
5. streamlit run app.py

## Limitations
Trained on older emails (Enron and TREC), so it can miss new spam styles
and very short messages.