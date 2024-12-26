import pickle
import re
from datetime import datetime

import nltk
import pandas as pd
from nltk import word_tokenize

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download("punkt")


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.csv"""
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')

    current_year = datetime.now().year
    df = df[(df['Year-Of-Publication'].notnull()) &
            (df['Year-Of-Publication'] > 0) &
            (df['Year-Of-Publication'] <= current_year)]

    df = df.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"], errors="ignore")

    df = df.fillna({
        'Book-Author': 'Unknown Author',
        'Publisher': 'Unknown Publisher'
    })

    return df



def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.csv"""
    df = df[df['Book-Rating'] > 0]

    mean_ratings = df.groupby('ISBN')['Book-Rating'].mean().reset_index()
    mean_ratings.columns = ['ISBN', 'Mean-Rating']

    rating_counts = df.groupby('ISBN').size().reset_index(name='Rating-Count')

    df = pd.merge(mean_ratings, rating_counts, on='ISBN')

    return df


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title"""
    stop_words = set(stopwords.words('english'))
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Выполнение шагов по обучению и тестированию модели"""
    books['Book-Author'] = books['Book-Author'].astype('category').cat.codes
    books['Publisher'] = books['Publisher'].astype('category').cat.codes

    books['Year-Of-Publication'] = StandardScaler().fit_transform(
        books[['Year-Of-Publication']]
    )

    books['Processed-Title'] = books['Book-Title'].apply(title_preprocessing)

    tfidf = TfidfVectorizer(max_features=1000)
    title_vectors = tfidf.fit_transform(books['Processed-Title']).toarray()

    books = books.drop(columns=['Book-Title', 'Processed-Title'])
    final_books = pd.concat([books.reset_index(drop=True),
                              pd.DataFrame(title_vectors)], axis=1)

    data = pd.merge(ratings, final_books, on='ISBN')

    X = data.drop(columns=['Mean-Rating', 'ISBN'])
    y = data['Mean-Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    linreg = SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE модели: {mae}")

    if mae < 1.5:
        print("Модель прошла тест")
    else:
        print("Модель не прошла тест")

    with open("linreg.pkl", "wb") as file:
        pickle.dump(linreg, file)
    print("Модель сохранена в 'linreg.pkl'")


if __name__ == "__main__":
    books = pd.read_csv("Books.csv", dtype={"ISBN": "str"}, low_memory=False)
    ratings = pd.read_csv("Ratings.csv", dtype={"ISBN": "str", "User-ID": "int64"})

    books = books_preprocessing(books)
    ratings = ratings_preprocessing(ratings)

    modeling(books, ratings)