import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import SGDRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from surprise import SVD, Dataset, Reader

books = pd.read_csv('Books.csv', encoding='latin-1', low_memory=False)
ratings = pd.read_csv('Ratings.csv', encoding='latin-1')


def preprocess_data(books, ratings):
    if books.empty or ratings.empty:
        raise ValueError("DataFrames 'books' or 'ratings' are empty.")

    books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
    books['Image-URL-L'] = books['Image-URL-L'].fillna('')

    books['Book-Author'] = books['Book-Author'].fillna('Unknown Author')
    books['Publisher'] = books['Publisher'].fillna('Unknown Publisher')

    imputer = SimpleImputer(strategy='median')
    numeric_cols = books.select_dtypes(include=[np.number]).columns
    books[numeric_cols] = imputer.fit_transform(books[numeric_cols])

    books = books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)

    title_mapping = dict(enumerate(books['Book-Title'].factorize()[1]))
    books['Book-Title-Code'] = books['Book-Title'].factorize()[0]

    return books, ratings, title_mapping


books, ratings, title_mapping = preprocess_data(books, ratings)


def find_user_with_most_zeros(ratings):
    zero_ratings = ratings[ratings['Book-Rating'] == 0]
    user_with_most_zeros = zero_ratings['User-ID'].value_counts().idxmax()
    return user_with_most_zeros


def generate_recommendations_svd(user_id, books, ratings):
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset = data.build_full_trainset()

    try:
        with open('svd_model.pkl', 'rb') as f:
            svd = pickle.load(f)
    except FileNotFoundError:
        svd = SVD()
        svd.fit(trainset)
        with open('svd_model.pkl', 'wb') as f:
            pickle.dump(svd, f)

    zero_ratings = ratings[ratings['Book-Rating'] == 0]
    predictions = []
    for index, row in zero_ratings.iterrows():
        prediction = svd.predict(row['User-ID'], row['ISBN'])
        predictions.append((row['ISBN'], prediction.est))

    predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)
    recommended_books_svd = [book for book, _ in predictions_sorted if _ >= 8]

    return recommended_books_svd


def generate_recommendations_linreg(user_id, books, ratings):
    user_ratings = ratings[ratings['User-ID'] == user_id]
    merged_data = user_ratings.merge(books, on='ISBN', how='inner')

    if merged_data.empty:
        raise ValueError("No data available after merging for the user.")

    X_user = merged_data.drop(['User-ID', 'Book-Rating', 'ISBN'], axis=1)
    y_user = merged_data['Book-Rating']

    X_user = X_user.select_dtypes(include=[np.number])

    if X_user.empty or y_user.empty:
        raise ValueError("No valid numeric data available for training.")

    try:
        with open('linreg.pkl', 'rb') as f:
            linreg_model = pickle.load(f)
    except FileNotFoundError:
        linreg_model = Pipeline([
            ('preprocessor', ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), X_user.columns)
                ])),
            ('regressor', SGDRegressor())
        ])
        linreg_model.fit(X_user, y_user)
        with open('linreg.pkl', 'wb') as f:
            pickle.dump(linreg_model, f)

    # Убедимся, что у нас те же колонки в X_books, что и у пользователя
    X_books = books.drop(['ISBN'], axis=1).select_dtypes(include=[np.number])

    if X_books.empty:
        raise ValueError("No valid numeric data available for predictions.")

    # Приводим к тем же колонкам
    X_books = X_books[X_user.columns]

    predicted_ratings = linreg_model.predict(X_books)

    recommended_books = books['ISBN'][predicted_ratings >= 8].tolist()
    return recommended_books, linreg_model


user_id = find_user_with_most_zeros(ratings)

recommended_books_svd = generate_recommendations_svd(user_id, books, ratings)

recommended_books_linreg, linreg_model = generate_recommendations_linreg(user_id, books, ratings)

# Сортировка рекомендованных книг по прогнозируемым рейтингам
recommended_books_sorted = sorted(recommended_books_linreg, key=lambda book: linreg_model.predict(
    books[books['ISBN'] == book].drop(['ISBN'], axis=1).select_dtypes(include=[np.number])), reverse=True)
