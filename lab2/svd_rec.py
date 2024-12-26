import pandas as pd
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.csv"""
    df = df[df['Book-Rating'] > 0]
    df = df.dropna(subset=['User-ID', 'ISBN', 'Book-Rating'])

    df['User-ID'] = df['User-ID'].astype(str)
    df['ISBN'] = df['ISBN'].astype(str)

    user_counts = df['User-ID'].value_counts()
    df = df[df['User-ID'].isin(user_counts[user_counts >= 5].index)]

    book_counts = df['ISBN'].value_counts()
    df = df[df['ISBN'].isin(book_counts[book_counts >= 5].index)]

    print(f"После предобработки осталось {df.shape[0]} записей.")
    return df


def modeling(ratings: pd.DataFrame, random_state: int = 42) -> None:
    """В этой функции выполняются шаги обучения модели SVD."""
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=random_state)
    print(f"Размер обучающего набора: {len(trainset.build_testset())}")
    print(f"Размер тестового набора: {len(testset)}")

    # Оптимизация гиперпараметров SVD
    svd = SVD(n_factors=50, random_state=random_state, lr_all=0.005, reg_all=0.2, n_epochs=30)
    svd.fit(trainset)

    predictions = svd.test(testset)
    mae = accuracy.mae(predictions)

    print(f"Текущий MAE: {mae}")

    if mae < 1.3:
        print("Модель прошла тест!")
    else:
        print("Модель не прошла тест")

    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)
    print("Модель сохранена в 'svd.pkl'")


if __name__ == "__main__":
    ratings = pd.read_csv("ratings.csv", dtype={"User-ID": str, "ISBN": str, "Book-Rating": float})

    ratings = ratings_preprocessing(ratings)

    modeling(ratings)
