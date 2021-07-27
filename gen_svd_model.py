import pandas as pd

ratings_data = pd.read_csv('ratings.csv')
books_metadata = pd.read_csv('books.csv')
ratings_data.head()

from surprise import Dataset
from surprise import Reader

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_data[['user_id', 'book_id', 'rating']], reader)

from surprise import SVD
from surprise.model_selection import cross_validate

svd = SVD(verbose=True, n_epochs=10)
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

trainset = data.build_full_trainset()
svd.fit(trainset)

import pickle

with open('model.p', 'wb') as pickle_file:
    pickle.dump(svd, pickle_file)

