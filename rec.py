import numpy as np
import pandas as pd


data = pd.io.parsers.read_csv('ratings.csv',
    names=['book_id', 'user_id', 'rating'],
    engine='python', delimiter=',')
book_data = pd.io.parsers.read_csv('books.csv',
    names=['id','book_id','best_book_id','work_id','books_count','isbn','isbn13','authors','original_publication_year','original_title','title','language_code','average_rating','ratings_count','work_ratings_count','work_text_reviews_count','ratings_1','ratings_2','ratings_3','ratings_4','ratings_5','image_url','small_image_url'],
    engine='python', delimiter=',')


#Creating the rating matrix (rows as books, columns as users)
ratings_mat = np.ndarray(
    shape=(np.max(data.book_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.book_id.values-1, data.user_id.values-1] = data.rating.values

#Normalizing the matrix(subtract mean off)
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

#Computing the Singular Value Decomposition (SVD)
A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)

#Function to calculate the cosine similarity (sorting by most similar and returning the top N)
def top_cosine_similarity(data, book_id, top_n=10):
    index = book_id - 1 # book id starts from 1 in the dataset
    book_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(book_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Function to print top N similar book
def print_similar_book(book_data, book_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    book_data[book_data.book_id == book_id].title.values[0]))
    for id in top_indexes + 1:
        # print(id)
        print(book_data[book_data.id == id].title.values[0])

#k-principal components to represent books, book_id to find recommendations, top_n print n results
k = 100
book_id = 2 # (getting an id from dataset)
top_n = 10
print("========")
print(V)
print("===========")
sliced = V.T[:, :k] # representative data
# indexes = top_cosine_similarity(sliced, book_id, top_n)
#
# #Printing the top N similar book
# print_similar_book(book_data, book_id, indexes)