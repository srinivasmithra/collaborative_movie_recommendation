import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# mean ratingby movie id
mean_rating = ratings.groupby('movieId')[['rating']].mean()

movie_stats = ratings.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()

# Normal mean calculation doesnot gives appropraite results ratings, so that we are taking bayesian average into consideration.
C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()

def bayesian_avg(ratings):
    bayesian_avg = (C*m+ratings.sum())/(C+ratings.count())
    return bayesian_avg

bayesian_avg_ratings = ratings.groupby('movieId')['rating'].agg(bayesian_avg).reset_index()
bayesian_avg_ratings.columns = ['movieId', 'bayesian_avg']
movie_stats = movie_stats.merge(bayesian_avg_ratings, on='movieId')

movie_stats = movie_stats.merge(movies[['movieId', 'title']])
movie_stats.sort_values(by ='bayesian_avg', ascending=False).head()


def create_X(df):
    N = df['userId'].nunique()
    M = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper



def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []

    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids




if __name__ == '__main__':
    movie_titles = dict(zip(movies['movieId'], movies['title']))
    movie_id = 3
    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)
    X.todense()
    sparsity = X.count_nonzero() / (X.shape[0] * X.shape[1])
    # print(f"Matrix sparsity: {round(sparsity * 100, 2)}%")

    similar_ids = find_similar_movies(movie_id, X, k=10)
    movie_title = movie_titles[movie_id]
    def similar_movies():
        print(f"Because you watched {movie_title}")
        for i in similar_ids:
            print(movie_titles[i])

    similar_movies()


