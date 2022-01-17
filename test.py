"""
# Movie Recommender System

A recommender system is a system that attempts to predict the rating or preference a user would give to a certain item.
In this case, it is intended to create a movie recommender system.

A popular algorithm for these kind of systems is the Singular Value Decomposition (SVD), that has been utilized to achieve better results, as it will be demonstrated ahead.
"""

"""
## Dataset

The MovieLens Dataset is most often used for the purpose of recommender systems, which aim to predict user movie ratings based on other users’ ratings.

The dataset used was extracted from MovieLens and contains 100836 ratings and 3683 tag applications across 9742 movies. This data was created by 610 users between March 29, 1996 and September 24, 2018.

The data are contained in the files:
- Movies.csv: movieId, title, genres.
- Ratings.csv: userId, movieId, rating, timestamp.
- Tags.csv: userId, movieId, tag, timestamp.
"""

# Import all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from funk_svd.dataset import fetch_ml_ratings
from funk_svd import SVD
from sklearn.metrics import mean_absolute_error


"""
Data Analysis
"""

# Read Movie Data
i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('ml-100k/u.item',  sep='|', names=i_cols, encoding='latin-1')

movies.head()

num_movies = len(movies)

print(num_movies)


# user
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

users.head()

num_users = len(users)

print(num_users)




# Ratings
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=r_cols, encoding='latin-1')

ratings.head()

ratings.describe()

print(len(ratings.user_id.unique()))
print(len(ratings.movie_id.unique()))


"""
The Sparsity of a matrix is measured by the number of cells that do not have a value.
As it can be seen bellow, the matrix of ratings in this dataset is going to be very sparse, having a sparsity of 93.7%,
which means that the majority of users only rated a small percentage of the movies.
"""

sparsity = 1 - len(ratings) / (num_users * num_movies)

print(f"Sparsity: {sparsity:.3f}")


"""

"""

plt.hist(ratings.rating, ec='black', bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.title("Distribution of Ratings")
plt.xticks([1, 2, 3, 4, 5])
plt.show()


"""
Construct Matrix
"""

ratings_matrix = ratings.pivot(index="user_id", columns="movie_id", values="rating")

print(ratings_matrix.shape)
print(ratings_matrix.iloc[:5, :5])


"""
As it can be seen above, there are NaN entries on the matrix, which need to be replaced by some other value in order to perform the SVD.
There are several approaches to this problem, such as replacing with zero value, or average of all ratings, or even the average rating of a user.
"""

ratings_matrix = ratings_matrix.fillna(0)
print(ratings_matrix.iloc[:5, :5])

"""
The sparsity mentioned above, can be verified by the percentage of zeros present in the matrix.
"""

sparsity = 1 - np.count_nonzero(ratings_matrix) / (num_users * num_movies)
print(f"Sparsity: {sparsity:.3f}")


"""
# SVD

Opa ya é o svd e tal

"""

U, S, VT = np.linalg.svd(ratings_matrix, full_matrices=False)

print(f"U: {pd.DataFrame(U).iloc[:5, :5]}")
print(f"S: {pd.DataFrame(S).iloc[:5, :]}")
print(f"VT: {pd.DataFrame(VT).iloc[:5, :5]}")
print(U.shape)
print(S.shape)
print(VT.shape)


"""
# SVD Matrix Reconstruction

"""
reconstructed_matrix = np.dot(U * S, VT)

print(reconstructed_matrix[1][:4])
print(ratings_matrix.iloc[1, :4].to_numpy())

mae = np.average(np.absolute(ratings_matrix - reconstructed_matrix))
print("Reconstruction Error: ", mae)

"""
Truncated SVD
dimensionality reduction
"""

reconstructed_matrix = np.dot(U[:,:10] * S[:10], VT[:10,:])
mae = np.average(np.absolute(ratings_matrix - reconstructed_matrix))
print("Reconstruction Error: ", mae)


"""
Funk SVD
"""
df = fetch_ml_ratings(variant='100k')

train = df.sample(frac=0.8, random_state=7)
val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
test = df.drop(train.index.tolist()).drop(val.index.tolist())

svd = SVD(lr=0.001, reg=0.005, n_epochs=100, n_factors=15, early_stopping=True,
	shuffle=False, min_rating=1, max_rating=5)

svd.fit(X=train, X_val=val)


"""
MAE RMSE w/e
"""

pred = svd.predict(test)
mae = mean_absolute_error(test['rating'], pred)
print(f'Test MAE: {mae:.2f}')


"""
Hyperparameters Tunning
"""

# lrs = [0.0001, 0.001, 0.01, 0.1]
# factors = [5, 10, 15, 20, 25, 30]

# mae_outputs = {}
# for lr in lrs:
# 	for n in factors:
# 		svd = SVD(lr=lr, n_epochs=100, n_factors=n, early_stopping=False,
# 			shuffle=False, min_rating=1, max_rating=5)

# 		svd.fit(X=train, X_val=val)

# 		pred = svd.predict(test)
# 		mae = mean_absolute_error(test['rating'], pred)
# 		mae_outputs.setdefault(lr, [])
# 		mae_outputs[lr].append(mae)
# 		print(f'Latent Factors: {n}\nLearning Rate: {lr}\nTest MAE: {mae:.2f}')

# for lr in mae_outputs:
# 	plt.plot(factors, mae_outputs[lr], label=f"lr={lr}")

# plt.ylabel("MAE")
# plt.xlabel("Number of Latent Factors")
# plt.xticks(factors)
# plt.title("Funk SVD MAE with different Latent Factors and Learning Rates")
# plt.legend()
# plt.show()


# opt_lr, lr_n_factors = min(mae_outputs.items(), key=lambda x: min(x[1]))
# opt_n_factor = factors[lr_n_factors.index(min(lr_n_factors))]

# print(f"Optimal Learning Rate: {opt_lr}")
# print(f"Optimal Number of Latent Factors: {opt_n_factor}")

"""
Predict for a rating of a user for non rated movies example
"""
# Train SVD with optimal hyperparameters calculated previously
svd = SVD(lr=0.01, n_epochs=100, n_factors=10, early_stopping=False,
			shuffle=False, min_rating=1, max_rating=5)

svd.fit(X=train, X_val=val)

user_id = 1
# hand selected a movie that was rated by the user
movie_id_rated = 1
# hand selected a movie that wasn't rated by the user
movie_id_unrated = 1682

# initialize prediction to global mean rating
pred = svd.global_mean_

# getting index assigned to the user_id by the Funk SVD
u_ix = svd.user_mapping_[user_id]

# adding the bias associated with this user id
pred += svd.bu_[u_ix]

# getting index assigned to the movie_id by the Funk SVD
i_ix_unrated = svd.item_mapping_[movie_id_unrated]
i_ix_rated = svd.item_mapping_[movie_id_rated]

# adding the bias associated with this user id
pred_unrated = pred + svd.bi_[i_ix_unrated]
pred_rated = pred + svd.bi_[i_ix_rated]

# dot product between the associated user's and movie's latent factors
pred_unrated += np.dot(svd.pu_[u_ix], svd.qi_[i_ix_unrated])
pred_rated += np.dot(svd.pu_[u_ix], svd.qi_[i_ix_rated])

print(f"For user id: {user_id}")
print(f"Rating prediction for a rated movie {movie_id_rated}: {pred_rated:.1f}, Actual Rating: {ratings_matrix.iloc[user_id - 1, movie_id_rated - 1]}")
print(f"Rating prediction for an unrated movie {movie_id_unrated}: {pred_unrated:.1f}")


"""
Similarity Analysis

Calculate cosine similarity, sort by most similar and return the top N.
"""

def top_cosine_similarity(data, item_id, top_n=10):
    index = item_id - 1 #Ids starts from 1 in the dataset
    row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
        movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])

# Function to print top N similar users
def print_similar_users(user_data, user_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
        user_data[user_data.user_id == user_id]))
    for id in top_indexes + 1:
        print(user_data[user_data.user_id == id])


movie_id = 1
user_id = 1
top_n = 5
top_indexes_movies = top_cosine_similarity(VT, movie_id, top_n)
print_similar_movies(movies, movie_id, top_indexes_movies)

for i in top_indexes_movies:
	plt.plot(U[i,0], U[i,1], 'o')
	plt.annotate(movies[movies.movie_id == i+1].title.values[0], (U[i,0], U[i,1]))

plt.ylabel("Latent Factor 1")
plt.xlabel("Latent Factor 2")
plt.title("Normal SVD")
plt.legend()
plt.show()

top_indexes_users = top_cosine_similarity(U, user_id, top_n)
print_similar_users(users, user_id, top_indexes_users)




top_indexes_movies = top_cosine_similarity(svd.qi_, movie_id, top_n)
print_similar_movies(movies, movie_id, top_indexes_movies)

for i in top_indexes_movies:
	plt.plot(svd.qi_[i,0], svd.qi_[i,1], 'o')
	plt.annotate(movies[movies.movie_id == i+1].title.values[0], (svd.qi_[i,0], svd.qi_[i,1]))

plt.ylabel("Latent Factor 1")
plt.xlabel("Latent Factor 2")
plt.title("Funk SVD")
plt.legend()
plt.show()

top_indexes_users = top_cosine_similarity(svd.pu_, user_id, top_n)
print_similar_users(users, user_id, top_indexes_users)