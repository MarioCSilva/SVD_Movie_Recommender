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
Split Data
"""
train = ratings.sample(frac=0.8)
test = ratings.drop(train.index.tolist())

print(train.shape)
print(test.shape)


"""
Construct Matrixes
"""

rating_matrix_train = train.pivot(index="user_id", columns="movie_id", values="rating")
rating_matrix_test = test.pivot(index="user_id", columns="movie_id", values="rating")
matrix_train = pd.DataFrame(rating_matrix_train.values)
matrix_test = pd.DataFrame(rating_matrix_train.values)

print(matrix_train.shape)
print(matrix_test.shape)

print(matrix_train.iloc[:5, :5])


"""
As it can be seen above, there are NaN entries on the matrix, which need to be replaced by some other value in order to perform the SVD.
There are several approaches to this problem, such as replacing with zero value, or average of all ratings, or even the average rating of a user.
"""

matrix_train = matrix_train.fillna(0)
matrix_test = matrix_test.fillna(0)


print(matrix_train.iloc[:5, :5])

"""
The sparsity mentioned above, can be verified by the percentage of zeros present in the matrix.
"""

sparsity = 1 - np.count_nonzero(matrix_train) / (num_users * num_movies)
print(f"Sparsity: {sparsity:.3f}")


"""
# SVD

Opa ya é o svd e tal

"""


U, S, V = np.linalg.svd(matrix_train)

print(f"U: {pd.DataFrame(U).iloc[:5, :5]}")
print(f"S: {pd.DataFrame(S).iloc[:5, :]}")
print(f"VT: {pd.DataFrame(V.transpose()).iloc[:5, :5]}")



"""
# SVD Predictions
explicar o predict
"""




"""
Truncated SVD
dimensionality reduction
"""




"""
Funk SVD
"""



"""
Predict for a rating of a user for non rated movies example
"""



"""
Hyperparameters Tunning
"""


"""
(Not Accuracies) MAE RMSE w/e
"""




"""
Similarity Analysis 
"""





def cross_validation():

    df = fetch_ml_ratings(variant='100k')

    train = df.sample(frac=0.8, random_state=7)
    val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
    test = df.drop(train.index.tolist()).drop(val.index.tolist())

    svd = SVD(lr=0.001, reg=0.005, n_epochs=500, n_factors=15, early_stopping=False,
            shuffle=False, min_rating=1, max_rating=5)

    svd.fit(X=train, X_val=val)

    pred = svd.predict(test)
    mae = mean_absolute_error(test['rating'], pred)

    print(f'Test MAE: {mae:.2f}')


def k_fold_cross_validation(k=5):
    df = fetch_ml_ratings(variant='100k')

    train = df.sample(frac=0.8, random_state=7)
    test = df.drop(train.index.tolist())
    
    fold_size = int(train.shape[0] / k)
    
    print(fold_size)

    svd = SVD(lr=0.001, reg=0.005, n_epochs=100, n_factors=15, early_stopping=True,
            shuffle=False, min_rating=1, max_rating=5)
    
    for i in range(k):
        val = train[fold_size*i: fold_size*(i+1)+1]
        train_ = train.drop(val.index.tolist())
        svd.fit(X=train_, X_val=val)

    pred = svd.predict(test)
    mae = mean_absolute_error(test['rating'], pred)

    print(f'Test MAE: {mae:.2f}')

