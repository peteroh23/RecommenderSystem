from __future__ import (absolute_import, division, print_function,unicode_literals)
from surprise import Dataset
from surprise import AlgoBase
from surprise import SVD, SlopeOne, CoClustering, BaselineOnly, KNNBasic
from surprise import PredictionImpossible
from surprise import Reader
from surprise.model_selection import train_test_split
import numpy as np
import pandas as pd
from six import iteritems
import heapq
from random import *
from collections import defaultdict


# loading the movie-lens dataset

ratings_base = pd.read_csv('ml-latest-small/ratings.csv', sep=',', encoding='latin-1')


items = pd.read_csv('ml-latest-small/movies.csv', sep = ',', encoding = 'latin-1')

movie = items['title']

list1 = []
while len(list1) < 40:
        newUserItem = randrange(1,9126)

        if newUserItem not in list1:
            list1.append(newUserItem)
            
print ('\n Hi! This is a movie recommender system. \n Before we can recommend you movies, we need to get an understanding for your likes and dislikes. ')

print (' \n We need you to rate 20 different movies you may or may not have seen. \n If you have not seen the movie, please give a neutral rating of 3. \n ')

for i in range (0,40):

    if i < 20:
        movieID = list1[i]
        movieName = movie[movieID]
        rating1 = input('Movie: ' + movieName + '. Please rate this movie on a scale of 1-5. Please enter a NUMBER. ')
        rating = float(rating1)
        newRatingData = {'userId': [672], 'movieId': [movieID], 'rating': [rating], 'timestamp': [randrange(1260759131, 1280759131)]}
        newRating = pd.DataFrame(data = newRatingData)
        ratings_base = ratings_base.append(newRating)
    else:
        movieID = list1[i]
        movieName = movie[movieID]
        newRatingData = {'userId': [672], 'movieId': [movieID], 'rating': [3.0], 'timestamp': [randrange(1260759131, 1280759131)]}
        newRating = pd.DataFrame(data = newRatingData)
        ratings_base = ratings_base.append(newRating)



#converting pandas dataframe to Surprise data
ratings_dict = {'itemId': list(ratings_base.movieId),
                'userId': list(ratings_base.userId),
                'rating': list(ratings_base.rating)}

df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale = (0.5,5.0))

data = Dataset.load_from_df(df[['userId', 'itemId','rating']], reader)


#training and testing dataset
trainset, testset = train_test_split(data)

print('')

# top-N recommendation for specific user
def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n



#################predictions using BaselineOnly

algo4 = BaselineOnly()
algo4.fit(trainset)

predictions4 = algo4.test(testset)
dictMovies4 = get_top_n(predictions4)
topMovies4 = dictMovies4.get(672)

print('\n Here are the top 5 recommendations based on Baseline algorithm! ')

for i in range(5):

    movieRecc4 = topMovies4[i]
    movieRawID4 = movieRecc4[0]
    movieName4 = movie[movieRawID4]
    print(str(i+1) + '. ' + movieName4 )


######################predicitons using Matrix-Factorization

algo1 = SVD()
algo1.fit(trainset)

predictions1 = algo1.test(testset)
dictMovies1 = get_top_n(predictions1)
topMovies1 = dictMovies1.get(672)

print('\n Here are the top 5 recommendations based on Matrix-Factorization! ')

for i in range(5):

    movieRecc1 = topMovies1[i]
    movieRawID1 = movieRecc1[0]
    movieName1 = movie[movieRawID1]
    print(str(i+1) + '. ' + movieName1 )

print('')

###################predictions using K-NN

algo = KNNBasic()
algo.fit(trainset)

predictions = algo.test(testset)
dictMovies = get_top_n(predictions)
topMovies = dictMovies.get(672)

print('\n Here are the top 5 recommendations based on K-NN! ')

for i in range(5):

    movieRecc = topMovies[i]
    movieRawID = movieRecc[0]
    movieName = movie[movieRawID]
    print(str(i+1) + '. ' + movieName )



#################predictions using Slope-One

algo2 = SlopeOne()
algo2.fit(trainset)

predictions2 = algo2.test(testset)
dictMovies2 = get_top_n(predictions2)
topMovies2 = dictMovies2.get(672)

print('\n Here are the top 5 recommendations based on Slope-One! ')

for i in range(5):

    movieRecc2 = topMovies2[i]
    movieRawID2 = movieRecc2[0]
    movieName2 = movie[movieRawID2]
    print(str(i+1) + '. ' + movieName2 )





#############predictions using Co-Clustering

algo3 = CoClustering()
algo3.fit(trainset)

predictions3 = algo3.test(testset)
dictMovies3 = get_top_n(predictions3)
topMovies3 = dictMovies3.get(672)

print('\n Here are the top 5 recommendations based on Co-Clustering! ')

for i in range(5):

    movieRecc3 = topMovies3[i]
    movieRawID3 = movieRecc3[0]
    movieName3 = movie[movieRawID3]
    print(str(i+1) + '. ' + movieName3 )

print('')
print('Thanks for using this movie recommender system. Bye! \n')
