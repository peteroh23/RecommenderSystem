import pandas as pd
import graphlab
from random import *

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

# The data is already split into training (base) and test
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')

# CSV of all Movies
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

movie = items['movie title']

#Inputting new user ratings based on interactive user

print('\n Hi! This is a Movie Recommender System. \n Based on the ratings that you give to the system, it will recommend you 5 movies to watch! \n Before we begin, there are several questions we need to ask you. \n ')


numberNewUser = input("How many times would you like to run this recommendation system? Please enter a NUMBER. ")



for x in range(numberNewUser):

    print('\n Trial number ' + str(x+1) + ' now begins.')
    
    list1 = []
    while len(list1) < 20:
            newUserItem = randrange(1,1683)

            if newUserItem not in list1:
                list1.append(newUserItem)
                


    print (' \n We need you to rate 20 different movies you may or may not have seen. If you have not seen the movie, please give a neutral rating of 3. \n ')

    for i in range (0,20):
        movieID = list1[i]
        movieName = movie[movieID]
        rating = input('Movie: ' + movieName + '. Please rate this movie on a scale of 1-5. Please enter a NUMBER. ')
        newRatingData = {'user_id': [944+x], 'movie_id': [movieID], 'rating': [rating], 'unix_timestamp': [randrange(8800000000, 8900000000)]}
        newRating = pd.DataFrame(data = newRatingData)
        ratings_base = ratings_base.append(newRating)

    #training SFrame dataset

    train_data = graphlab.SFrame(ratings_base)

    matrixFactorization1 = graphlab.recommender.ranking_factorization_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

    #Make Recommendations:
    MF_recomm1 = matrixFactorization1.recommend(users=[944+x],k=5)


    #movie_id list

    movie = items['movie title']

    print('\n Here are your top five movie recommendations!')

    for i in range(5):
        x = MF_recomm1['movie_id'][i]
        print(str(i +1) + '.' + movie[x])


print('\n We hope that this system was user-friendly and met your satisfaction. Thank you for your time! ')