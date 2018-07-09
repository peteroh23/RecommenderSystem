import pandas as pd
import graphlab
from random import *

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

# The data is already split into training (base) and test
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')

#Manually inputing one new user (944)

list1 = []
while len(list1) < 20:
        newUserItem = randrange(1,1682)
        if newUserItem not in list1:
            list1.append(newUserItem)
            

#Adding new ratings to the user file
for i in range(0,20):
    #Random Rating
    newUserRating = randrange(1,6)
    newRatingData = {'user_id': [944], 'movie_id': [list1[i]], 'rating': [newUserRating], 'unix_timestamp': [randrange(8800000000, 8900000000)]}
    newRating = pd.DataFrame(data = newRatingData)
    ratings_base = ratings_base.append(newRating)


#training SFrame dataset

train_data = graphlab.SFrame(ratings_base)

matrixFactorization1 = graphlab.recommender.ranking_factorization_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

#Make Recommendations:
MF_recomm1 = matrixFactorization1.recommend(users=[944],k=5)


# movie ID for top 5 recommendations
print(MF_recomm1['movie_id'])

# CSV of all Movies
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

#movie_id list

movie = items['movie title']

print('Here are the top five movie recommendations!')

for i in range(5):
    x = MF_recomm1['movie_id'][i]
    print(str(i +1) + '.' + movie[x])