import pandas as pd
# pip install --upgrade --no-cache-dir https://get.graphlab.com/GraphLab-Create/2.1/jo299@cornell.edu/B7D7-8CF5-428E-2A56-4BED-6243-C49A-0823/GraphLab-Create-License.tar.gz
import graphlab
from random import *

# this is using u.users file to make columns
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

# this is using u.data file to make columns
r_cols = ['user_id', 'item_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

# this is using u.item file to make columns

i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

# The data is already split into training (base) and test
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')


# in order to utilize graphlab, we must convert the data into SFrames
train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)

"""
popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)
"""

# Collaborative Filtering model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

#Make Recommendations:
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=25)

newUser = pd.DataFrame([944, 23, "M", "student", 60645])


#Making a list of 20 random movies
list1 = []
while len(list1) < 20:
        newUserItem = randrange(1,1682)
        if newUserItem not in list1:
            list1.append(newUserItem)

newDF = pd.DataFrame()

#Adding new ratings to the user file
for i in range(0,19):
    #Random Rating
    newUserRating = randrange(1,6)
    newRating = pd.DataFrame([944, list1[i], newUserRating, randrange(8800000000, 8900000000)])
    newDF = newDF.append(newRating)

print(newDF.head(10))

train_data1 = graphlab.SFrame(ratings_base)


item_sim_model = graphlab.item_similarity_recommender.create(train_data1, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

#Make Recommendation to new User (90571):
item_sim_recomm = item_sim_model.recommend(users=range(1001, 1005),k=5)
item_sim_recomm.print_rows(num_rows=25)

