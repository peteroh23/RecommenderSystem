# RecommenderSystem

#### Credit for Algorithms: Graphlab (github.com/apple/turicreate) and Surprise (surpriselib.com)
#### Credit for Datasets: MovieLens (grouplens.org/datasets/movielens)

This is a project that I have been working on that piggy-backs off of my REU research work at East Carolina University.

RS1 and RS2 are  movie recommender systems that make 5 distict recommendations based on a user's ratings of 20 different movies. RS3 and RS4 uses interactive data from the user–– allows the user to rate 20 different movies on a scale fo 1 to 5–– to product more personalized recommendations.In RS1-RS4, graphLab library is used for the recommender system algorithms.

SurpriseRS uses the Surprise library to incoporate 4 different kinds of K-NN algorithms: K-NN Basic, K-NN Baseline, K-NN w/ Means, and K-NN w/ Z Score. This python script predicts ratings of a specific user and a specific item. SurpriseRS2 also used the Surprise library. This recommender system asks the user to rate 20 different movies. Then, with the new user information, 5 different algorithms are used to produce 25 movie recommendations in total. The recommendation algorithms used are: Surprise Baseline, K-NN, Matrix-Factorization, Slope-One, Co-Clustering. 

The best movie recommender system: SurpriseRS2.py
