from __future__ import (absolute_import, division, print_function,unicode_literals)
from surprise import Dataset
from surprise import AlgoBase
from surprise import PredictionImpossible
from surprise import Reader
import numpy as np
import pandas as pd
from six import iteritems
import heapq
from random import *

"""
# loading the movie-lens 100k dataset
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

# The data is already split into training (base) and test
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')

#items data
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')
"""

#ratings_base = pd.read_csv('ml-latest-small/ratings.csv', sep=',', encoding='latin-1')

"""
items = pd.read_csv('ml-latest-small/movies.csv', sep = ',', encoding = 'latin-1')

movie = items['title']

list1 = []
while len(list1) < 20:
        newUserItem = randrange(1,9126)

        if newUserItem not in list1:
            list1.append(newUserItem)
            


print (' \n We need you to rate 20 different movies you may or may not have seen. If you have not seen the movie, please give a neutral rating of 3. \n ')

for i in range (0,20):
    movieID = list1[i]
    movieName = movie[movieID]
    rating1 = input('Movie: ' + movieName + '. Please rate this movie on a scale of 1-5. Please enter a NUMBER. ')
    rating = float(rating1)
    newRatingData = {'userId': [672], 'movieId': [movieID], 'rating': [rating], 'timestamp': [randrange(1260759131, 1280759131)]}
    newRating = pd.DataFrame(data = newRatingData)
    ratings_base = ratings_base.append(newRating)

"""
"""
#converting pandas dataframe to Surprise data

ratings_dict = {'itemId': list(ratings_base.movieId),
                'userId': list(ratings_base.userId),
                'rating': list(ratings_base.rating)}

df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale = (0.5,5.0))

data = Dataset.load_from_df(df[['userId', 'itemId','rating']], reader)

#training dataset

trainset = data.build_full_trainset()
"""
##############################

data1 = Dataset.load_builtin('ml-100k')

trainset1 = data1.build_full_trainset()


# Collaborative Filtering Algorithm Classes from GitHub: https://github.com/NicolasHug/Surprise/blob/master/surprise/prediction_algorithms/knns.py


class SymmetricAlgo(AlgoBase):
    """This is an abstract class aimed to ease the use of symmetric algorithms.
    A symmetric algorithm is an algorithm that can can be based on users or on
    items indifferently, e.g. all the algorithms in this module.
    When the algo is user-based x denotes a user and y an item. Else, it's
    reversed.
    """

    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        return self

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff


class KNNBasic(SymmetricAlgo):
    """A basic collaborative filtering algorithm.
    The prediction :math:`\\hat{r}_{ui}` is set as:
    .. math::
        \hat{r}_{ui} = \\frac{
        \\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v) \cdot r_{vi}}
        {\\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v)}
    or
    .. math::
        \hat{r}_{ui} = \\frac{
        \\sum\\limits_{j \in N^k_u(i)} \\text{sim}(i, j) \cdot r_{uj}}
        {\\sum\\limits_{j \in N^k_u(j)} \\text{sim}(i, j)}
    depending on the ``user_based`` field of the ``sim_options`` parameter.
    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set the the global mean of all ratings. Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


class KNNWithMeans(SymmetricAlgo):
    """A basic collaborative filtering algorithm, taking into account the mean
    ratings of each user.
    The prediction :math:`\\hat{r}_{ui}` is set as:
    .. math::
        \hat{r}_{ui} = \mu_u + \\frac{ \\sum\\limits_{v \in N^k_i(u)}
        \\text{sim}(u, v) \cdot (r_{vi} - \mu_v)} {\\sum\\limits_{v \in
        N^k_i(u)} \\text{sim}(u, v)}
    or
    .. math::
        \hat{r}_{ui} = \mu_i + \\frac{ \\sum\\limits_{j \in N^k_u(i)}
        \\text{sim}(i, j) \cdot (r_{uj} - \mu_j)} {\\sum\\limits_{j \in
        N^k_u(i)} \\text{sim}(i, j)}
    depending on the ``user_based`` field of the ``sim_options`` parameter.
    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the neighbor
            aggregation is set to zero (so the prediction ends up being
            equivalent to the mean :math:`\mu_u` or :math:`\mu_i`). Default is
            ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               verbose=verbose, **kwargs)

        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb])
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        return est, details


class KNNBaseline(SymmetricAlgo):
    """A basic collaborative filtering algorithm taking into account a
    *baseline* rating.
    The prediction :math:`\\hat{r}_{ui}` is set as:
    .. math::
        \hat{r}_{ui} = b_{ui} + \\frac{ \\sum\\limits_{v \in N^k_i(u)}
        \\text{sim}(u, v) \cdot (r_{vi} - b_{vi})} {\\sum\\limits_{v \in
        N^k_i(u)} \\text{sim}(u, v)}
    or
    .. math::
        \hat{r}_{ui} = b_{ui} + \\frac{ \\sum\\limits_{j \in N^k_u(i)}
        \\text{sim}(i, j) \cdot (r_{uj} - b_{uj})} {\\sum\\limits_{j \in
        N^k_u(j)} \\text{sim}(i, j)}
    depending on the ``user_based`` field of the ``sim_options`` parameter. For
    the best predictions, use the :func:`pearson_baseline
    <surprise.similarities.pearson_baseline>` similarity measure.
    This algorithm corresponds to formula (3), section 2.2 of
    :cite:`Koren:2010`.
    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the neighbor
            aggregation is set to zero (so the prediction ends up being
            equivalent to the baseline). Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options. It is recommended to use the :func:`pearson_baseline
            <surprise.similarities.pearson_baseline>` similarity measure.
        bsl_options(dict): A dictionary of options for the baseline estimates
            computation. See :ref:`baseline_estimates_configuration` for
            accepted options.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, bsl_options={},
                 verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               bsl_options=bsl_options, verbose=verbose,
                               **kwargs)

        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.bu, self.bi = self.compute_baselines()
        self.bx, self.by = self.switch(self.bu, self.bi)
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        est = self.trainset.global_mean
        if self.trainset.knows_user(u):
            est += self.bu[u]
        if self.trainset.knows_item(i):
            est += self.bi[i]

        x, y = self.switch(u, i)

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            return est

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                nb_bsl = self.trainset.global_mean + self.bx[nb] + self.by[y]
                sum_ratings += sim * (r - nb_bsl)
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # just baseline again

        details = {'actual_k': actual_k}
        return est, details


class KNNWithZScore(SymmetricAlgo):
    """A basic collaborative filtering algorithm, taking into account
        the z-score normalization of each user.
    The prediction :math:`\\hat{r}_{ui}` is set as:
    .. math::
        \hat{r}_{ui} = \mu_u + \sigma_u \\frac{ \\sum\\limits_{v \in N^k_i(u)}
        \\text{sim}(u, v) \cdot (r_{vi} - \mu_v) / \sigma_v} {\\sum\\limits_{v
        \in N^k_i(u)} \\text{sim}(u, v)}
    or
    .. math::
        \hat{r}_{ui} = \mu_i + \sigma_i \\frac{ \\sum\\limits_{j \in N^k_u(i)}
        \\text{sim}(i, j) \cdot (r_{uj} - \mu_j) / \sigma_j} {\\sum\\limits_{j
        \in N^k_u(i)} \\text{sim}(i, j)}
    depending on the ``user_based`` field of the ``sim_options`` parameter.
    If :math:`\sigma` is 0, than the overall sigma is used in that case.
    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the neighbor
            aggregation is set to zero (so the prediction ends up being
            equivalent to the mean :math:`\mu_u` or :math:`\mu_i`). Default is
            ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)

        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)

        self.means = np.zeros(self.n_x)
        self.sigmas = np.zeros(self.n_x)
        # when certain sigma is 0, use overall sigma
        self.overall_sigma = np.std([r for (_, _, r)
                                     in self.trainset.all_ratings()])

        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])
            sigma = np.std([r for (_, r) in ratings])
            self.sigmas[x] = self.overall_sigma if sigma == 0.0 else sigma

        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb]) / self.sigmas[nb]
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim * self.sigmas[x]
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        return est, details



# Training the KNN-basic algorithm

#algo = KNNBasic()
#algo.fit(trainset)

algo1 = KNNBasic()
algo2 = KNNBaseline()
algo3 = KNNWithMeans()
algo4 = KNNWithZScore()

algo1.fit(trainset1)
algo2.fit(trainset1)
algo3.fit(trainset1)
algo4.fit(trainset1)



uid = str(1)
iid = str(31)

pred = algo1.predict (uid, iid, verbose = True)
pred = algo2.predict (uid, iid, verbose = True)
pred = algo3.predict (uid, iid, verbose = True)
pred = algo4.predict (uid, iid, verbose = True)




