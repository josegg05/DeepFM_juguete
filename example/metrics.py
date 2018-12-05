
import numpy as np
import math
from scipy import spatial
from decimal import Decimal
import pandas as pd

import config


def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


#ndcg------------------------------------------------------------------------------------------

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)

    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg


def ndcg_total_at_k(dfResult):
    user_ndcg = []
    for u_i in dfResult['user_id'].unique():
        recomendation_list = dfResult.loc[dfResult['user_id'] == u_i].sort_values(by='prediction', ascending=False).head(10)

        #ndcg = ndcg_at_k(recomendation_list['relevance'], 10)
        ndcg = ndcg_at_k(np.isin( recomendation_list.index, recomendation_list.loc[recomendation_list['relevance'] == 2].index, assume_unique=True).astype(int), 10)
        user_ndcg.append(ndcg)

    return np.mean(user_ndcg)


#map
def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r)!= 0

    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


def mean_average_precision_total(dfResult):
    user_map = []
    for u_i in dfResult['user_id'].unique():
        recomendation_list = dfResult.loc[dfResult['user_id'] == u_i].sort_values(by='prediction', ascending=False).head(10)

        #Hay que medir cuantos de los elementos recomendados son efectivamente relevantes
        rel_vector = [np.isin( recomendation_list.index, recomendation_list.loc[recomendation_list['relevance'] == 2].index, assume_unique=True).astype(int)]

        map10 = mean_average_precision(rel_vector)
        user_map.append(map10)

    return np.mean(user_map)

#novelty----------------------------------------------------------------------------------------
def nth_root(value, n_root):
    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)


def cosine_distance(set1, set2):
    return spatial.distance.cosine(set1, set2)


def minkowski_distance(x,y,p_value):
    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)


def novelty_total(df, df_game):
    user_novelties = []

    for u_i in df['user_id'].unique():
        novelty = 0
        #print(u_i)

        recomendation_list = df.loc[df['user_id'] == u_i].sort_values(by='prediction', ascending=False).head(10)

        for i, n in enumerate(recomendation_list.index):

            discount = 1 / math.log((n + 1) + 1, 2)
            item_i = recomendation_list.loc[[n]]
            item_i_rel = item_i['prediction'].values[0]
            id_i = item_i['item_id'].values[0]
            item_i_features = df_game.loc[df_game['QueryID'] == id_i]

            for j in np.delete(np.array(recomendation_list.index), i):
                item_j = recomendation_list.loc[[j]]
                item_j_rel = item_j['prediction'].values[0]
                id_j = item_j['item_id'].values[0]
                item_j_features = df_game.loc[df_game['QueryID'] == id_j]

                distance = cosine_distance(item_i_features.values[0][1:], item_j_features.values[0][1:])

                novelty += distance * discount * item_j_rel * item_i_rel

            user_novelties.append(novelty)

    return np.mean(user_novelties)


#diversity

def diversity_total(df, df_game):
    user_diversity = []

    for u_i in df['user_id'].unique():
        Diversity = 0.0
        recomendation_list = df.loc[df['user_id'] == u_i].sort_values(by='prediction', ascending=False)

        for i, n in enumerate(recomendation_list.index):

            discount_n = 1 / math.log((i + 1) + 1, 2)
            item_i = recomendation_list.loc[[n]]
            item_i_rel = item_i['prediction'].values[0]
            id_i = item_i['item_id'].values[0]
            item_i_features = df_game.loc[df_game['QueryID'] == id_i]

            for k, j in enumerate(np.array(recomendation_list.index)[:i]):
                discount_j = 1 / math.log((k + 1) + 1, 2)
                item_j = recomendation_list.loc[[j]]
                id_j = item_j['item_id'].values[0]
                item_j_features = df_game.loc[df_game['QueryID'] == id_j]

                distance = cosine_distance(item_i_features.values[0][1:], item_j_features.values[0][1:])

                Diversity += distance * discount_n * discount_j * item_i_rel

            user_diversity.append(2 * Diversity)

    print(np.mean(user_diversity))

