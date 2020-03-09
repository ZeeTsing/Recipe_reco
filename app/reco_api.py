import pandas as pd
import numpy as np
import random
from scipy.sparse import coo_matrix
import pickle
from surprise import SVD
from collections import defaultdict

file = open("output/SVD_algo.pkl",'rb')
SVD_algo = pickle.load(file)

file = open("output/recipes_names.pkl",'rb')
rep_names = pickle.load(file)

file = open("output/rep_mtx.pkl",'rb')
rep_U = pickle.load(file)

def get_recipe_similar_score(iids, U = rep_U):
    users_to_rec = [iid for iid in range(U.shape[0]) if iid not in iids]
   
    user_sim_score = []

    for user in users_to_rec:
        user_sim_score.append(float(np.mean([np.dot(U[userid],U[user]) for userid in iids])))

    return users_to_rec,user_sim_score

def get_users_pred_score(iids,algo = SVD_algo,uid = 226571):
    
    # create the list to search in
    iid_to_test = [iid for iid in range(231637) if iid not in iids]
    # build data for surprise
    test_set = [[uid,iid,4.] for iid in iid_to_test]
    # predict
    predictions = algo.test(test_set)
    #get prediction
    pred_ratings = [pred.est for pred in predictions]
    # return top_n indexes
    return pred_ratings

def translate_recipe_names(results,rep_names = rep_names):
    return [pretty_text(rep_names[r]) for r in results]

def pretty_text (text):
    ''' This function takes in text and try to put it in a human readable format by putting back \' and making it capitalize
    '''
    text = text.replace(" s ","\'s ")
    text_split = text.split(" ")
    #print(text_split)
    text_split = [t.strip().capitalize() for t in text_split if t != '']
    #print(text_split)
    return " ".join(text_split)

def hybrid_model_reco(iids,n_reco = 10):
    reco_id,rep_sim_score = get_recipe_similar_score(iids)
    
    pred_ratings = get_users_pred_score(iids)
    
    final_rating = [(ss+pr)*0.5 for ss,pr in zip(rep_sim_score,pred_ratings)]
    
    final_rating = zip(reco_id,final_rating)
    
    final_rec = [i[0] for i in sorted(final_rating,key=lambda x: x[1],reverse=True)]

    return translate_recipe_names(final_rec[:n_reco])