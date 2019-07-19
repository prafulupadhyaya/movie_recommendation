# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:26:50 2019

@author: praful
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###########functions to get title or index
def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]
##########################################


df=pd.read_csv('movie_dataset.csv')
    #print(df.head())
    #print(df.columns)

features=['keywords','cast','genres','director']

for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    return row["keywords"] +" "+row["cast"]+" "+row["genres"]+" "+row["director"]

df["combined_features"] = df.apply(combine_features,axis=1)
    #print(df["combined_features"].values[0])
    #print(df["keywords"][0])
    
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

cosine_sim = cosine_similarity(count_matrix) 
movie_user_likes = "Space Dogs"
try:
    movie_index = get_index_from_title(movie_user_likes)
except:
    print("no movie found by this name")
    movie_index=77777777
    
#print(movie_index)
similar_movies =  list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

i=0
for element in sorted_similar_movies:
		print(get_title_from_index(element[0]))
		i=i+1
		if i>50:
			break
