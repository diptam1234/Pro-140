from types import NoneType
import pandas as pd 
import numpy as np 
import plotly.express as pe

# ---------------------------------------------------------

df1=pd.read_csv('tmdb_5000_credits.csv')
df2=pd.read_csv('tmdb_5000_movies.csv')

df1.head()
df2.head()

df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')

df2.head(5)


# ------------------------------- class 139 ---------------------------------------

# IMDb Formula For Getting Metric Score :((v/(v+m))*R)+((m/(v+m))*C)

# ● v - The number of votes for the movies (or number of ratings/reviews in case of an amazon product)
# ● m - The minimum votes required to be listed in the chart
# ● R - Average rating of the movie
# ● C - Mean votes across the whole report


c = df2['vote_average'].mean()
print(c)

m = df2['vote_count'].quantile(0.9)
print(m)

q_movies = df2.copy().loc[df2['vote_count'] >= m]
print(q_movies)

def weighted_rating(x , m=m , C=c):
    v = x['vote_count']
    R = x['vote_average']
    return ((v/(v+m))*R)+((m/(v+m))*C)

q_movies['score'] = q_movies.apply(weighted_rating , axis=1)

q_movies = q_movies.sort_values('score',ascending = False)
q_movies[['title','vote_count','vote_average','score']].head(10)

fig = pe.bar( (q_movies.head(10).sort_values('score',ascending = True) ) , x = "score",y = "title",orientation = 'h')
fig.show()

# -------------------------------------------------- class 140 [Content Based Filtering] --------------------------------------------------------

df2[['title', 'cast', 'crew', 'keywords', 'genres']].head()


from ast import literal_eval

features =['cast', 'crew', 'keywords', 'genres']

for i in features:
    df2[i] = df2[i].apply(literal_eval)

df2.dtypes 

# ---------------------------------------------------------------
def get_director(a):
    for i in a:
        if i['job'] == 'Director':
            return i['name']

    return np.nan


df2['director'] = df2['crew'].apply(get_director)


# -------------------------------------------------------
def getList(x):
    if isinstance(x , list):
        names = [i['name'] for i in x]
        return names
    return []

features = ['cast' , 'keyword' , 'genres']

for i in features:
    df2[i] = df2[i].apply(getList)

# isinstance will help u to check that if the value of a column is a list or not
# -----------------------------------------------------------------------------------
df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)

# --------------------------------------------------------------------------------

def cleanData(x):
    if isinstance(x , list):
        return [str.lower(i.replace("",""))for i in x]
    else:
        if isinstance(x,str):
            return str.lower(x.replace("",""))
        else:
            return ''    

  
features =['cast', 'keywords','director' , 'genres']

for i in features:
    df2[i] = df2[i].apply(cleanData)
          

# ---------------------------------------------------
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast'])  + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup , axis = 1)

# -----------------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])


from sklearn.metrics.pairwise import cosine_similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)



df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])


def get_recommendations(title, cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices]


get_recommendations('Fight Club', cosine_sim2)

get_recommendations('The Shawshank Redemption', cosine_sim2)

get_recommendations('The Godfather', cosine_sim2)
