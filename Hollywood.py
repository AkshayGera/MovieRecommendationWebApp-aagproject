import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
#from surprise import Reader, Dataset, SVD, evaluate


from flask import Flask, render_template,request


import warnings; warnings.simplefilter('ignore')
######################################################################################################################
##############################           venv\scripts\activate

###########################################
md = pd.read_csv('./DB/movies_metadata.csv')

md['titlesm'] = md['title'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))

#print(md)

###########################adding url for poster
def convert_poster_url(poster_path):
    url="https://image.tmdb.org/t/p/w500"
    poster_path=str(poster_path)
    return str(url+poster_path)
md['poster_path'] = md['poster_path'].apply(convert_poster_url)

###########################adding url for imdb
def convert_imdb_url(imdb_id):
    url="https://www.imdb.com/title/"
    imdb_id=str(imdb_id)
    return str(url+imdb_id)
md['imdb_id'] = md['imdb_id'].apply(convert_imdb_url)


md['genres'] = md['genres'].apply(literal_eval).apply(lambda x: [i['name'] for i in x])
#print(md[md['vote_count'].notnull()]['vote_count'])
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
#print(C)
m = vote_counts.quantile(0.95)
#print(m)
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0])
qualified = md[(md['vote_count'] >= m) & (md['vote_average'].notnull()) ][['title', 'vote_count', 'vote_average', 'year','poster_path','release_date','overview','imdb_id','genres']]
#print(qualified)
#print(qualified.shape)

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


qualified['wr'] = qualified.apply(weighted_rating, axis=1)

qualified = qualified.sort_values('wr', ascending=False).head(250)
output= qualified[['title','wr','genres']]
#if list output=output.head(10).iloc[:, 0].tolist()
#output = output.as_matrix()
output=output.head(10)
#output=output.values

#print(output)

######################################################################################################################
##############################
###########################################
######################################################################################################################
##############################
###########################################
####################Genre based
#####################################################


s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)

def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'vote_count', 'vote_average', 'year','poster_path','release_date','overview','imdb_id','popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    ###########################
    output=qualified[['title', 'vote_count', 'vote_average', 'year','poster_path','release_date','overview','imdb_id']]
    print(output.head(20))
    
    
    ###############    
    return output

#build_chart('Romance').head(15)

########################################################
######################################################################################################################
##############################       Content based

links_small = pd.read_csv('./DB/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]
print(smd.shape)
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
tfidf_matrix.shape
cosine_simt = linear_kernel(tfidf_matrix, tfidf_matrix)
#cosine_simt[0]
smd = smd.reset_index()
titles = smd['titlesm']
indices = pd.Series(smd.index, index=smd['titlesm'])
def get_recommendations(title):
    title=title.lower().replace(" ",'')
    idx = indices[title]
    sim_scores = list(enumerate(cosine_simt[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    a=smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year','poster_path','release_date','overview','imdb_id']]

    return a


    #out=titles.iloc[movie_indices]
    ###############convert series to dataframe
    #df1 = pd.DataFrame(data=out.index, columns=['index'])
    #df2 = pd.DataFrame(data=out.values, columns=['title'])
    #df = pd.merge(df1, df2, left_index=True, right_index=True)
    #print(df)
    #return(df)
############get_recommendations('The Godfather').head(10)
#############get_recommendations('The Dark Knight').head(10)

###########################################
######################################################################################################################
##############################
#########################################################################################################################################
#########################
##########Metadata Based Recommender
########################


credits = pd.read_csv('./DB/credits.csv')
keywords = pd.read_csv('./DB/keywords.csv')
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')
##############md.shape
md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
smd = md[md['id'].isin(links_small)]
########################smd.shape

smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
def get_writer(x):
    for i in x:
        if i['job'] == 'Writer':
            return i['name']
    return np.nan
t=smd
smd['director'] = smd['crew'].apply(get_director)
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:5] if len(x) >=5 else x)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
###writter included
temp= smd
temp['writer']=t['crew'].apply(get_writer)
temp['writer']=temp['writer'].fillna('')
temp['writer'] = temp['writer'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
temp['writer'] = temp['writer'].apply(lambda x: [x,x])
smd=temp
###

smd['director'] = smd['director'].apply(lambda x: [x,x])

s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'

s = s.value_counts()

################################s[:5]
s = s[s > 1]
stemmer = SnowballStemmer('english')
stemmer.stem('dogs')
def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words
smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director']+ smd['genres'] + smd['writer'] 
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
smd = smd.reset_index()
titles = smd['titlesm']
indices = pd.Series(smd.index, index=smd['titlesm'])


def get_recommendationsByMeta(title):
    title=title.lower().replace(" ",'')
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    a = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year','poster_path','release_date','overview','imdb_id']]

    #a=titles.iloc[movie_indices]
  #  a.to_frame()
    return a

    #out=titles.iloc[movie_indices]
    ###############convert series to dataframe
    #df1 = pd.DataFrame(data=out.index, columns=['index'])
    #df2 = pd.DataFrame(data=out.values, columns=['title'])
    #df = pd.merge(df1, df2, left_index=True, right_index=True)
    #print(df)
   # return(a.head(10))
#get_recommendationsByMeta('The Dark Knight').head(10)
#get_recommendationsByMeta('Pulp Fiction').head(10)

###########################################
######################################################################################################################
###########################
###########################
#Popularity and Ratings
###########################
###########################
def improved_recommendations(title):
    title=title.lower().replace(" ",'')
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year','poster_path','release_date','overview','imdb_id']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    ####(.six0)
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(30)
    return qualified



#improved_recommendations('The Dark Knight')
#improved_recommendations('Pulp Fiction')


def movie_details(title):
 title=title.lower().replace(" ",'')
 return md[(md['titlesm']== title)]
    
	
##############################
###########################################
######################################################################################################################
##############################
###########################################
######################################################################################################################
##############################
###########################################
######################################################################################################################
##############################
###########################################
def output(title):
    #b=pd.read_csv('./output.csv',index_col=0)
    #print(b)
   # os.remove('./output.csv')
    a=improved_recommendations(title)[['title']].reset_index().drop('index',axis=1)
    a['popularity_based']=improved_recommendations(title)[['title']].reset_index().drop('index',axis=1)
    a['content_based']=get_recommendations(title).reset_index().drop('index',axis=1)
    a['metadata_based']=get_recommendationsByMeta(title).reset_index().drop('index',axis=1)
    a=(a.drop('title',axis=1).rename(index={0:title,1:title,2:title,3:title,4:title,5:title,6:title,7:title,8:title,9:title}))
    #b.append(a)
   # b.to_csv('output.csv')
    return a
def intocsv(title):
    b=pd.read_csv('./output.csv',index_col=0)
    b=b.append(output(title))
    os.remove('./output.csv')
    b.to_csv('output.csv')
    return b    
def intocsv2(title):
    #b=pd.read_csv('./output.csv',index_col=0)
    b=output(title)
    #os.remove('./output.csv')
    b.to_csv('./movies/'+title+'.csv')
    return b    

######################################################################################################################
##############################
###########################################
# convert your links to html tags 
def path_to_image_html(path):
    return '<img src="'+ path + '" width="100" >'
def link_for_imdb(path):
    return '<a href="'+ path + '"  >'





app = Flask(__name__)

@app.route('/Home')
def Index():
   return render_template('index.html')


@app.route('/ByRating')
def By_Rating():
   out=qualified[['title', 'vote_count', 'vote_average', 'year','poster_path','release_date','overview','imdb_id']]
   out.reset_index()
   return render_template('mainresult.html',results=out,method="Rating")

@app.route('/ByGenres',methods = ['POST', 'GET'])
def By_Genres():
   try:
     if request.method == 'POST':
      genres = request.form['genres']
      print(genres)
      out=build_chart(genres)#.head(15)
      print(out)
      return render_template('mainresult.html',results =out,name=genres,method="Genre")
     elif request.args.get('genres')!=None:
      genres = request.args.get('genres')
      print(genres)
      out=build_chart(genres)
      print(out)
      return render_template('mainresult.html',results =out,name=genres,method="Genres")
     else:
      return render_template("genres.html")
   except KeyError:
     return render_template("genres.html",err='Enter Proper Genre name(Our Dataset is limited)') 
   except ValueError:  
     return render_template("genres.html",err='Enter Proper Genre name(Our Dataset is limited)') 
   
@app.route('/ByContentBased',methods = ['POST', 'GET'])
def By_ContentBased():
   if request.method == 'POST':
     try:
      movie = request.form['movie']
      print(movie)
      out=get_recommendations(movie)#.head(10)
      print(out)
      return render_template('mainresult.html',results =out,name=movie,method="Content")
     except KeyError:
      return render_template("contentbased.html",err='Enter Proper Movie name(Our Dataset is limited)')
     except ValueError:
      return render_template("contentbased.html",err='Enter Proper Movie name(Our Dataset is limited)')
   else:
      return render_template("contentbased.html")


@app.route('/ByMetadataBased',methods = ['POST', 'GET'])
def By_MetadataBased():
   if request.method == 'POST':
     try:
      movie = request.form['movie']
      print(movie)
      out=get_recommendationsByMeta(movie)#.head(10)
      print(out)
      return render_template('mainresult.html',results =out,name=movie,method="Metadata")
     except KeyError:
      return render_template("metadatabased.html",err='Enter Proper Movie name(Our Dataset is limited)')
     except ValueError:
      return render_template("metadatabased.html",err='Enter Proper Movie name(Our Dataset is limited)')
   else:
      return render_template("metadatabased.html")


@app.route('/ByPopularityBased',methods = ['POST', 'GET'])
def By_PopularityBased():
   if request.method == 'POST':
     try:
      movie = request.form['movie']
      print(movie)
      
      out=improved_recommendations(movie)#.head(10)
      print(out)
      return render_template('mainresult.html',results =out,name=movie,method="Popularity")
     except KeyError:
      return render_template("popularitybased.html",err='Enter Proper Movie name(Our Dataset is limited)')
     except ValueError:
      return render_template("popularitybased.html",err='Enter Proper Movie name(Our Dataset is limited)')
   else:
      return render_template("popularitybased.html")

@app.route('/MovieDetails',methods = ['POST', 'GET'])
def moviedetails():
   try:
     
	   if request.method == 'POST':
	      movie = request.form['movie']
	      print(movie)
	      out=movie_details(movie)
	      print(out)
	      out1=improved_recommendations(movie).head(10)
	      out2=get_recommendationsByMeta(movie).head(10)
	      out3=get_recommendations(movie).head(10)
	      return render_template('movie.html',results =out,results1=out1,results2=out2,results3=out3,name=movie)

	   elif request.args.get('movie')!=None:
	      movie = request.args.get('movie')
	      print(movie)
	      out=movie_details(movie)
	      print(out)
	      out1=improved_recommendations(movie).head(10)
	      out2=get_recommendationsByMeta(movie).head(10)
	      out3=get_recommendations(movie).head(10)

	      return render_template('movie.html',results =out,results1=out1,results2=out2,results3=out3,name=movie)

	   else:   	
	      return render_template("index.html")
   except KeyError:
    return render_template("index.html",err='Enter Proper Movie name(Our Dataset is limited)')
   except ValueError:
    return render_template("index.html",err='Enter Proper Movie name(Our Dataset is limited)')


if __name__ == '__main__':
   app.run(debug = True, use_reloader=False)



