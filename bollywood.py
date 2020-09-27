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
from flask import Flask, render_template,request

#from surprise import Reader, Dataset, SVD, evaluate
import os
###for poster
import urllib.request
import json
import pandas as pd
from pandas.io.json import json_normalize
###
import warnings; warnings.simplefilter('ignore')



md = pd.read_csv('./DB/BollywoodMovieDetail.csv')

md['titlesm'] = md['title'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
def convert_imdb_url(imdb_id):
    url="https://www.imdb.com/title/"
    imdb_id=str(imdb_id)
    return str(url+imdb_id)
md['imdb_id_url'] = md['imdbId'].apply(convert_imdb_url)
'''
only for creating poster 
'''
def poster_path(m):
        t =m['imdbId']
        str1="https://api.themoviedb.org/3/find/"+t+"?api_key=7b10daf226762522e7497c18a1ff2f26&external_source=imdb_id"
        #"https://api.themoviedb.org/3/find/tt0111161?api_key=7b10daf226762522e7497c18a1ff2f26&external_source=imdb_id"
        #print(str1)
        with urllib.request.urlopen(str1) as url:
            response = url.read()
        charset = url.info(). get_content_charset('utf-8')  # UTF-8 is the JSON default
        data = json.loads(response.decode(charset))
        a=json_normalize(data['movie_results'])
        #print(a[['backdrop_path']])
        return ("https://image.tmdb.org/t/p/w500"+a['backdrop_path'])
'''
only for overview
'''
def overview(m):
        t =(m['imdbId'])
        str1="https://api.themoviedb.org/3/find/"+t+"?api_key=7b10daf226762522e7497c18a1ff2f26&external_source=imdb_id"
        #"https://api.themoviedb.org/3/find/tt0111161?api_key=7b10daf226762522e7497c18a1ff2f26&external_source=imdb_id"
        #print(str1)
        with urllib.request.urlopen(str1) as url:
            response = url.read()
        charset = url.info(). get_content_charset('utf-8')  # UTF-8 is the JSON default
        data = json.loads(response.decode(charset))
        a=json_normalize(data['movie_results'])
        #print(a[['overview']])
        return (a['overview'])

md['genre'] = md['genre'].astype(str).apply(lambda x:x.split('|'))
s = md.apply(lambda x: pd.Series(x['genre']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'genre'
gen_md = md.drop('genre', axis=1).join(s)
#print(gen_md)

def build_chart(genre):
    df = gen_md[gen_md['genre'] == genre]
    print(df)
    qualified = df[(df['hitFlop'].notnull())][['imdbId','title', 'releaseYear','actors', 'hitFlop','imdb_id_url']]
    qualified = qualified.sort_values('hitFlop', ascending=False).head(20)
    print((qualified))
    '''
    qualified['poster_path']=qualified.apply(poster_path,axis=1)
    qualified['overview']=qualified.apply(overview,axis=1)
    '''
    return qualified
smd = md
smd['writers'] = smd['writers'].astype(str).apply(lambda x:x.split('|'))
smd['actors'] = smd['actors'].astype(str).apply(lambda x:x.split('|'))
smd['directors'] = smd['directors'].astype(str).apply(lambda x:x.split('|'))
smd['soup'] =smd['actors'] + smd['directors']+ smd['genre'] + smd['writers']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

count_matrix = count.fit_transform(smd['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
smd = smd.reset_index()
titles = smd['titlesm']
indices = pd.Series(smd.index, index=smd['titlesm'])
#Main Prem Ki Diwani Hoon
def get_recommendationsMt(title):
    title=title.lower().replace(" ",'')
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    #a=titles.iloc[movie_indices]
    qualified = smd.iloc[movie_indices][['imdbId','genre','title', 'releaseYear','actors', 'hitFlop','imdb_id_url']]
    '''
    qualified['poster_path']=qualified.apply(poster_path,axis=1)
    qualified['overview']=qualified.apply(overview,axis=1)
    '''
    return qualified
    
    



def improve_recommendations(title):
    title=title.lower().replace(" ",'')
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    qualified = smd.iloc[movie_indices][['imdbId','title','genre', 'releaseYear','actors', 'hitFlop','imdb_id_url']]
    qualified = qualified.sort_values('hitFlop', ascending=False).head(20)
    '''###only for poster and overview
    qualified['poster_path']=qualified.apply(poster_path,axis=1)
    qualified['overview']=qualified.apply(overview,axis=1)
    '''#######
    return qualified

template_dir = os.path.abspath('../MRS/templates for bollywood')
app = Flask(__name__, template_folder=template_dir)

@app.route('/Home')
def Index():
   return render_template('index.html')

@app.route('/ByGenres',methods = ['POST', 'GET'])
def By_Genres():
   if request.method == 'POST':
     try:
      genres = request.form['genres']
      print(genres)
      out=build_chart(genres).head(20)
      print(out)
     #return render_template('result2.html',results = out)
      return render_template('result3.html',results = out,name=genres,method="Genre")     
     except KeyError:
      return render_template("genres.html",err='Enter Proper Moviename')
     except ValueError:
      return render_template("genres.html",err='Enter Proper Moviename')
   else:
      return render_template("genres.html")


@app.route('/ByMetadataBased',methods = ['POST', 'GET'])
def By_MetadataBased():
   if request.method == 'POST':
     try:
      movie = request.form['movie']
      print(movie)
      out=get_recommendationsMt(movie).head(20)
      print(out)
      #return render_template('result2.html',results = out)
      return render_template('result3.html',results = out,name=movie,method="Metadata")
     except KeyError:
      return render_template("metadatabased.html",err='Enter Proper Moviename')
     except ValueError:
      return render_template("metadatabased.html",err='Enter Proper Moviename')

   else:
      return render_template("metadatabased.html")


@app.route('/ByPopularityBased',methods = ['POST', 'GET'])
def By_PopularityBased():
   if request.method == 'POST':
     try:
      movie = request.form['movie']
      print(movie)
      out=improve_recommendations(movie).head(20)
      print(out)
      #return render_template('result2.html',results = out)
      return render_template('result3.html',results = out,name=movie,method="Popularity")
     except KeyError:
      return render_template("popularitybased.html",err='Enter Proper Moviename')
     except ValueError:
      return render_template("popularitybased.html",err='Enter Proper Moviename')
   else:
      return render_template("popularitybased.html")
    

@app.route('/MovieDetails',methods = ['POST', 'GET'])
def movie_details():
   try:
     
	   if request.method == 'POST':
	      movie = request.form['movie']
	      print(movie)
	      out1=improve_recommendations(movie).head(20)
	      out2=get_recommendationsMt(movie).head(20)
	      return render_template('movie.html',result1=out1,result2=out2,name=movie)

	   elif request.args.get('movie')!=None:
	      movie = request.args.get('movie')
	      print(movie)
	      out1=improve_recommendations(movie).head(20)
	      out2=get_recommendationsMt(movie).head(20)
	      
	      return render_template('movie.html',result1=out1,result2=out2,name=movie)

	   else:   	
	      return render_template("index.html")
   except KeyError:
    return render_template("index.html",err='Enter Proper Movie name')
   except ValueError:
    return render_template("index.html",err='Enter Proper Movie name')

if __name__ == '__main__':
   app.run(debug = True,port='4999')
