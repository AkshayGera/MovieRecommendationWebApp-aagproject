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


md = pd.read_csv('./DB/movies_metadata.csv')

links_small = pd.read_csv('./DB/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]
