import csv
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import string
import uritools
import urlextract
from langdetect import detect
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

extractor = urlextract.URLExtract()
ps = PorterStemmer()


data = {'airline_sentiment':[],'text':[]}
airline_sentiment = []
corpus = []

with open('Tweets.csv', 'r', encoding='utf8') as f:
    tweets = csv.reader(f)
    for row in tweets:        
        data['airline_sentiment'].append(row[1])
        data['text'].append(row[10])
        corpus.append(row[10])

def clean(words):
    urls = extractor.find_urls(words+" ")
    for url in urls:
        words = words.replace(url,'')
    tknzr = TweetTokenizer()
    words = tknzr.tokenize(words)
    exclude = set(string.punctuation)
    words = [word.lower() for word in words if not word.lower() in exclude]
    words = [word.lower() for word in words 
            if not word in set(stopwords.words('english')) and not word.isdigit()]
    words = [ps.stem(word) for word in words]
    words = ' '.join(words)
    return words

def CleanWithoutFilter():
    corpus = []
    corpusText=''
    with open('Tweets.csv',  encoding='utf8') as File:
        spamreader = csv.reader(File)
        for row in spamreader:       
            corpusText =  clean(row[10])
            corpus.append(corpusText)
    return corpus

def similarity(docs):
    vectorizer = TfidfVectorizer()
    Docsdf = vectorizer.fit_transform(docs)
    Docsdf = (Docsdf * Docsdf.T).A
    a = 1
    b = 0
    for a in range(len(Docsdf)):
        for b in range(a):
            x = Docsdf[b][len(Docsdf)-a]
            if(x>0.9 and not (len(Docsdf)-a == b)):
                del docs[b]
                del data['airline_sentiment'][b]
                break
    return docs
    

def CleanWithFilter():
    corpus = []
    corpusText=''
    counter = 0
   
    with open('Tweets.csv',  encoding='utf8') as File:
        spamreader = csv.reader(File)
        for row in spamreader:  
            
            corpusText =  clean(row[10])
                      
            if(not(corpusText.__contains__("RT") or (len(corpusText )<20) or (detect(row[10])=="en"))):
                corpus.append(corpusText)
                counter+=1
            else:
                del data['airline_sentiment'][counter]
            
    corpusFinal = similarity(corpus)
    return corpusFinal

def vectorizerFunction(filterOrNoFilter = CleanWithoutFilter()):
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train, X_test, y_train, y_test = train_test_split(filterOrNoFilter,  data['airline_sentiment'], test_size = 0.2)
    vectorizer.fit(X_train)
    XTrain = vectorizer.transform(X_train)
    XTest = vectorizer.transform(X_test)
    return XTrain, XTest, y_train, y_test

def MNBClassifier():
    XTrain, XTest, y_train, y_test = vectorizerFunction(CleanWithoutFilter())
    clf = MultinomialNB(alpha = 1.0, class_prior = None, fit_prior = True)
    clf.fit(XTrain, y_train)
    predictions = clf.predict(XTest)
    score = f1_score(y_test, predictions, average = 'micro')  
    print(score)
    print(predictions)

def KNeighbourClassifiers():
    XTrain, XTest, y_train, y_test = vectorizerFunction(CleanWithoutFilter())
    neigh = KNeighborsClassifier(n_neighbors = 5)
    neigh.fit(XTrain, y_train) 
    predictions = neigh.predict(XTest)
    score = f1_score(y_test, predictions, average = 'micro')  
    print(score)
    print(predictions)

def RForestClassifiers():
    XTrain, XTest, y_train, y_test = vectorizerFunction(CleanWithFilter())
    clf = RandomForestClassifier(random_state = 0)
    clf.fit(XTrain, y_train)
    predictions = clf.predict(XTest)
    score = f1_score(y_test, predictions, average = 'micro')  
    print(score)
    print(predictions)

MNBClassifier()
