
import pandas as pd

user_books = pd.read_csv('static/books/user_reads.csv')
all_books = pd.read_csv('static/books/books.csv')
read_books = user_books[user_books['Exclusive Shelf']=='read']


read_books_title=read_books.values[:,1]

all_books_title=all_books.values[:,10]
all_bookss=[]
for k in all_books_title:
    all_bookss.append(k)


import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
# tfidf_all = TfidfVec.fit_transform(all_books_title)
# print(tfidf_all)
# for title in all_books_title:
#     title=title.lower()
#     try:
#         print(title)
#         # TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
#         # tfidf = TfidfVec.fit_transform([title])
#         # all_books_title_vec.append(tfidf)
#     except Exception as e:
#         print(""+str(e))


for title in read_books_title:
    title=title.lower()
    all_bookss.append(title)
    try:
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(all_bookss)
        vals = cosine_similarity(tfidf[-1], tfidf)
        vals=vals[0]
        print(vals)
        print(title)
        for i in range(len(vals)):
            kr={}
            if vals[i]>.5 and title!=all_bookss[i]:
                # all_books.values[i, :]
                kr['id']=all_books.values[i, 1]
                kr['title']=all_books.values[i, 10]
                kr['authors']=all_books.values[i, 7]
                print(vals[i],all_bookss[i],kr)
    except Exception as e:
        print(e)
    del all_bookss[-1]

    print("---------------")
