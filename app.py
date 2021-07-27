from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import difflib
import random
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

static_path="C:\\Users\\user\\Desktop\\my project\\BookRec\\static\\"

lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def get_book_id(book_title, metadata):
    """
    Gets the book ID for a book title based on the closest match in the metadata dataframe.
    """

    existing_titles = list(metadata['title'].values)
    closest_titles = difflib.get_close_matches(book_title, existing_titles)
    book_id = metadata[metadata['title'] == closest_titles[0]]['id'].values[0]
    return book_id

def get_book_info_all(book_id, metadata):
    """
    Returns some basic information about a book given the book id and the metadata dataframe.
    """

    book_info = metadata[metadata['book_id'] == book_id][['book_id',
                                                     'authors', 'title']]
    return book_info.to_dict(orient='records')


def get_book_info(book_id, metadata,rating):
    """
    Returns some basic information about a book given the book id and the metadata dataframe.
    """

    book_info = metadata[metadata['id'] == book_id][['id',
                                                     'authors', 'title']]
    book_info['score']=[rating]
    return book_info.to_dict(orient='records')


def predict_review(user_id, book_title, model, metadata):
    """
    Predicts the review (on a scale of 1-5) that a user would assign to a specific book.
    """

    book_id = get_book_id(book_title, metadata)
    review_prediction = model.predict(uid=user_id, iid=book_id)
    return review_prediction.est


def generate_recommendation(user_id, model, metadata, thresh=4):
    """
    Generates a book recommendation for a user based on a rating threshold. Only
    books with a predicted rating at or above the threshold will be recommended
    """

    book_titles = list(metadata['title'].values)
    random.shuffle(book_titles)
    rec_books=[]
    for book_title in book_titles:
        rating = predict_review(user_id, book_title, model, metadata)

        if rating >= thresh:
            book_id = get_book_id(book_title, metadata)
            rec_books.extend(get_book_info(book_id, metadata,rating))
            print(rating, book_title,len(rec_books))
            if len(rec_books)>=50:
                break
    return rec_books

@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/results',methods=['post'])
def results():
    user_id=request.form['user_id']
    csv_file=request.files['file']
    import time
    tm = time.strftime("%Y%m%d_%H%M%S")
    filename=tm+".csv"
    csv_file.save(static_path+"user_csv\\"+filename)

    recommended_books=[]
    recommended_books_id = []

    recommended_books2=[]
    recommended_books_id2 = []

    import pickle
    with open('model.p', 'rb') as fp:
        svd = pickle.load(fp)
    # svd.predict(uid=10, iid=100)
    books_metadata = pd.read_csv('static/books/books.csv')
    recommended_books.extend(generate_recommendation(user_id, svd, books_metadata))
    print("=====-------------------------========="+str(len(recommended_books)))
    print(recommended_books)
    for k in recommended_books:
        recommended_books_id.append(k['id'])

    #content based
    user_books = pd.read_csv('static/user_csv/'+filename)
    all_books = pd.read_csv('static/books/books.csv')
    read_books = user_books[user_books['Exclusive Shelf'] == 'read']

    read_books_title = read_books.values[:, 1]

    all_books_title = all_books.values[:, 10]
    all_bookss = []
    for k in all_books_title:
        all_bookss.append(k)

    for title in read_books_title:
        title = title.lower()
        all_bookss.append(title)
        try:
            TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
            tfidf = TfidfVec.fit_transform(all_bookss)
            vals = cosine_similarity(tfidf[-1], tfidf)
            vals = vals[0]
            print(vals)
            print(title)
            for i in range(len(vals)):
                kr = {}
                if vals[i] > .6 and title != all_bookss[i]:
                    if all_books.values[i, 1] not in recommended_books_id2:
                        kr['id'] = all_books.values[i, 1]
                        kr['title'] = all_books.values[i, 10]
                        kr['authors'] = all_books.values[i, 7]
                        kr['score2'] = vals[i]*10
                        recommended_books_id2.append(all_books.values[i, 1])
                        recommended_books2.append(kr)
                        print(len(recommended_books2),vals[i], all_bookss[i], kr)
                        if len(recommended_books2)>=50:
                            break
                    if len(recommended_books2) >= 50:
                        break
                if len(recommended_books2) >= 50:
                    break
        except Exception as e:
            print(e)
        del all_bookss[-1]

    print("########################")
    print(recommended_books)
    print("$$$$$$$$$$$$$$$$$$$$$$$")
    print(recommended_books2)

    df = pd.DataFrame(recommended_books, columns=['id', 'title','authors','score'])
    df2 = pd.DataFrame(recommended_books2, columns=['id', 'title','authors','score2'])
    # Combining the results by contentId
    recs_df = df.merge(df2,
                               how='outer',
                               left_on='id',
                               right_on='id').fillna(0.0)
    print("======MERGE=====")
    print(recs_df)
    recs_df['recStrengthHybrid'] = (recs_df['score'] * 2) \
                                   + (recs_df['score2'] * 1)
    print("======MERGE=====")
    print(recs_df['recStrengthHybrid'])
    topn=10
    print("*********FINAL************")
    recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)
    print(recommendations_df)

    final_out=[]
    for i1 in range(topn):
        try:
            k=get_book_info_all(recommendations_df.values[i1,0],books_metadata)
            final_out.extend(k)
        except Exception as e:
            print("eee")

    return render_template('results.html',data=final_out)


if __name__ == '__main__':
    app.run()
