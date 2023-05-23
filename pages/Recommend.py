import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction, explode, desc
from pyspark.sql.types import StringType, ArrayType
from pyspark.mllib.recommendation import ALS
from pyspark.sql import Row

import pyspark as spark
spark = SparkSession \
    .builder \
    .master("local") \
    .appName("book_recommendation") \
    .getOrCreate()
sc = spark.sparkContext

books = spark.read.csv('D:/learning/Big_data/goodbooks-10k/books.csv', header=True, inferSchema=True)
ratings = spark.read.csv('D:/learning/Big_data/goodbooks-10k/ratings.csv', header=True, inferSchema=True)

rating_data = sc.pickleFile('rating_data')


train, validation, test = rating_data.randomSplit([6, 2, 2], seed=99)
# cache data
train.cache()
validation.cache()
test.cache()

def get_bookId(df_movies, fav_book_list):
    bookId_list = []
    for book in fav_book_list:
        bookIds = df_movies \
            .filter(books.title.like('%{}%'.format(book))) \
            .select('book_id') \
            .rdd \
            .map(lambda r: r[0]) \
            .collect()
        bookId_list.extend(bookIds)
    return list(set(bookId_list))



def add_new_user_to_data(train_data, bookId_list, spark_context):
    new_user_id = train_data.map(lambda r: r[1]).max() + 1
    new_user_rows = spark_context.parallelize([(new_user_id, bookId, 5.0) for bookId in bookId_list])
    train_data = train_data.union(new_user_rows)
    return train_data
def get_inference_data(train_data, df_books, bookId_list):
    new_user_id = train_data.map(lambda r: r[1]).max() + 1
    return df_books.rdd \
        .map(lambda r: r[1]) \
        .distinct() \
        .filter(lambda x: x not in bookId_list) \
        .map(lambda x: (new_user_id, x))

def make_recommendation(best_model_params, ratings_data, df_books, 
                        fav_book_list, n_recommendations, spark_context):
    bookId_list = get_bookId(df_books, fav_book_list)
    train_data = add_new_user_to_data(ratings_data, bookId_list, spark_context)

    model = ALS.train(train_data, best_model_params['rank'],
                        iterations=best_model_params['iterations'],
                        lambda_=best_model_params['lambda_'],
                        seed=99)
    inference_data = get_inference_data(train_data, df_books, bookId_list)
    predictions = model.predictAll(inference_data)
    recommendations = predictions \
        .sortBy(lambda x: x[2], ascending=False) \
        .map(lambda x: (x[1], x[2])) \
        .take(n_recommendations)
    recommendations = [(r[0],
                        df_books.filter(df_books.book_id == r[0])
                        .select('image_url').collect()[0][0],
                        df_books.filter(df_books.book_id == r[0])
                        .select('title').collect()[0][0],
                        df_books.filter(df_books.book_id == r[0])
                        .select('authors').collect()[0][0],
                        r[1]) for r in recommendations]
    return recommendations

st.title('Book Recommendation System')
st.write('This is a book recommendation system based on collaborative filtering. It uses the ALS algorithm to predict the ratings of books that the user has not rated yet. The predictions are based on the ratings of other users who have similar ratings with the user. The top n books with the highest predicted ratings are recommended to the user.')

fav_book_1 = st.sidebar.text_input('Enter your favorite book 1')

my_favorite_book = [fav_book_1]

if st.sidebar.button('Recommend'):
    recommendations = make_recommendation(
        best_model_params={'iterations': 10, 'rank': 20, 'lambda_': 0.05}, 
        ratings_data=rating_data, 
        df_books=books, 
        fav_book_list=my_favorite_book, 
        n_recommendations=10, 
        spark_context=sc)
    
    rec_df = pd.DataFrame(recommendations, columns=['book_id', 'image_url', 'title', 'authors', 'rating'])

    st.header('Recommendations')
    for i in range(len(rec_df)):
        book_id = rec_df['book_id'][i]
        st.subheader('Recommendation {}'.format(i+1))
        col1, col2 = st.columns(2)
        with col1:
            st.image(rec_df['image_url'][i], width=200)

        with col2:
            st.write('Book title: {}'.format(rec_df['title'][i]))
            st.write('Author: {}'.format(rec_df['authors'][i]))
            st.write('Our rating: {}'.format(rec_df['rating'][i]))
            st.write('Link to book: https://www.goodreads.com/book/show/{}'.format(book_id))
