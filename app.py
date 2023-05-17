from flask import Flask, render_template, request
from pyspark.ml.recommendation import ALSModel
import pyspark as ps
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import col
import numpy as np

app = Flask(__name__)

spark = ps.sql.SparkSession.builder \
            .master("local") \
            .appName("book_recommendation") \
            .getOrCreate()

sc = spark.sparkContext
sqlContext = SQLContext(sc)

# Load ratings and books dataframe
ratings_df = spark.read.csv('goodbooks-10k/ratings.csv', header=True, inferSchema=True)
books_df = spark.read.csv('goodbooks-10k/books.csv', header=True, inferSchema=True)

# Train the model
training_df, test_df = ratings_df.randomSplit([0.8, 0.2])
iterations = 10
regularization_parameter = 0.1
ranks = 5
als = ALS(maxIter=iterations, regParam=regularization_parameter, rank=ranks, userCol="user_id", itemCol="book_id", ratingCol="rating")
model = als.fit(training_df)

# Generate top 50 books by average rating
ranked_books = ratings_df.groupBy("book_id").agg({"rating": "avg"}).withColumnRenamed("avg(rating)", "avg_rating").orderBy(col("avg_rating").desc())
top_books = ranked_books.join(books_df, "book_id").select("book_id", "title", "authors", "image_url").limit(50).toPandas()

@app.route('/')
def index():
    return render_template("index.html", top_books=top_books)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    book_name = request.form['book_name']
    book_id = books_df.filter(books_df.title.like('%{}%'.format(book_name))).select('book_id').rdd.flatMap(lambda x: x).collect()
    if not book_id:
        return render_template("recommend.html", error="Sorry, no book found with this name")
    else:
        book_id = book_id[0]
        user_id = 1000
        new_user_ratings = [(user_id, book_id, 5)]
        new_user_ratings_df = spark.createDataFrame(new_user_ratings, ["user_id", "book_id", "rating"])
        ratings_df_with_new_user = ratings_df.union(new_user_ratings_df)
        new_model = als.fit(ratings_df_with_new_user)
        recommended_books_df = new_model.recommendForUserSubset(new_user_ratings_df.select("user_id"), 10).selectExpr("explode(recommendations.book_id) as book_id")
        recommended_books = recommended_books_df.join(books_df, "book_id").select("title", "authors", "image_url").toPandas()
        return render_template("recommend.html", recommended_books=recommended_books.values.tolist())

if __name__ == '__main__':
    app.run(debug=True)
