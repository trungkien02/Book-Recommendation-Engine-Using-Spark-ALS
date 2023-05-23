import streamlit as st
import pickle

st.set_page_config(
    page_title="Book Recommendation System",
)
st.title('Book Recommendation System')

st.sidebar.success('Select "Recommendation" to start making recommendations')

with open('top_books.pkl', 'rb') as f:
    top_books = pickle.load(f)

st.header('Top 50 Books')
st.write('These are the top 50 books in the dataset. The ratings are based on the average ratings of all users.')
for i in range(len(top_books)):
    st.subheader('Top {}'.format(i+1))
    col1, col2 = st.columns(2)

    with col1:
        st.image(top_books['image_url'][i], width=150)
    with col2:
        st.write('Book title: {}'.format(top_books['title'][i]))
        st.write('Author(s): {}'.format(top_books['authors'][i]))
        st.write('Average rating: {}'.format(top_books['avg_rating'][i]))
        st.write('Link to book: https://www.goodreads.com/book/show/{}'.format(top_books['book_id'][i]))