import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

class MovieDataLoader:
    def __init__(self, movies_file, credits_file):
        self.movies_df = pd.read_csv(movies_file)
        self.credits_df = pd.read_csv(credits_file)
        self.merge_datasets()

    def merge_datasets(self):
        self.movies_df = self.movies_df.merge(self.credits_df, on='title')

class MovieDataProcessor(MovieDataLoader):
    def __init__(self, movies_file, credits_file):
        super().__init__(movies_file, credits_file)
        self.process_data()

    def process_data(self):
        self.movies_df = self.movies_df[['movie_id', 'title', 'overview', 'keywords', 'cast', 'crew','genres']] #only displaying columns that are important for filtering
        self.movies_df.dropna(inplace=True)
        self.movies_df['genres'] = self.movies_df['genres'].apply(self.convert)
        self.movies_df['keywords'] = self.movies_df['keywords'].apply(self.convert)
        self.movies_df['cast'] = self.movies_df['cast'].apply(self.convert_cast)
        self.movies_df['crew'] = self.movies_df['crew'].apply(self.fetch_director)

    def convert(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def convert_cast(self, obj):
        L = []
        for i in ast.literal_eval(obj)[:3]:  # Take only top 3 cast members
            L.append(i['name'])
        return L

    def fetch_director(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
        return L

class MovieRecommender(MovieDataProcessor):
    def __init__(self, movies_file, credits_file):
        super().__init__(movies_file, credits_file)
        self.create_similarity_matrix()

    def create_similarity_matrix(self):
        self.preprocess_text_data()
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(self.movies_df['tags']).toarray()
        self.similarity = cosine_similarity(vectors)

    def preprocess_text_data(self):
        self.movies_df['overview'] = self.movies_df['overview'].apply(lambda x: x.split())
        columns_to_combine = ['overview', 'genres', 'keywords', 'cast', 'crew']
        self.movies_df['tags'] = self.movies_df.apply(lambda row: ' '.join([' '.join(row[col]) for col in columns_to_combine]), axis=1)
        self.movies_df['tags'] = self.movies_df['tags'].apply(lambda x: x.lower())
        ps = PorterStemmer()
        self.movies_df['tags'] = self.movies_df['tags'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))

    def recommend(self, movie):
        if movie not in self.movies_df['title'].values:
            return "Movie not found in the dataset."
        movie_index = self.movies_df[self.movies_df['title'] == movie].index[0]
        distances = self.similarity[movie_index]
        movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        return [self.movies_df.iloc[i[0]].title for i in movie_indices]

if __name__ == "__main__":
    file_path_movies = 'movies.csv'
    file_path_credits = 'credits.csv'
    recommender = MovieRecommender(file_path_movies, file_path_credits)
    movie_title = input("Enter a movie title for recommendations: ")
    recommendations = recommender.recommend(movie_title)
    if isinstance(recommendations, list):
        print("Recommendations for {}:".format(movie_title))
        for title in recommendations:
            print(title)
    else:
        print(recommendations)
