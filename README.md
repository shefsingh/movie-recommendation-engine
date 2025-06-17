# Python Movie Recommendation System

This is a simple content-based movie recommendation system built in Python. It uses metadata like cast, crew, genres, and keywords to suggest movies similar to a given title.

## Project Files

- **Python_Final_Project_Presentation.pptx**  
  A PowerPoint presentation explaining the project structure, methodology, and sample outputs.

- **Python_movie_recommendation.mp4**  
  A short demo video showing how the recommendation system works in action.

- **Shefali_Singh_Python_Project.zip**  
  Contains the complete project code, datasets, and scripts needed to run the movie recommender locally.

---

## ⚙️ How It Works

The project is divided into three key components:

### 1. `MovieDataLoader`
Loads and merges movie and credit datasets into a unified DataFrame for processing.

### 2. `MovieDataProcessor`
Cleans the data, extracts meaningful features (e.g., cast, director), and prepares the content for vectorization.

### 3. `MovieRecommender`
Uses text preprocessing and cosine similarity to recommend the top 5 most similar movies based on a user-input title.

---

## How to Run

1. Unzip the project files from `Shefali_Singh_Python_Project.zip`
2. Ensure you have Python and the required libraries installed:
   ```bash
   pip install pandas scikit-learn numpy
3. Run the main script from your terminal or IDE:
   python movie_recommender.py
4. Enter a movie title when prompted (case-sensitive, e.g., Batman Begins)
5. Watch the video demo: Python_movie_recommendation.mp4

*This project was created as part of my Python coursework and is my first GitHub upload. Feedback and suggestions are welcome!* 
