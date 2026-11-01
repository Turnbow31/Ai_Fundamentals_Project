Movie Rating Predictor

Overview:
This project uses a Random Forest Regression model to predict movie ratings based on historical movie information.
The model uses movie attributes such as genre, release year, rating classification, director, writer, lead actor, country, and runtime to estimate a movie's score.

Technologies Used:
Python 3
Pandas
Scikit-learn
Random Forest Regression

Project Structure:
Movie-Rating-Predictor/
├── movie_predictor_v1.py
├── movies_v1_training.csv
├── practice_test_results.csv
├── test_movies.csv
├── movies.csv
├── requirements.txt
└── README.md

Features:
Predict ratings for individual movies.
Predict ratings for multiple movies using CSV files.
Automatically evaluate model performance.
Calculate Mean Absolute Error (MAE).
Export prediction results to CSV files.

Installation:
Clone or download this repository.
Navigate to the Movie-Rating-Predictor directory.
Create a Python virtual environment:
python -m venv .venv
Activate the virtual environment:
source .venv/bin/activate
Install dependencies:
pip install -r requirements.txt

Running the Program
Run the following command:
python movie_predictor_v1.py
The program evaluates the model, trains the final model, and displays a menu with individual movie prediction and batch CSV prediction options.

Batch CSV Format:
Batch prediction files must include these columns:

genre
year
rating
director
writer
star
country
runtime

The name column is optional. The score column is optional and can be included to evaluate prediction accuracy.


Model Evaluation:
The program uses an 80/20 training and testing split and reports Mean Absolute Error (MAE).

Dataset
The project uses the Kaggle movie dataset:
https://www.kaggle.com/datasets/danielgrijalvas/movies
