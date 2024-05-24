from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Load the model and necessary data
model = joblib.load('model/random_forest_model.pkl')
all_genres = joblib.load('model/all_genres.pkl')
all_countries = joblib.load('model/all_countries.pkl')
numeric_columns = joblib.load('model/numeric_columns.pkl')
imputer = joblib.load('model/imputer.pkl')

class IMDBScorePredictor:
    def __init__(self, model, all_genres, all_countries, numeric_columns, imputer):
        self.model = model
        self.all_genres = all_genres
        self.all_countries = all_countries
        self.numeric_columns = pd.Index(numeric_columns).unique()
        self.imputer = imputer

    def feature_engineering_single(self, data):
        genres_dummies = data['genres'].apply(lambda x: pd.Series(1, index=x)).reindex(columns=self.all_genres, fill_value=0)
        countries_dummies = data['production_countries'].apply(lambda x: pd.Series(1, index=x)).reindex(columns=self.all_countries, fill_value=0)
        enhanced_df = pd.concat([data.drop(columns=['genres', 'production_countries']), genres_dummies, countries_dummies], axis=1)
        enhanced_df = enhanced_df.loc[:, ~enhanced_df.columns.duplicated()]
        enhanced_df = enhanced_df.drop(columns=['imdb_score', 'tmdb_score'], errors='ignore')
        enhanced_df = enhanced_df.reindex(columns=self.numeric_columns, fill_value=0)
        return enhanced_df

    def predict_imdb_score(self, single_entry):
        single_entry_df = pd.DataFrame([single_entry])
        single_entry_enhanced_df = self.feature_engineering_single(single_entry_df)
        single_entry_imputed = self.imputer.transform(single_entry_enhanced_df)
        predicted_imdb_score = self.model.predict(single_entry_imputed)
        return predicted_imdb_score[0]

predictor = IMDBScorePredictor(model, all_genres, all_countries, numeric_columns, imputer)

@app.route('/')
def index():
    return render_template('index.html', all_genres=all_genres, all_countries=all_countries)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        single_entry = {
            'title': data['title'],
            'type': data['type'],
            'release_year': int(data['release_year']),
            'age_certification': data['age_certification'],
            'runtime': int(data['runtime']),
            'genres': data.getlist('genres'),
            'production_countries': data.getlist('production_countries')
        }
        predicted_imdb_score = predictor.predict_imdb_score(single_entry)
        return render_template('result.html', prediction=predicted_imdb_score, entry=single_entry)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)