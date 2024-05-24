import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load your datasets
titles_df = pd.read_csv('data/titles.csv')
credits_df = pd.read_csv('data/credits.csv')

# Data cleaning and preprocessing
def clean_dataset(df):
    df = df.copy()
    df.dropna(subset=['imdb_score'], inplace=True)
    df['genres'] = df['genres'].apply(lambda x: eval(x) if pd.notnull(x) else [])
    df['production_countries'] = df['production_countries'].apply(lambda x: eval(x) if pd.notnull(x) else [])
    return df

cleaned_titles_df = clean_dataset(titles_df)

# Feature Engineering
def feature_engineering(data):
    genres_dummies = data['genres'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
    countries_dummies = data['production_countries'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
    enhanced_df = pd.concat([data, genres_dummies, countries_dummies], axis=1)
    return enhanced_df

enhanced_df = feature_engineering(cleaned_titles_df)

# Ensure only numeric columns are included
numeric_columns = enhanced_df.select_dtypes(include=['number']).columns.drop(['imdb_score', 'tmdb_score'])

# Prepare data for modeling
X = enhanced_df[numeric_columns]
y = enhanced_df['imdb_score']

# Handle missing values by imputing with the median value
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model and necessary data
joblib.dump(model, 'model/random_forest_model.pkl')
joblib.dump(enhanced_df['genres'].explode().unique(), 'model/all_genres.pkl')
joblib.dump(enhanced_df['production_countries'].explode().unique(), 'model/all_countries.pkl')
joblib.dump(numeric_columns, 'model/numeric_columns.pkl')
joblib.dump(imputer, 'model/imputer.pkl')  # Save the imputer to handle missing values during prediction