import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data(filename):
    plots = []
    genres = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:  
                continue
            parts = line.split(':::')
            if len(parts) < 3:  
                print(f"Skipping invalid line: {line}")  
                continue
            genre = parts[2].strip()  
            plot = parts[3].strip()
            plots.append(plot)
            genres.append(genre)
    return plots, genres

filename = "train_data.txt"
plots, genres = load_data(filename)

for genre, plot in zip(genres, plots):
    print(f"Genre: {genre}\nPlot: {plot}\n")

df = pd.DataFrame({'plot': plots, 'genre': genres})
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['plot'])
y = df['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


