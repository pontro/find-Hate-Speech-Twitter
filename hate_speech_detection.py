import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from flask import Flask, render_template

app = Flask(__name__)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words and non-alphabetic words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # Stemming (you can use lemmatization instead if preferred)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def main():
    dataframe = pd.read_csv('train.csv')

    # Features (tweets)
    x = dataframe['tweet']
    # Target columns
    y = dataframe['count']

    # Text preprocessing
    dataframePrePros = [preprocess_text(document) for document in x]

    # Create the vectorizer
    vectorizer = TfidfVectorizer(stop_words=None, token_pattern=r'\b\w+\b|@\w+|\#\w+')
    # Fit and transform the text data
    x_vectorized = vectorizer.fit_transform(dataframePrePros)

    x_train, x_test, y_train, y_test = train_test_split(x_vectorized, y, test_size=0.2, random_state=49)

    clf = RandomForestClassifier(n_estimators=20)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    results_df = pd.DataFrame({
        'Tweet': x_test,
        'Count': y_test,
        'Count pred': y_pred 
    })
    #print(results_df)

    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Accuracy of hate speech: {accuracy}")

    return(y_pred[0])



# Select one tree from the forest (e.g., the first tree)
#tree_to_visualize = clf.estimators_[0]

# Plot the selected tree
#plt.figure(figsize=(16, 12))
#plot_tree(tree_to_visualize, feature_names=vectorizer.get_feature_names_out(), filled=True, rounded=True, class_names=True, max_depth=3)
#plt.show()


@app.route('/', methods=['GET'])
def predict():
    results = main()
    #results = results_df.to_dict(orient='records')

    return render_template('index.html', results=results)  


if __name__ == '__main__':
    app.run(debug=True)