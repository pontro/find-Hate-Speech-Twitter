import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

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
# Load the data
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

x_train, x_test, y_train, y_test = train_test_split(x_vectorized, y, test_size=0.2, random_state=40)

# Train an SVM classifier 
classifier = SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(x_test)

#unvectorize x
x_test_unvectorized = vectorizer.inverse_transform(x_test)

# Create a DataFrame 
results_df = pd.DataFrame({
    'Tweet': [' '.join(tokens) for tokens in x_test_unvectorized],
    'Count': y_test,
    'Count pred': y_pred 
})

# Print or display the results DataFrame
print(results_df)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of hate speech: {accuracy}")




