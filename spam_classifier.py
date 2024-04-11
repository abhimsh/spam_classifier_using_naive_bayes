import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

if __name__ == "__main__":

    list_of_stopwords = set(stopwords.words("english"))
    stemmer = SnowballStemmer("english")
    lemmatizerer = WordNetLemmatizer()
    # bad_of_words = CountVectorizer(max_features=5000)
    tf_idf = TfidfVectorizer(max_features=5000)

    mail_data = pd.read_csv(os.path.join("data", "SMSSpamCollection"),
                            sep="\t", names=["label", "message"])
    corpus = []
    for i in range(len(mail_data)):
        cleaned_data = re.sub("[^a-zA-Z]", " ", mail_data["message"][i])
        cleaned_data = cleaned_data.lower()
        # req_words = [stemmer.stem(_) for _ in cleaned_data.split() if _ not in list_of_stopwords]
        req_words = [lemmatizerer.lemmatize(_) for _ in cleaned_data.split() if _ not in list_of_stopwords]
        corpus.append(" ".join(req_words))

    # X = bad_of_words.fit_transform(corpus).toarray()
    X = tf_idf.fit_transform(corpus).toarray()

    Y = pd.get_dummies(mail_data["label"])
    Y = Y.iloc[:, 1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    model = MultinomialNB().fit(X_train, Y_train)
    y_predict = model.predict(X_test)

    print("Confusion Matrix: ", confusion_matrix(Y_test, y_predict), sep="\n")
    print("Accuracy: ", accuracy_score(Y_test, y_predict), sep="\n")

