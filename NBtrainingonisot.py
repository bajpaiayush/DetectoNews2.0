import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
#Load datasets
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

#Add labels(1 = Real and 0 = Fake)
true["label"] = 1
fake["label"] = 0

#combine and shuffule
data = pd.concat([true, fake], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

print("Dataset shape:", data.shape)
print(data.head())
#Preparing feature and labels(will clarify further..)
X = data["text"]
Y = data["label"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,test_size = 0.2, random_state=42 )
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


#training :
model = MultinomialNB()
model.fit(X_train_tfidf, Y_train)
#evaluating my model
Y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and Vectorizer saved successfully!!")