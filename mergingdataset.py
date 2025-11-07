#importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
bf_fake = pd.read_csv("BuzzFeed_fake_news_content.csv")
bf_real = pd.read_csv("BuzzFeed_real_news_content.csv")
pf_fake = pd.read_csv("PolitiFact_fake_news_content.csv")
pf_real = pd.read_csv("PolitiFact_real_news_content.csv")

true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

bf_fake["label"] = 0
bf_real["label"] = 1
pf_fake["label"] = 0
pf_real["label"] = 1

true["label"] = 1
fake["label"] = 0

misinfo_fake = pd.read_csv("DataSet_Misinfo_FAKE.csv")
misinfo_true = pd.read_csv("DataSet_Misinfo_TRUE.csv")

# Add Labels
misinfo_fake["label"] = 0
misinfo_true["label"] = 1

#Merge into one DataFrame
misinfo = pd.concat([misinfo_fake, misinfo_true], axis=0)

#keep only the text + label columns
if "text" in misinfo.columns:
  misinfo = misinfo[["text","label"]]
elif "content" in misinfo.columns:
  misinfo = misinfo.rename(columns={"content": "text"})[["text","label"]]

#Drop empty/duplicates
misinfo = misinfo.dropna().drop_duplicates()

# print("Misinfo Dataset Shape:", misinfo.shape)
# print(misinfo['label'].value_counts())
# print(misinfo.head())

#ISOT Dataset Cleaning
true["label"] = 1
fake["label"] = 0

# Combine title + text for richer content
true["text"] = true["title"].fillna("") + " " + true["text"].fillna("")
fake["text"] = fake["title"].fillna("") + " " + fake["text"].fillna("")

# Keep only text + label
isot = pd.concat([true[["text","label"]], fake[["text","label"]]], axis=0).dropna().drop_duplicates()

# print("ISOT dataset shape:", isot.shape)
# print(isot['label'].value_counts())
# print(isot.head())


# FakeNewsNet: BuzzFeed + PolitiFact (already loaded earlier as bf_fake, bf_real, pf_fake, pf_real)

# Function to normalize FakeNewsNet datasets
def prepare_fnn(df):
    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "content" in df.columns:
        df = df.rename(columns={"content": "text"})
    return df[["text","label"]].dropna()

bf_fake = prepare_fnn(bf_fake)
bf_real = prepare_fnn(bf_real)
pf_fake = prepare_fnn(pf_fake)
pf_real = prepare_fnn(pf_real)

fakenewsnet = pd.concat([bf_fake, bf_real, pf_fake, pf_real], axis=0).drop_duplicates()

# print("FakeNewsNet shape:", fakenewsnet.shape)
# print(fakenewsnet['label'].value_counts())
# print(fakenewsnet.head())

# Merge everything together
data = pd.concat([misinfo, isot, fakenewsnet], axis=0)

# Drop NAs and duplicates, shuffle
data = data.dropna().drop_duplicates().sample(frac=1).reset_index(drop=True)

print("Final merged dataset shape:", data.shape)
print(data['label'].value_counts())
print(data.head())
#Saving the Cleaned and Final Dataset
data.to_csv("final_merged_dataset.csv", index=False)
