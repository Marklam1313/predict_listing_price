import re
import random
import numpy as np
import pandas as pd
import pickle
from os.path import expanduser
from collections import Counter
import spacy
from collections import defaultdict
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def get_new_feature(fea):
    return [fea[0], "0 hab.", fea[1], fea[-1]]


def clean_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.pos_ in pos_list]
    return tokens

def all_features(d, key_map, loc_map):
    fea = []
    fea.append(tokens_to_features(d["tokens"], key_map))
    fea.append([loc_to_feature(d["loc_string"], loc_map)])
    fea.append(features_to_features(d["features"], d["price"]))
    return np.concatenate(fea)

def tokens_to_features(tokens, key_map):
    feature = np.zeros(len(key_map))
    for k in tokens:
        if k in key_map:
            feature[key_map[k]] = 1
    return feature

def loc_to_feature(loc, loc_map):
    loc = loc.split("\n")[0]
    return loc_map.get(loc, -1)

def features_to_features(features, price):
    m2 = int(features[0].replace(" m2", ""))
    bedrooms = int(features[1].replace(" hab.", ""))
    bathrooms = int(features[2].replace(" baños", "").replace(" baño", ""))
    price = float(price.replace(" €", ""))
    return np.array([m2, bedrooms, bathrooms, price])

# Load the data
filename = expanduser("~/data/train.pickle")
with open(filename,'rb') as f:
    data = pickle.load(f, encoding='utf-8')

# Data preprocessing

# Removing ads from multiple units
data_v1 = [] 
for i, d in enumerate(data):
    features = d["features"]
    if bool(re.match("^desde", features[0])) == False:
        if bool(re.search("\d+ m2", features[0])) == True:
            data_v1.append(d)


data_v2 = []
for i, d in enumerate(data_v1):
    features = d["features"]
    if not bool(re.match("\d hab", features[1])):
        if len(features) == 3:
            features = get_new_feature(features)
            d["features"] = features
    if len(features) == 4:
        data_v2.append(d)

# Tokenizing descriptions and cleaning text
nlp = spacy.load("es_core_news_sm")
pos_list = ["VERB", "NOUN", "ADJ", "ADV"]

for d in data_v2:
    main_tokens = clean_text(d["desc"])
    d["tokens"] = main_tokens

# Splitting into training and testing sets
random.seed(4)
random.shuffle(data_v2)
N = int(len(data_v2) * .95)
train = data_v2[:N]
test = data_v2[N:]

# Feature engineering
doc_freq = defaultdict(list)
for i, d in enumerate(train):
    tokens = set(d["tokens"])
    for token in tokens:
        doc_freq[token].append(i)

locations = np.unique([d["loc_string"].split("\n")[0] for d in train])
loc_map = {v:k for k, v in enumerate(locations)}
prices = [float(d["price"].replace(" €", "")) for d in train]

median_prices = {}
for word in doc_freq:
    L = len(doc_freq[word])
    if L >= 20 and L < 400:
        price_word = [prices[i] for i in doc_freq[word]]
        median_prices[word] = np.median(price_word)

keywords = []
High = 350 + 35
Low = 350 - 35

for word in median_prices:
    if median_prices[word] > High or median_prices[word] < Low:
        keywords.append(word)

key_map = {k: i for i, k in enumerate(keywords)}

header = np.array(keywords + ["loc", "size", "bedrooms", "bathrooms", "price"])

# Creating features
train_features = np.stack([all_features(d, key_map, loc_map) for d in train])
test_features = np.stack([all_features(d, key_map, loc_map) for d in test])

# Dataframe creation for train and test data
df_train = pd.DataFrame(train_features, columns=header)
df_test = pd.DataFrame(test_features, columns=header)

# Define labels and features
train_labels = df_train["price"]
train_features = df_train.drop(columns=["price"])
val_labels = df_test["price"]
val_features = df_test.drop(columns=["price"])
print("Start training:::::")
# Model training and evaluation
base_models = [
    ('xgb', GridSearchCV(XGBRegressor(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]})),
    ('rf', GridSearchCV(RandomForestRegressor(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}))
]

meta_model = GridSearchCV(RandomForestRegressor(), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]})
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)

stacking_regressor.fit(train_features, train_labels)
y_pred = stacking_regressor.predict(val_features)
r2 = r2_score(val_labels, y_pred)
print("r2 Error:", r2)

# Kaggle test data processing
filename = expanduser("~/data/test_kaggle.pickle")
with open(filename,'rb') as f:
    test_data = pickle.load(f, encoding='utf-8')

test_data_v1 = [] 
for i, d in enumerate(test_data):
    features = d["features"]
    if bool(re.match("^desde", features[0])) == False:
        if bool(re.search("\d+ m2", features[0])) == True:
            test_data_v1.append(d)

test_data_v2= []
for i, d in enumerate(test_data_v1):
    features = d["features"]
    if not bool(re.match("\d hab", features[1])):
        if len(features) == 2:
            features = get_new_feature(features)
            d["features"] = features
    if len(features) == 3:
        test_data_v2.append(d)

for d in test_data_v2:
    main_tokens = clean_text(d["desc"])
    d["tokens"] = main_tokens

for dictionary in test_data_v2:
    dictionary.pop('description', None)
    dictionary['price'] = '-1'

test_kaggle = np.stack([all_features(d, key_map, loc_map) for d in test_data_v2])

kaggke_test = pd.DataFrame(test_kaggle, columns=header)
kaggke_test.drop(columns='price', inplace=True)

# Kaggle predictions
kaggle_preds = stacking_regressor.predict(kaggke_test)

df_kaggle_preds = pd.DataFrame(kaggle_preds)
column_titles = ['price']
df_kaggle_preds.columns = column_titles
df_kaggle_preds.reset_index(inplace=True)
df_kaggle_preds.rename(columns={'index': 'id'}, inplace=True)
df_kaggle_preds.to_csv('solution.csv', index=False, header=True)
