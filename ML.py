import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import getpass

#take raw data
def input_data(fname,limit):
    ini_data = pd.read_csv(fname, error_bad_lines=False)
    data = ini_data.head(limit)
    return data

#print few data
def print_data(data,limit):
    data.head(limit)

#clean the null values
def clean_null(data):
    data = data.dropna()
    return data

#training input
def X(data_column):
    X = np.array(data_column)
    return X

#the validator (Label)
def Y(data_column):
    Y = np.array(data_column)
    return Y

#tokenizer
def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character



data = input_data("data.csv",10000)
print_data(data,10)
clean_null(data)

#assign meaning to number
data["strength"] = data["strength"].map({0: "Weak", 
                                         1: "Medium",
                                         2: "Strong"})

x = X(data["password"])
y = Y(data["strength"])

#train 
tdif = TfidfVectorizer(tokenizer=word)
x = tdif.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.5, 
                                                random_state=42)
model = RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

#test_interface
while True:
    user = getpass.getpass("Enter Password: ")
    if user == "q":
        break
    data = tdif.transform([user]).toarray()
    output = model.predict(data)
    print(output)



















































































































"""
ini_data = pd.read_csv("data.csv", error_bad_lines=False)
data = ini_data.head(50000)
print(data.head())
data = data.dropna()

data["strength"] = data["strength"].map({0: "Weak", 
                                         1: "Medium",
                                         2: "Strong"})

print(data.sample(5))

def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character
  

x = np.array(data["password"])
y = np.array(data["strength"])

tdif = TfidfVectorizer(tokenizer=word)
x = tdif.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.05, 
                                                random_state=42)
model = RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


user = getpass.getpass("Enter Password: ")
data = tdif.transform([user]).toarray()
output = model.predict(data)
print(output)
"""