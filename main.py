import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer  # for working on text data and conversion of text data to number data for machine learning modal

from sklearn.model_selection import train_test_split # for test and train modal
from sklearn.naive_bayes import MultinomialNB # for text count and classification 

data = pd.read_csv("language.csv")

info = data.isnull().sum()

#  22 language with 1000 sentence
language =data['language'].value_counts() 


#  converting data type of data
x = np.array(data['Text'])
y = np.array(data['language'])
cv = CountVectorizer()
X = cv.fit_transform(x)

X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size = 0.33 , random_state = 42)

model = MultinomialNB()

model.fit(X_train,y_train)

score =  model.score(X_test,y_test)

user = input("Enter your text : ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)