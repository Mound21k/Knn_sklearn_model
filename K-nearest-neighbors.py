import numpy as np
import sklearn 
import pandas as pd 
import joblib 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv("car.data")

myPreprocessor = preprocessing.LabelEncoder()

buying = myPreprocessor.fit_transform(list(data["buying"]))
maint = myPreprocessor.fit_transform(list(data["maint"]))
door = myPreprocessor.fit_transform(list(data["door"]))
persons = myPreprocessor.fit_transform(list(data["persons"]))
lug_boot = myPreprocessor.fit_transform(list(data["lug_boot"]))
safety = myPreprocessor.fit_transform(list(data["safety"]))
clas = myPreprocessor.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying , maint , door , persons , lug_boot , safety))
y = np.array(data[predict])
file_name = "KNN_model.sav"
x_train , x_test , y_train , y_test = train_test_split(X ,y ,test_size= 0.1 , random_state= 123)
model = joblib.load(file_name)
acc = model.score(x_test , y_test)
print(acc)