
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import pickle


# cleaning data
def clean_data(data):
    data.drop("Unnamed: 32", axis=1, inplace=True)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    data.drop(["id"], axis=1, inplace=True)

#training data
def create_model(data):
    X=data.drop("diagnosis",axis=1)
    y=data["diagnosis"]
    scaler=StandardScaler()
    x=scaler.fit_transform(X)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    
    #keras preprocessing ,splitting training,testing
    # x1=scaler.fit_transform(x_train)
    # x2=scaler.transform(x_test)
    # model=keras.Sequential([
    #     keras.layers.Dense(30, activation="relu", input_shape=(30,)),
    #     keras.layers.Dense(30, activation="relu"),
    #     keras.layers.Dense(1, activation="sigmoid")
    # ])
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy","precision","recall"])
    # history=model.fit(x1, y_train, epochs=100, batch_size=32, validation_split=0.2)
    # loss,accuracy=model.evaluate(x1,x2,verbose=0)
    # print("loss: {loss:.4f}")
    # print("accuracy: {accuracy}")
    
    model=LogisticRegression()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model,scaler

      
data=pd.read_csv("data.csv")

# data cleaned
clean_data(data)

#split and train
model,scaler=create_model(data)

with open("scaler.pkl","wb") as s:
    pickle.dump(scaler,s)
with open("model.pkl","wb") as m:
    pickle.dump(model,m)




