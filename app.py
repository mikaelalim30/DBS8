#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv(r"C:\Users\PC\Downloads\DBS_SingDollar.csv")
print(df)


# In[4]:


X=df.loc[:,["SGD"]]
Y=df.loc[:,["DBS"]]


# In[5]:


from sklearn import linear_model


# In[6]:


model=linear_model.LinearRegression()


# In[7]:


model.fit(X,Y)


# In[8]:


pred=model.predict(X)


# In[9]:


from sklearn.metrics import mean_squared_error


# In[10]:


rmse=mean_squared_error(Y,pred)**0.5


# In[11]:


import joblib


# In[12]:


joblib.dump(model,"regression")


# In[13]:


from sklearn import tree


# In[14]:


model=tree.DecisionTreeRegressor()


# In[15]:


model.fit(X,Y)
pred=model.predict(X)
print("rmse:",mean_squared_error(Y,pred)**0.5)


# In[16]:


joblib.dump(model,"tree")


# In[17]:


from flask import Flask, request, render_template


# In[18]:


import joblib


# In[19]:


app=Flask(__name__)


# In[20]:


@app.route("/", methods = ["GET","POST"])
def index():
    if request.method == "POST":
        rates = float(request.form.get("rates"))
        print(rates)
        model1 = joblib.load("regression")
        r1 = model1.predict([[rates]])
        model2 = joblib.load("tree")
        r2 = model2.predict([[rates]])
        return (render_template("index.html", result1 = r1, result2 = r2))
    else:
        return(render_template("index.html", result1 = "WSGI", result2 = "WSGI"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




