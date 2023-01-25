#!/usr/bin/env python
# coding: utf-8

# # Checkpoint 1

# Reminder: 
# 
# - You are being evaluated for compeletion and effort in this checkpoint. 
# - Avoid manual labor / hard coding as much as possible, everything we've taught you so far are meant to simplify and automate your process.

# We will be working with the same `states_edu.csv` that you should already be familiar with from the tutorial.
# 
# We investigated Grade 8 reading score in the tutorial. For this checkpoint, you are asked to investigate another test. Here's an overview:
# 
# * Choose a specific response variable to focus on
# >Grade 4 Math, Grade 4 Reading, Grade 8 Math
# * Pick or create features to use
# >Will all the features be useful in predicting test score? Are some more important than others? Should you standardize, bin, or scale the data?
# * Explore the data as it relates to that test
# >Create at least 2 visualizations (graphs), each with a caption describing the graph and what it tells us about the data
# * Create training and testing data
# >Do you want to train on all the data? Only data from the last 10 years? Only Michigan data?
# * Train a ML model to predict outcome 
# >Define what you want to predict, and pick a model in sklearn to use (see sklearn <a href="https://scikit-learn.org/stable/modules/linear_model.html">regressors</a>.
# * Summarize your findings
# >Write a 1 paragraph summary of what you did and make a recommendation about if and how student performance can be predicted
# 
# Include comments throughout your code! Every cleanup and preprocessing task should be documented.
# 
# Of course, if you're finding this assignment interesting (and we really hope you do!), you are welcome to do more than the requirements! For example, you may want to see if expenditure affects 4th graders more than 8th graders. Maybe you want to look into the extended version of this dataset and see how factors like sex and race are involved. You can include all your work in this notebook when you turn it in -- just always make sure you explain what you did and interpret your results. Good luck!

# <h2> Data Cleanup </h2>
# 
# Import `numpy`, `pandas`, and `matplotlib`.
# 
# (Feel free to import other libraries!)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load in the "states_edu.csv" dataset and take a look at the head of the data

# In[2]:


df = pd.read_csv('../data/states_edu.csv')


# You should always familiarize yourself with what each column in the dataframe represents. Read about the states_edu dataset here: https://www.kaggle.com/noriuk/us-education-datasets-unification-project

# Use this space to rename columns, deal with missing data, etc. _(optional)_

# In[12]:


df.rename({
    'GRADES_PK_G':'ENROLL_PREK',
    'GRADES_KG_G':'ENROLL_KINDER',
    'GRADES_4_G':'ENROLL_4',
    'GRADES_8_G':'ENROLL_8',
    'GRADES_12_G':'ENROLL_12',
    'GRADES_1_8_G':'ENROLL_PRIMARY',
    'GRADES_9_12_G':'ENROLL_HS',
    'GRADES_ALL_G':'ENROLL_ALL',
    'ENROLL':'ENROLL_ALL_EST'
    },
    axis=1,inplace=True)
df.dropna(subset=['AVG_READING_8_SCORE'], inplace=True)
df.head()


# <h2>Exploratory Data Analysis (EDA) </h2>

# Chosen Outcome Variable for Test: Grade 4 Reading

# How many years of data are logged in our dataset? 

# In[11]:


print(df["YEAR"].nunique())


# Let's compare Michigan to Ohio. Which state has the higher average outcome score across all years?

# In[27]:


df.groupby('STATE')['AVG_READING_4_SCORE'].mean()

OHIO


# Find the average for your outcome score across all states in 2019

# In[31]:


df_G = df.query('YEAR == 2019')
df_G = df_G.filter(['STATE','AVG_READING_4_SCORE'])
df_G.groupby('STATE').mean()


# Find the maximum outcome score for every state. 
# 
# Refer to the `Grouping and Aggregating` section in Tutorial 0 if you are stuck.

# In[33]:


df.groupby(['STATE'])['AVG_READING_4_SCORE'].max()


# <h2> Feature Engineering </h2>
# 
# After exploring the data, you can choose to modify features that you would use to predict the performance of the students on your chosen response variable. 
# 
# You can also create your own features. For example, perhaps you figured that maybe a state's expenditure per student may affect their overall academic performance so you create a expenditure_per_student feature.
# 
# Use this space to modify or create features.

# In[34]:


df['INSTRUCTION_EXPENDITURE_PER_STUDENT'] = df['INSTRUCTION_EXPENDITURE'] / df['ENROLL_ALL']


# Feature engineering justification: I want to know how instruction expenditure per student affect the reading grade. 

# <h2>Visualization</h2>
# 
# Investigate the relationship between your chosen response variable and at least two predictors using visualizations. Write down your observations.
# 
# **Visualization 1**

# In[35]:


df.plot.scatter(x='INSTRUCTION_EXPENDITURE_PER_STUDENT', y='AVG_READING_8_SCORE', alpha=0.6)


# INSTRUCTION_EXPENDITURE_PER_STUDENT does not really correlate to AVG_READING_8_SCORE

# **Visualization 2**

# In[36]:


df.plot.scatter(x='FEDERAL_REVENUE', y='AVG_READING_8_SCORE', alpha=0.6)


# FEDERAL_REVENUE does not have correlation with AVG_READING_8_SCORE

# <h2> Data Creation </h2>
# 
# _Use this space to create train/test data_

# In[44]:


from sklearn.model_selection import train_test_split


# In[40]:


X = df[['FEDERAL_REVENUE','INSTRUCTION_EXPENDITURE_PER_STUDENT']].dropna()
y = df.loc[X.index]['AVG_READING_8_SCORE']
y.fillna(y.median(), inplace=True)


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=20, random_state=42)


# <h2> Prediction </h2>

# ML Models [Resource](https://medium.com/@vijaya.beeravalli/comparison-of-machine-learning-classification-models-for-credit-card-default-data-c3cf805c9a5a)

# In[46]:


from sklearn.linear_model import LinearRegression


# In[47]:


model = LinearRegression()


# In[48]:


model.fit(X_train, y_train)


# In[49]:


y_pred = model.predict(X_test)


# ## Evaluation

# Choose some metrics to evaluate the performance of your model, some of them are mentioned in the tutorial.

# In[54]:


print(model.intercept_)
print(model.coef_)


# We have copied over the graphs that visualize the model's performance on the training and testing set. 
# 
# Change `col_name` and modify the call to `plt.ylabel()` to isolate how a single predictor affects the model.

# In[51]:


col_name = 'FEDERAL_REVENUE'

f = plt.figure(figsize=(12,6))
plt.scatter(X_train[col_name], y_train, color = "red")
plt.scatter(X_train[col_name], model.predict(X_train), color = "green")

plt.legend(['True Training','Predicted Training'])
plt.xlabel(col_name)
plt.ylabel('NAME OF THE PREDICTOR')
plt.title("Model Behavior On Training Set")


# In[53]:


col_name = 'INSTRUCTION_EXPENDITURE_PER_STUDENT'

f = plt.figure(figsize=(12,6))
plt.scatter(X_test[col_name], y_test, color = "blue")
plt.scatter(X_test[col_name], model.predict(X_test), color = "black")

plt.legend(['True testing','Predicted testing'])
plt.xlabel(col_name)
plt.ylabel('NAME OF THE PREDICTOR')
plt.title("Model Behavior on Testing Set")


# <h2> Summary </h2>

# This model is to find out how the instruction expenditure per student and. federal revenue affects the performance of the average 8th grade reading score. With the test size of 20, the predicted model is actually similar to the actual data as the graphs above shows. ALso, as the regression line shows, there is a correlation between the two independent factors in one dependent variable. 
