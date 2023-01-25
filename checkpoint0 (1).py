#!/usr/bin/env python
# coding: utf-8

# # Checkpoint 0 

# These exercises are a mix of Python and Pandas practice. Most should be no more than a few lines of code! 

# In[ ]:


# here is a Python list:

a = [1, 2, 3, 4, 5, 6]


# In[2]:


a = [1, 2, 3, 4, 5, 6]
a[3:]


# In[6]:


# create a list of numbers from 1 to 20
sample_list = list(range(1,21))
sample_list


# In[9]:


# now get a list with only the even numbers between 1 and 100
# you may or may not make use of the list you made in the last cell
sample_list = list(range(2,100,2))
sample_list


# In[10]:


# write a function that takes two numbers as arguments
# and returns the first number divided by the second
def two_number(number1,number2):
    x = number1/number2
    print(x)
    


# In[12]:


# write a function that takes a string as input
# and return that string in all caps
def caps_func(value):
    x = value.upper()
    print(x)


# In[22]:


# fizzbuzz
# you will need to use both iteration and control flow 
# go through all numbers from 1 to 30 in order
# if the number is a multiple of 3, print fizz
# if the number is a multiple of 5, print buzz
# if the number is a multiple of 3 and 5, print fizzbuzz and NOTHING ELSE
# if the number is neither a multiple of 3 nor a multiple of 5, print the number
for num in range(1,31):
    if num % 3 == 0 and num % 5 == 0:
        print("fizzbuzz")
        print("NOTHING ELSE")
    elif num % 5 == 0:
        print("buzz")
    elif num % 3 == 0:
        print("fizz")
    else:
        print (num)


    


# In[24]:


# create a dictionary that reflects the following menu pricing (taken from Ahmo's)
# Gyro: $9 
# Burger: $9
# Greek Salad: $8
# Philly Steak: $10
hello_dictionary = {"Gyro":9, "Burger":9, "Greek Salad":8, "Philly Steak":10}


# In[26]:


# load in the "starbucks.csv" dataset
# refer to how we read the cereal.csv dataset in the tutorial
import pandas as pd
df = pd.read_csv("../data/starbucks.csv")


# In[36]:


# output the calories, sugars, and protein columns only of every 40th row. 
import pandas as pd
df = pd.read_csv("../data/starbucks.csv")
df.loc[::40][["calories", "sugars", "protein"]]


# In[38]:


# select all rows with more than and including 400 calories
df[df["calories"] >= 400]


# In[40]:


# select all rows whose vitamin c content is higher than the iron content
df[df["vitamin c"] > df["iron"]]


# In[42]:


# create a new column containing the caffeine per calories of each drink
df["caffeine_per_calories"] = df["caffeine"] / df["calories"]


# In[43]:


# what is the average calorie across all items?
df["calories"].mean()


# In[51]:


# how many different categories of beverages are there?
print(df["beverage_category"].nunique())


# In[54]:


# what is the average # calories for each beverage category?
temp = df.groupby("beverage_category")
temp["calories"].mean()


# In[55]:


# plot the distribution of the number of calories in drinks with a histogram
df["calories"].plot.hist(edgecolor='black', alpha=0.8, title="calories Distrbution")


# In[57]:


# plot calories against total fat with a scatterplot
df.plot.scatter(x="calories", y="total fat", title="calories vs fat Content")


# In[ ]:




