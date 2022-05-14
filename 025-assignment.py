#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>Assignment: Predicting Apartment Prices in Mexico City 🇲🇽</strong></font>

# In[70]:


import warnings

import wqet_grader

warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 2 Assessment")


# <div class="alert alert-block alert-warning">
#     <b>Note:</b> In this project there are graded tasks in both the lesson notebooks and in this assignment. Together they total 24 points. The minimum score you need to move to the next project is 22 points. Once you get 22 points, you will be enrolled automatically in the next project, and this assignment will be closed. This means that you might not be able to complete the last two tasks in this notebook. If you get an error message saying that you've already passed the course, that's good news. You can stop this assignment and move onto the project 3. 
# </div>

# In this assignment, you'll decide which libraries you need to complete the tasks. You can import them in the cell below. 👇

# In[71]:


# Import libraries here
import warnings
from glob import glob

import pandas as pd
import seaborn as sns
import wqet_grader
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

import matplotlib.pyplot as plt


# # Prepare Data

# ## Import

# **Task 2.5.1:** (8 points) Write a `wrangle` function that takes the name of a CSV file as input and returns a DataFrame. The function should do the following steps:
# 
# 1. Subset the data in the CSV file and return only apartments in Mexico City (`"Distrito Federal"`) that cost less than \$100,000.
# 2. Remove outliers by trimming the bottom and top 10\% of properties in terms of `"surface_covered_in_m2"`.
# 3. Create separate `"lat"` and `"lon"` columns.
# 4. Mexico City is divided into [16 boroughs](https://en.wikipedia.org/wiki/Boroughs_of_Mexico_City). Create a `"borough"` feature from the `"place_with_parent_names"` column.
# 5. Drop columns that are more than 50\% null values.
# 6. Drop columns containing low- or high-cardinality categorical values. 
# 7. Drop any columns that would constitute leakage for the target `"price_aprox_usd"`.
# 8. Drop any columns that would create issues of multicollinearity. 
# 
# <div class="alert alert-block alert-info">
# <b>Tip:</b> Don't try to satisfy all the criteria in the first version of your <code>wrangle</code> function. Instead, work iteratively. Start with the first criteria, test it out with one of the Mexico CSV files in the <code>data/</code> directory, and submit it to the grader for feedback. Then add the next criteria.</div>

# In[72]:


# Build your `wrangle` function
def wrangle(filepath):
    df = pd.read_csv(filepath)
    mask1 = (df['property_type'] == 'apartment')
    mask2 = (df['price_aprox_usd'] < 100000)
    mask3 = (df['place_with_parent_names'].str.contains('Distrito Federal'))
    
    
    df = df[mask1 & mask2 & mask3]
    
    low, high = df['surface_covered_in_m2'].quantile([0.1, 0.9])
    maskArea = df['surface_covered_in_m2'].between(low, high)
    df = df[maskArea]
    
    df[['lat', 'lon']] = df['lat-lon'].str.split(',', expand=True).astype(float)
    df = df.drop(columns='lat-lon')
    
    df['borough'] = df['place_with_parent_names'].str.split('|', expand=True)[1]
    df = df.drop(columns='place_with_parent_names')
    
    columns_na = [i for i in df.columns if df[i].isna().sum() > int(len(df)/2)]
    df = df.drop(columns = columns_na)
    
    list_card = ["operation", "property_type",  "currency", "properati_url"]
    df = df.drop(columns = list_card)
    
    #leakage 
    highlow_cardinality = ['price', 'price_aprox_local_currency', 'price_per_m2']
    df = df.drop(columns = highlow_cardinality)
    
    
    
    return df


# In[73]:


df.info()


# In[74]:


# Use this cell to test your wrangle function and explore the data
frame = []
for i in files:
    df = wrangle(i)
    frame.append(df)


# In[75]:


df = pd.concat(frame)
df.info()


# In[76]:



wqet_grader.grade(
    "Project 2 Assessment", "Task 2.5.1", wrangle("data/mexico-city-real-estate-1.csv")
)


# **Task 2.5.2:** Use glob to create the list `files`. It should contain the filenames of all the Mexico City real estate CSVs in the `./data` directory, except for `mexico-city-test-features.csv`.

# In[107]:


files = glob('data/mexico-city-real-estate-[0-5].csv')
files


# In[108]:


wqet_grader.grade("Project 2 Assessment", "Task 2.5.2", files)


# **Task 2.5.3:** Combine your `wrangle` function, a list comprehension, and `pd.concat` to create a DataFrame `df`. It should contain all the properties from the five CSVs in `files`. 

# In[79]:


frame = []
for i in files:
    df = wrangle(i)
    frame.append(df)


# In[80]:


df = pd.concat(frame)
print(df.info())
df.head()


# In[81]:


df.shape


# In[82]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.3", df)


# ## Explore

# **Task 2.5.4:** Create a histogram showing the distribution of apartment prices (`"price_aprox_usd"`) in `df`. Be sure to label the x-axis `"Area [sq meters]"`, the y-axis `"Count"`, and give it the title `"Distribution of Apartment Prices"`.
# 
# What does the distribution of price look like? Is the data normal, a little skewed, or very skewed?

# In[83]:


# Plot distribution of price
plt.hist(df["price_aprox_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel('Count')
plt.title("Distribution of Apartment Sizes")

# Don't delete the code below 👇
plt.savefig("images/2-5-4.png", dpi=150)


# In[84]:


with open("images/2-5-4.png", "rb") as file:
    wqet_grader.grade("Project 2 Assessment", "Task 2.5.4", file)


# **Task 2.5.5:** Create a scatter plot that shows apartment price (`"price_aprox_usd"`) as a function of apartment size (`"surface_covered_in_m2"`). Be sure to label your axes `"Price [USD]"` and `"Area [sq meters]"`, respectively. Your plot should have the title `"Mexico City: Price vs. Area"`.
# 
# Do you see a relationship between price and area in the data? How is this similar to or different from the Buenos Aires dataset?

# In[85]:


# Plot price vs area
plt.scatter(x = df['surface_covered_in_m2'], y=df['price_aprox_usd'], )
plt.ylabel('Price [USD]')
plt.xlabel('Area [sq meters]')
plt.title('Mexico City: Price vs. Area')
# Don't delete the code below 👇
plt.savefig("images/2-5-5.png", dpi=150)


# In[86]:


with open("images/2-5-5.png", "rb") as file:
    wqet_grader.grade("Project 2 Assessment", "Task 2.5.5", file)


# **Task 2.5.6:** **(UNGRADED)** Create a Mapbox scatter plot that shows the location of the apartments in your dataset and represent their price using color. 
# 
# What areas of the city seem to have higher real estate prices?

# In[87]:


# Plot Mapbox location and price



# ## Split

# **Task 2.5.7:** Create your feature matrix `X_train` and target vector `y_train`. Your target is `"price_aprox_usd"`. Your features should be all the columns that remain in the DataFrame you cleaned above.

# In[88]:


df.columns


# In[89]:


feature = 'price_aprox_usd'
X_train = df[['surface_covered_in_m2', 'lat', 'lon', 'borough']]
y_train = df[feature]


# In[90]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.7a", X_train)


# In[92]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.7b", y_train)


# # Build Model

# ## Baseline

# **Task 2.5.8:** Calculate the baseline mean absolute error for your model.

# In[93]:


y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)


# In[94]:


wqet_grader.grade("Project 2 Assessment", "Task 2.5.8", [baseline_mae])


# ## Iterate

# **Task 2.5.9:** Create a pipeline named `model` that contains all the transformers necessary for this dataset and one of the predictors you've used during this project. Then fit your model to the training data.

# In[95]:


model = make_pipeline(
    OneHotEncoder(use_cat_names = True), 
    SimpleImputer(),
    Ridge()
)
model.fit(X_train, y_train)


# In[96]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.9", model)


# ## Evaluate

# **Task 2.5.10:** Read the CSV file `mexico-city-test-features.csv` into the DataFrame `X_test`.

# <div class="alert alert-block alert-info">
# <b>Tip:</b> Make sure the <code>X_train</code> you used to train your model has the same column order as <code>X_test</code>. Otherwise, it may hurt your model's performance.
# </div>

# In[97]:


X_test = pd.read_csv('./data/mexico-city-test-features.csv')
print(X_test.info())
X_test.head()


# In[98]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.10", X_test)


# **Task 2.5.11:** Use your model to generate a Series of predictions for `X_test`. When you submit your predictions to the grader, it will calculate the mean absolute error for your model.

# In[99]:


y_test_pred = model.predict(X_test)


# In[100]:


wqet_grader.grade("Project 2 Assessment", "Task 2.5.11", y_test_pred)


# # Communicate Results

# **Task 2.5.12:** Create a Series named `feat_imp`. The index should contain the names of all the features your model considers when making predictions; the values should be the coefficient values associated with each feature. The Series should be sorted ascending by absolute value.  

# In[101]:


coefficients = model.named_steps['ridge'].coef_


# In[102]:


model.named_steps['ridge'].get_params


# In[103]:


coefficients = model.named_steps['ridge'].coef_
features = df.columns
feat_imp = pd.Series(coefficients[0:5], index=features)
feat_imp


# In[104]:



wqet_grader.grade("Project 2 Assessment", "Task 2.5.12", feat_imp)


# **Task 2.5.13:** Create a horizontal bar chart that shows the **10 most influential** coefficients for your model. Be sure to label your x- and y-axis `"Importance [USD]"` and `"Feature"`, respectively, and give your chart the title `"Feature Importances for Apartment Price"`.

# In[105]:


# Create horizontal bar chart
plt.barh(features,feat_imp)
plt.title('Feature Importances for Apartment Price')
plt.ylabel('Feature')
plt.xlabel('Importance [USD]')
plt.show()
# Don't delete the code below 👇
plt.savefig("images/2-5-13.png", dpi=150)


# In[106]:


with open("images/2-5-13.png", "rb") as file:
    wqet_grader.grade("Project 2 Assessment", "Task 2.5.13", file)


# ---
# Copyright © 2022 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 
