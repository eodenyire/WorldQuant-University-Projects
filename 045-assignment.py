#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>4.5. Earthquake Damage in Kavrepalanchok 🇳🇵</strong></font>

# In this assignment, you'll build a classification model to predict building damage for the district of [Kavrepalanchok](https://en.wikipedia.org/wiki/Kavrepalanchok_District).

# In[6]:


import warnings

import wqet_grader

warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 4 Assessment")


# In[7]:


# Import libraries here
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from category_encoders import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline


# # Prepare Data

# ## Connect

# Run the cell below to connect to the `nepal.sqlite` database.

# In[8]:


get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite:////home/jovyan/nepal.sqlite')


# **Task 4.5.1:** What districts are represented in the `id_map` table? Determine the unique values in the **`district_id`** column.

# In[9]:



get_ipython().run_cell_magic('sql', '', 'SELECT distinct(district_id)\nFROM id_map')


# In[10]:


result = _.DataFrame().squeeze()  # noqa F821

wqet_grader.grade("Project 4 Assessment", "Task 4.5.1", result)


# What's the district ID for Kavrepalanchok? From the lessons, you already know that Gorkha is `4`; from the textbook, you know that Ramechhap is `2`. Of the remaining districts, Kavrepalanchok is the one with the largest number of observations in the `id_map` table.

# **Task 4.5.2:** Calculate the number of observations in the `id_map` table associated with district `1`.

# In[11]:


get_ipython().run_cell_magic('sql', '', 'SELECT count(*)\nFROM id_map\nWHERE district_id = 1')


# In[12]:


result = [_.DataFrame().astype(float).squeeze()]  # noqa F821
wqet_grader.grade("Project 4 Assessment", "Task 4.5.2", result)


# **Task 4.5.3:** Calculate the number of observations in the `id_map` table associated with district `3`.

# In[13]:


get_ipython().run_cell_magic('sql', '', 'SELECT count(*)\nFROM id_map\nWHERE district_id = 3')


# In[14]:


result = [_.DataFrame().astype(float).squeeze()]  # noqa F821
wqet_grader.grade("Project 4 Assessment", "Task 4.5.3", result)


# **Task 4.5.4:** Join the unique building IDs from Kavrepalanchok in `id_map`, all the columns from  `building_structure`, and the **`damage_grade`** column from `building_damage`, limiting. Make sure you rename the **`building_id`** column in `id_map` as **`b_id`** and limit your results to the first five rows of the new table.

# In[16]:


get_ipython().run_cell_magic('sql', '', 'SELECT distinct(i.building_id) AS b_id, s.*, d.damage_grade\nFROM id_map AS i\nJOIN building_structure AS s ON i.building_id = s.building_id\nJOIN building_damage AS d ON i.building_id = d.building_id\nWHERE district_id=3\nLIMIT 5')


# In[17]:


result = _.DataFrame().set_index("b_id")  # noqa F821

wqet_grader.grade("Project 4 Assessment", "Task 4.5.4", result)


# ## Import

# **Task 4.5.5:** Write a `wrangle` function that will use the query you created in the previous task to create a DataFrame. In addition your function should:
# 
# 1. Create a `"severe_damage"` column, where all buildings with a damage grade greater than `3` should be encoded as `1`. All other buildings should be encoded at `0`. 
# 2. Drop any columns that could cause issues with leakage or multicollinearity in your model.

# In[18]:


def wrangle(db_path):
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Construct query
    query = """
        SELECT distinct(i.building_id) AS b_id, s.*, d.damage_grade
        FROM id_map AS i
        JOIN building_structure AS s ON i.building_id = s.building_id
        JOIN building_damage AS d ON i.building_id = d.building_id
        WHERE district_id=3
        """

    # Read query results into DataFrame
    df = pd.read_sql(query, conn, index_col="b_id")

    # Identify leaky columns
    drop_cols=[col for col in df.columns if "post_eq" in col]

    # Create binany target
    df["damage_grade"] = df["damage_grade"].str[-1].astype(int)
    df["severe_damage"] = (df["damage_grade"] > 3).astype(int)

    drop_cols.extend(("damage_grade", "count_floors_pre_eq", "building_id"))
    # Drop columns
    df.drop(columns=drop_cols, inplace=True)

    return df


# Use your `wrangle` function to query the database at `"/home/jovyan/nepal.sqlite"` and return  your cleaned results.

# In[19]:


df = wrangle("/home/jovyan/nepal.sqlite")
df.head()


# In[20]:



wqet_grader.grade(
    "Project 4 Assessment", "Task 4.5.5", wrangle("/home/jovyan/nepal.sqlite")
)


# ## Explore

# **Task 4.5.6:** Are the classes in this dataset balanced? Create a bar chart with the normalized value counts from the `"severe_damage"` column. Be sure to label the x-axis `"Severe Damage"` and the y-axis `"Relative Frequency"`. Use the title `"Kavrepalanchok, Class Balance"`.

# In[21]:


# Plot value counts of `"severe_damage"`
# Plot value counts of `"severe_damage"`
df["severe_damage"].value_counts(normalize=True).plot(
    kind="bar", xlabel="Severe Damage", ylabel="Relative Frequency", title="Kavrepalanchok, Class Balance"
);

# Don't delete the code below 👇
plt.savefig("images/4-5-6.png", dpi=150)


# In[22]:


with open("images/4-5-6.png", "rb") as file:
    wqet_grader.grade("Project 4 Assessment", "Task 4.5.6", file)


# **Task 4.5.7:** Is there a relationship between the footprint size of a building and the damage it sustained in the earthquake? Use seaborn to create a boxplot that shows the distributions of the `"plinth_area_sq_ft"` column for both groups in the `"severe_damage"` column. Label your x-axis `"Severe Damage"` and y-axis `"Plinth Area [sq. ft.]"`. Use the title `"Kavrepalanchok, Plinth Area vs Building Damage"`. 

# In[23]:


sns.boxplot(x="severe_damage", y="plinth_area_sq_ft", data=df)
plt.xlabel("Severe Damage")
plt.ylabel("Plinth Area [sq. ft.]")
plt.title("Kavrepalanchok, Plinth Area vs Building Damage");
# Don't delete the code below 👇
plt.savefig("images/4-5-7.png", dpi=150)


# In[24]:


with open("images/4-5-7.png", "rb") as file:
    wqet_grader.grade("Project 4 Assessment", "Task 4.5.7", file)


# **Task 4.5.8:** Are buildings with certain roof types more likely to suffer severe damage? Create a pivot table of `df` where the index is `"roof_type"` and the values come from the `"severe_damage"` column, aggregated by the mean.

# In[25]:


roof_pivot = pd.pivot_table(
    df, index="roof_type", values="severe_damage", aggfunc=np.mean
)
roof_pivot


# In[26]:



wqet_grader.grade("Project 4 Assessment", "Task 4.5.8", roof_pivot)


# ## Split

# **Task 4.5.9:** Create your feature matrix `X` and target vector `y`. Your target is `"severe_damage"`. 

# In[27]:



target = "severe_damage"
X = df.drop(columns=target)
y = df[target]
print("X shape:", X.shape)
print("y shape:", y.shape)


# In[28]:



wqet_grader.grade("Project 4 Assessment", "Task 4.5.9a", X)


# In[30]:



wqet_grader.grade("Project 4 Assessment", "Task 4.5.9b", y)


# **Task 4.5.10:** Divide your dataset into training and validation sets using a randomized split. Your validation set should be 20% of your data.

# In[31]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# In[32]:


wqet_grader.grade("Project 4 Assessment", "Task 4.5.10", [X_train.shape == (61226, 11)])


# # Build Model

# ## Baseline

# **Task 4.5.11:** Calculate the baseline accuracy score for your model.

# In[33]:


acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 2))


# In[34]:


wqet_grader.grade("Project 4 Assessment", "Task 4.5.11", [acc_baseline])


# ## Iterate

# **Task 4.5.12:** Create a model `model_lr` that uses logistic regression to predict building damage. Be sure to include an appropriate encoder for categorical features. 

# In[37]:


model_lr = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    LogisticRegression(max_iter=1000)
)
model_lr.fit(X_train, y_train)


# In[38]:



wqet_grader.grade("Project 4 Assessment", "Task 4.5.12", model_lr)


# **Task 4.5.13:** Calculate training and validation accuracy score for `model_lr`. 

# In[40]:


lr_train_acc = model_lr.score(X_train, y_train)
lr_val_acc = model_lr.score(X_val, y_val)

print("Logistic Regression, Training Accuracy Score:", lr_train_acc)
print("Logistic Regression, Validation Accuracy Score:", lr_val_acc)


# In[41]:


submission = [lr_train_acc, lr_val_acc]
wqet_grader.grade("Project 4 Assessment", "Task 4.5.13", submission)


# **Task 4.5.14:** Perhaps a decision tree model will perform better than logistic regression, but what's the best hyperparameter value for `max_depth`? Create a `for` loop to train and evaluate the model `model_dt` at all depths from 1 to 15. Be sure to use an appropriate encoder for your model, and to record its training and validation accuracy scores at every depth. The grader will evaluate your validation accuracy scores only.

# In[42]:


depth_hyperparams = range(1, 16)
training_acc = []
validation_acc = []
for d in depth_hyperparams:
    model_dt = make_pipeline(
        OrdinalEncoder(),
        DecisionTreeClassifier(max_depth=d, random_state=42)
    )
    model_dt.fit(X_train, y_train)
    training_acc.append(model_dt.score(X_train, y_train))
    validation_acc.append(model_dt.score(X_val, y_val))


# In[43]:


submission = pd.Series(validation_acc, index=depth_hyperparams)

wqet_grader.grade("Project 4 Assessment", "Task 4.5.14", submission)


# **Task 4.5.15:** Using the values in `training_acc` and `validation_acc`, plot the validation curve for `model_dt`. Label your x-axis `"Max Depth"` and your y-axis `"Accuracy Score"`. Use the title `"Validation Curve, Decision Tree Model"`, and include a legend. 

# In[44]:


plt.plot(depth_hyperparams, training_acc, label="training")
plt.plot(depth_hyperparams, validation_acc, label="validation")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy Score")
plt.title("Validation Curve, Decision Tree Model")
plt.legend();
# Don't delete the code below 👇
plt.savefig("images/4-5-15.png", dpi=150)


# In[45]:


with open("images/4-5-15.png", "rb") as file:
    wqet_grader.grade("Project 4 Assessment", "Task 4.5.15", file)


# **Task 4.5.16:** Build and train a new decision tree model `final_model_dt`, using the value for `max_depth` that yielded the best validation accuracy score in your plot above. 

# In[46]:


final_model_dt = make_pipeline(OrdinalEncoder(), DecisionTreeClassifier(max_depth=10, random_state=42))
final_model_dt.fit(X_train, y_train)


# In[47]:



wqet_grader.grade("Project 4 Assessment", "Task 4.5.16", final_model_dt)


# ## Evaluate

# **Task 4.5.17:** How does your model perform on the test set? First, read the CSV file `"data/kavrepalanchok-test-features.csv"` into the DataFrame `X_test`. Next, use `final_model_dt` to generate a list of test predictions `y_test_pred`. Finally, submit your test predictions to the grader to see how your model performs.
# 
# **Tip:** Make sure the order of the columns in `X_test` is the same as in your `X_train`. Otherwise, it could hurt your model's performance.

# In[48]:


X_test = pd.read_csv("data/kavrepalanchok-test-features.csv", index_col="b_id")
y_test_pred = final_model_dt.predict(X_test)
y_test_pred[:5]


# In[50]:


submission = pd.Series(y_test_pred)
wqet_grader.grade("Project 4 Assessment", "Task 4.5.17", submission)


# # Communicate Results

# **Task 4.5.18:** What are the most important features for `final_model_dt`? Create a Series Gini `feat_imp`, where the index labels are the feature names for your dataset and the values are the feature importances for your model. Be sure that the Series is sorted from smallest to largest feature importance. 

# In[ ]:



feat_imp = ...
feat_imp.head()


# In[ ]:



wqet_grader.grade("Project 4 Assessment", "Task 4.5.18", feat_imp)


# **Task 4.5.19:** Create a horizontal bar chart of `feat_imp`. Label your x-axis `"Gini Importance"` and your y-axis `"Label"`. Use the title `"Kavrepalanchok Decision Tree, Feature Importance"`.
# 
# Do you see any relationship between this plot and the exploratory data analysis you did regarding roof type?

# In[ ]:


# Create horizontal bar chart of feature importances

# Don't delete the code below 👇
plt.tight_layout()
plt.savefig("images/4-5-19.png", dpi=150)


# In[ ]:


with open("images/4-5-19.png", "rb") as file:
    wqet_grader.grade("Project 4 Assessment", "Task 4.5.19", file)


# Congratulations! You made it to the end of Project 4. 👏👏👏

# ---
# Copyright © 2022 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 
