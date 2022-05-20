#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>5.5. Bankruptcy in Taiwan 🇹🇼</strong></font>

# In[1]:


import wqet_grader
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

wqet_grader.init("Project 5 Assessment")


# In[2]:


# Import libraries here
import gzip
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline


# # Prepare Data

# ## Import

# **Task 5.5.1:** Load the contents of the `"data/taiwan-bankruptcy-data.json.gz"` and assign it to the variable <code>taiwan_data</code>. 
# 
# Note that <code>taiwan_data</code> should be a dictionary. You'll create a DataFrame in a later task.

# In[7]:


# Load data file
with gzip.open("data/taiwan-bankruptcy-data.json.gz","r") as read_file:
    taiwan_data = json.load(read_file)
    
print(type(taiwan_data))


# In[8]:



wqet_grader.grade("Project 5 Assessment", "Task 5.5.1", taiwan_data["metadata"])


# **Task 5.5.2:** Extract the key names from <code>taiwan_data</code> and assign them to the variable <code>taiwan_data_keys</code>.
# 
# <div class="alert alert-info" role="alert">
#     <b>Tip:</b> The data in this assignment might be organized differently than the data from the project, so be sure to inspect it first. 
# </div>

# In[9]:


taiwan_data_keys = taiwan_data.keys()
print(taiwan_data_keys)


# In[10]:


wqet_grader.grade("Project 5 Assessment", "Task 5.5.2", list(taiwan_data_keys))


# **Task 5.5.3:** Calculate how many companies are in `taiwan_data` and assign the result to `n_companies`. 

# In[11]:


n_companies = len(taiwan_data["observations"])
print(n_companies)


# In[12]:


wqet_grader.grade("Project 5 Assessment", "Task 5.5.3", [n_companies])


# **Task 5.5.4:** Calculate the number of features associated with each company and assign the result to `n_features`.

# In[13]:


n_features = len(taiwan_data["observations"][0])
print(n_features)


# In[14]:


wqet_grader.grade("Project 5 Assessment", "Task 5.5.4", [n_features])


# **Task 5.5.5:** Create a `wrangle` function that takes as input the path of a compressed JSON file and returns the file's contents as a DataFrame. Be sure that the index of the DataFrame contains the ID of the companies. When your function is complete, use it to load the data into the DataFrame `df`.

# In[15]:


# Create wrangle function
def wrangle(filename):
    with gzip.open(filename, "r") as f:
        data = json.load(f)
    return pd.DataFrame().from_dict(data["observations"]).set_index("id")


# In[16]:


df = wrangle("data/taiwan-bankruptcy-data.json.gz")
print("df shape:", df.shape)
df.head()


# In[17]:



wqet_grader.grade("Project 5 Assessment", "Task 5.5.5", df)


# ## Explore

# **Task 5.5.6:** Is there any missing data in the dataset? Create a Series where the index contains the name of the columns in `df` and the values are the number of <code>NaN</code>s in each column. Assign the result to <code>nans_by_col</code>. Neither the Series itself nor its index require a name. 

# In[18]:


df.info()


# In[24]:


nans_by_col =  df.isnull().sum()
nans_by_col = pd.Series(nans_by_col)
print("nans_by_col shape:", nans_by_col.shape)
nans_by_col.head()


# In[25]:



wqet_grader.grade("Project 5 Assessment", "Task 5.5.6", nans_by_col)


# **Task 5.5.7:** Is the data imbalanced? Create a bar chart that shows the normalized value counts for the column `df["bankrupt"]`. Be sure to label your x-axis `"Bankrupt"`, your y-axis `"Frequency"`, and use the title `"Class Balance"`.

# In[18]:


# Plot class balance
df["bankrupt"].value_counts(normalize=True).plot(
    kind="bar",
    xlabel="Bankrupt",
    ylabel="Frequency",
    title="Class Balance"
);
# Don't delete the code below 👇
plt.savefig("images/5-5-7.png", dpi=150)


# In[19]:


with open("images/5-5-7.png", "rb") as file:
    wqet_grader.grade("Project 5 Assessment", "Task 5.5.7", file)


# ## Split

# **Task 5.5.8:** Create your feature matrix `X` and target vector `y`. Your target is `"bankrupt"`. 

# In[20]:


target = "bankrupt"
X = df.drop(columns=target)
y = df[target]
print("X shape:", X.shape)
print("y shape:", y.shape)


# In[22]:



wqet_grader.grade("Project 5 Assessment", "Task 5.5.8a", X)


# In[23]:



wqet_grader.grade("Project 5 Assessment", "Task 5.5.8b", y)


# **Task 5.5.9:** Divide your dataset into training and test sets using a randomized split. Your test set should be 20% of your data. Be sure to set `random_state` to `42`.

# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_train.shape)
print("y_test shape:", y_train.shape)


# In[25]:


wqet_grader.grade("Project 5 Assessment", "Task 5.5.9", list(X_train.shape))


# ## Resample

# **Task 5.5.10:** Create a new feature matrix `X_train_over` and target vector `y_train_over` by performing random over-sampling on the training data. Be sure to set the `random_state` to `42`.

# In[26]:


over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print("X_train_over shape:", X_train_over.shape)
X_train_over.head()


# In[27]:


wqet_grader.grade("Project 5 Assessment", "Task 5.5.10", list(X_train_over.shape))


# # Build Model

# ## Iterate

# **Task 5.5.11:** Create a classifier <code>clf</code> that can be trained on `(X_train_over, y_train_over)`. You can use any of the predictors you've learned about in the Data Science Lab. 

# In[28]:


clf = make_pipeline(
    SimpleImputer(),
    RandomForestClassifier(random_state=42)
)
clf.fit(X_train_over, y_train_over)


# In[29]:


wqet_grader.grade(
    "Project 5 Assessment",
    "Task 5.5.11",
    [isinstance(clf, (ClassifierMixin, Pipeline))],
)


# **Task 5.5.12:** Perform cross-validation with your classifier using the over-sampled training data, and assign your results to <code>cv_scores</code>. Be sure to set the <code>cv</code> argument to 5. 

# <div class="alert alert-info" role="alert">
#     <p><b>Tip:</b> Use your CV scores to evaluate different classifiers. Choose the one that gives you the best scores.</p>
# </div>

# In[30]:


cv_scores = cross_val_score(clf, X_train_over, y_train_over, cv=5, n_jobs=-1)
print(cv_scores)


# In[31]:


wqet_grader.grade("Project 5 Assessment", "Task 5.5.12", list(cv_scores))


# **Ungraded Task:** Create a dictionary <code>params</code> with the range of hyperparameters that you want to evaluate for your classifier. If you're not sure which hyperparameters to tune, check the [scikit-learn](https://scikit-learn.org/stable/) documentation for your predictor for ideas.

# <div class="alert alert-info" role="alert">
#     <p><b>Tip:</b> If the classifier you built is a predictor only (not a pipeline with multiple steps), you don't need to include the step name in the keys of your <code>params</code> dictionary. For example, if your classifier was only a random forest (not a pipeline containing a random forest), your would access the number of estimators using <code>"n_estimators"</code>, not <code>"randomforestclassifier__n_estimators"</code>.</p>
# </div>

# In[5]:


params = {
    "randomforestclassifier__n_estimators": range(25, 100, 25),
    "randomforestclassifier__max_depth": range(10,50,10)
}
params


# **Task 5.5.13:** Create a <code>GridSearchCV</code> named `model` that includes your classifier and hyperparameter grid. Be sure to set `cv` to 5, `n_jobs` to -1, and `verbose` to 1. 

# In[6]:



model = GridSearchCV(
    clf,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)
model


# In[34]:



wqet_grader.grade("Project 5 Assessment", "Task 5.5.13", model)


# **Ungraded Task:** Fit your model to the over-sampled training data. 

# In[35]:


model.fit(X_train_over, y_train_over)


# **Task 5.5.14:** Extract the cross-validation results from your model, and load them into a DataFrame named <code>cv_results</code>. Looking at the results, which set of hyperparameters led to the best performance?

# In[36]:


cv_results = pd.DataFrame(model.cv_results_)
cv_results.head(5)


# In[37]:



wqet_grader.grade("Project 5 Assessment", "Task 5.5.14", cv_results)


# **Task 5.5.15:** Extract the best hyperparameters from your model and assign them to <code>best_params</code>. 

# In[38]:


best_params = model.best_params_
print(best_params)


# In[39]:


wqet_grader.grade(
    "Project 5 Assessment", "Task 5.5.15", [isinstance(best_params, dict)]
)


# ## Evaluate

# **Ungraded Task:** Test the quality of your model by calculating accuracy scores for the training and test data.

# In[40]:



acc_train = model.score(X_train_over, y_train_over)
acc_test = model.score(X_test, y_test)

print("Model Training Accuracy:", round(acc_train, 4))
print("Model Test Accuracy:", round(acc_test, 4))


# **Task 5.5.16:** Plot a confusion matrix that shows how your model performed on your test set.

# In[41]:


ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);
# Don't delete the code below 👇
plt.savefig("images/5-5-16.png", dpi=150)


# In[42]:


with open("images/5-5-16.png", "rb") as file:
    wqet_grader.grade("Project 5 Assessment", "Task 5.5.16", file)


# **Task 5.5.17:** Generate a classification report for your model's performance on the test data and assign it to `class_report`.

# In[43]:


from sklearn.metrics import classification_report
class_report = classification_report(y_test, model.predict(X_test))
print(class_report)


# In[44]:


wqet_grader.grade("Project 5 Assessment", "Task 5.5.17", class_report)


# # Communicate

# **Task 5.5.18:** Create a horizontal bar chart with the 10 most important features for your model. Be sure to label the x-axis `"Gini Importance"`, the y-axis `"Feature"`, and use the title `"Feature Importance"`.

# In[45]:


features = X_train_over.columns
importances = model.best_estimator_.named_steps["randomforestclassifier"].feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values()
feat_imp.tail(10).plot(kind="barh")
plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("Feature Importance");
# Don't delete the code below 👇
plt.savefig("images/5-5-17.png", dpi=150)


# In[46]:


with open("images/5-5-17.png", "rb") as file:
    wqet_grader.grade("Project 5 Assessment", "Task 5.5.18", file)


# **Task 5.5.19:** Save your best-performing model to a a file named <code>"model-5-5.pkl"</code>.

# In[47]:


# Save model
with open("model-5-5.pkl", "wb") as f:
    pickle.dump(model, f)


# In[48]:



with open("model-5-5.pkl", "rb") as f:
    wqet_grader.grade("Project 5 Assessment", "Task 5.5.19", pickle.load(f))


# **Task 5.5.20:** Open the file <code>my_predictor_assignment.py</code>. Add your `wrangle` function, and then create a `make_predictions` function that takes two arguments: `data_filepath` and <code>model_filepath</code>. Use the cell below to test your module. When you're satisfied with the result, submit it to the grader. 

# In[3]:


# Import your module
from my_predictor_assignment import make_predictions

# Generate predictions
y_test_pred = make_predictions(
    data_filepath="data/taiwan-bankruptcy-data-test-features.json.gz",
    model_filepath="model-5-5.pkl",
)

print("predictions shape:", y_test_pred.shape)
y_test_pred.head()


# In[4]:


wqet_grader.grade(
    "Project 5 Assessment",
    "Task 5.5.20",
    make_predictions(
        data_filepath="data/taiwan-bankruptcy-data-test-features.json.gz",
        model_filepath="model-5-5.pkl",
    ),
)


# ---
# Copyright © 2022 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 
