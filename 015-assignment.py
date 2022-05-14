#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>Assignment: Housing in Brazil 🇧🇷</strong></font>

# In[8]:


import wqet_grader

wqet_grader.init("Project 1 Assessment")


# In this assignment, you'll work with a dataset of homes for sale in Brazil. Your goal is to determine if there are regional differences in the real estate market. Also, you will look at southern Brazil to see if there is a relationship between home size and price, similar to what you saw with housing in some states in Mexico. 

# <div class="alert alert-block alert-warning">
#     <b>Note:</b> There are are 19 graded tasks in this assignment, but you only need to complete 18. Once you've successfully completed 18 tasks, you'll be automatically enrolled in the next project, and this assignment will be closed. This means that you might not be allowed to complete the last task. So if you get an error saying that you've already complete the course, that's good news! Move to project 2. 
# </div>

# **Before you start:** Import the libraries you'll use in this notebook: Matplotlib, pandas, and plotly. Be sure to import them under the aliases we've used in this project.

# In[9]:


# Import Matplotlib, pandas, and plotly
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates

import matplotlib.pyplot as plt


# # Prepare Data

# In this assignment, you'll work with real estate data from Brazil.  In the `data` directory for this project there are two CSV that you need to import and clean.

# ## Import

# **Task 1.5.1:** Import the CSV file `data/brasil-real-estate-1.csv` into the DataFrame `df1`.

# In[18]:


df1 = pd.read_csv('data/brasil-real-estate-1.csv')
df1.head()


# In[19]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.1", df1)


# Before you move to the next task, take a moment to inspect `df1` using the `info` and `head` methods. What issues do you see in the data? What cleaning will you need to do before you can conduct your analysis?

# In[20]:


df1.info()


# **Task 1.5.2:** Drop all rows with `NaN` values from the DataFrame `df1`.

# In[21]:


df1.dropna(inplace=True)
df1.head()


# In[22]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.2", df1)


# **Task 1.5.3:** Use the `"lat-lon"` column to create two separate columns in `df1`: `"lat"` and `"lon"`. Make sure that the data type for these new columns is `float`.

# In[23]:


df1[["lat", "lon"]] = df1["lat-lon"].str.split(",", expand=True)
df1.info()
df1.head()


# In[24]:


df1["lat"] = df1.lat.astype(float)
df1['lon'] = df1.lon.astype(float)
df1.head


# In[25]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.3", df1)


# **Task 1.5.4:** Use the `"place_with_parent_names"` column to create a `"state"` column for `df1`. (Note that the state name always appears after `"|Brasil|"` in each string.)

# In[26]:


df1.loc[:,'state'] = df1.loc[:,'place_with_parent_names'].str.split('|', expand=True)[2]

df1.head()


# In[110]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.4", df1)


# **Task 1.5.5:** Transform the `"price_usd"` column of `df1` so that all values are floating-point numbers instead of strings. 

# In[28]:


df1["price_usd"]=df1.price_usd.str.replace('$','')
df1["price_usd"]=df1.price_usd.str.replace(',','')
df1.head()


# In[29]:


df1['price_usd'] = df1.price_usd.astype(float)
df1.head()


# In[30]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.5", df1)


# In[ ]:





# **Task 1.5.6:** Drop the `"lat-lon"` and `"place_with_parent_names"` columns from `df1`.

# In[31]:


df1=df1.drop('lat-lon', axis='columns')
df1=df1.drop('place_with_parent_names', axis='columns')
df1.head()


# In[32]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.6", df1)


# **Task 1.5.7:** Import the CSV file `brasil-real-estate-2.csv` into the DataFrame `df2`.

# In[33]:


df2 = pd.read_csv('data/brasil-real-estate-2.csv')


# In[34]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.7", df2)


# Before you jump to the next task, take a look at `df2` using the `info` and `head` methods. What issues do you see in the data? How is it similar or different from `df1`?

# In[35]:


df2.info()


# In[36]:


df2.head()


# **Task 1.5.8:** Use the `"price_brl"` column to create a new column named `"price_usd"`. (Keep in mind that, when this data was collected in 2015 and 2016, a US dollar cost 3.19 Brazilian reals.)

# In[37]:


df2['price_usd'] = df2['price_brl'] / 3.19
df2.head()


# In[38]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.8", df2)


# **Task 1.5.9:** Drop the `"price_brl"` column from `df2`, as well as any rows that have `NaN` values. 

# In[39]:


df2=df2.drop('price_brl', axis='columns')
df2.head()


# In[40]:


df2.dropna(inplace=True)
df2.head()


# In[41]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.9", df2)


# **Task 1.5.10:** Concatenate `df1` and `df2` to create a new DataFrame named `df`. 

# In[71]:


df = pd.concat([df1, df2])
print("df shape:", df.shape)


# In[72]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.10", df)


# <div class="alert alert-block alert-info">
#     <p><b>Frequent Question:</b> I can't pass this question, and I don't know what I've done wrong. 😠 What's happening?</p>
#     <p><b>Tip:</b> In this assignment, you're working with data that's similar — but not identical — the data used in the lessons. That means that you might need to make adjust the code you used in the lessons to work here. Take a second look at <code>df1</code> after you complete 1.5.6, and make sure you've correctly created the state names.</p>
# </div>

# ## Explore

# It's time to start exploring your data. In this section, you'll use your new data visualization skills to learn more about the regional differences in the Brazilian real estate market.
# 
# Complete the code below to create a `scatter_mapbox` showing the location of the properties in `df`.

# In[73]:


fig = px.scatter_mapbox(
    df,
    lat= df['lat'],
    lon= df['lon'],
    center={"lat": -14.2, "lon": -51.9},  # Map will be centered on Brazil
    width=600,
    height=600,
    hover_data=["price_usd"],  # Display price when hovering mouse over house
)

fig.update_layout(mapbox_style="open-street-map")

fig.show()


# **Task 1.5.11:** Use the `describe` method to create a DataFrame `summary_stats` with the summary statistics for the `"area_m2"` and `"price_usd"` columns.

# In[74]:


summary_stats = df[['area_m2','price_usd']].describe()
summary_stats.head()


# In[75]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.11", summary_stats)


# **Task 1.5.12:** Create a histogram of `"price_usd"`. Make sure that the x-axis has the label `"Price [USD]"`, the y-axis has the label `"Frequency"`, and the plot has the title `"Distribution of Home Prices"`.

# In[78]:


plt.hist(df["price_usd"])
plt.xlabel("Price [USD]")
plt.ylabel("Frequency")
plt.title("Distribution of Home Prices")
# Don't change the code below
plt.savefig("images/1-5-12.png", dpi=150)


# In[79]:


with open("images/1-5-12.png", "rb") as file:
    wqet_grader.grade("Project 1 Assessment", "Task 1.5.12", file)


# **Task 1.5.13:** Create a horizontal boxplot of `"area_m2"`. Make sure that the x-axis has the label `"Area [sq meters]"` and the plot has the title `"Distribution of Home Sizes"`.

# In[80]:


plt.boxplot(df["area_m2"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Frequency")
plt.title("Distribution of Home Sizes")
# Don't change the code below
plt.savefig("images/1-5-13.png", dpi=150)


# In[81]:


with open("images/1-5-13.png", "rb") as file:
    wqet_grader.grade("Project 1 Assessment", "Task 1.5.13", file)


# **Task 1.5.14:** Use the `groupby` method to create a Series named `mean_price_by_region` that shows the mean home price in each region in Brazil, sorted from smallest to largest.

# In[86]:


mean_price_by_region = df.groupby("region")["price_usd"].mean().sort_values()
mean_price_by_region


# In[87]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.14", mean_price_by_region)


# **Task 1.5.15:** Use `mean_price_by_region` to create a bar chart. Make sure you label the x-axis as `"Region"` and the y-axis as `"Mean Price [USD]"`, and give the chart the title `"Mean Home Price by Region"`.

# In[91]:


mean_price_by_region.plot(kind="bar", xlabel="Region", ylabel="Mean Price [USD]", title="Mean Home Price by Region")
# Don't change the code below
plt.savefig("images/1-5-15.png", dpi=150)


# In[92]:


with open("images/1-5-15.png", "rb") as file:
    wqet_grader.grade("Project 1 Assessment", "Task 1.5.15", file)


# <div class="alert alert-block alert-info">
#     <b>Keep it up!</b> You're halfway through your data exploration. Take one last break and get ready for the final push. 🚀
# </div>
# 
# You're now going to shift your focus to the southern region of Brazil, and look at the relationship between home size and price.
# 
# **Task 1.5.16:** Create a DataFrame `df_south` that contains all the homes from `df` that are in the `"South"` region. 

# In[93]:


df_south = df[df["region"]=="South"]
df_south.head()


# In[94]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.16", df_south)


# **Task 1.5.17:** Use the `value_counts` method to create a Series `homes_by_state` that contains the number of properties in each state in `df_south`. 

# In[95]:


homes_by_state = df_south["state"].value_counts()
homes_by_state


# In[96]:



wqet_grader.grade("Project 1 Assessment", "Task 1.5.17", homes_by_state)


# **Task 1.5.18:** Create a scatter plot showing price vs. area for the state in `df_south` that has the largest number of properties. Be sure to label the x-axis `"Area [sq meters]"` and the y-axis `"Price [USD]"`; and use the title `"<name of state>: Price vs. Area"`.

# <div class="alert alert-block alert-info">
#     <p><b>Tip:</b> You should replace <code>&lt;name of state&gt;</code> with the name of the state that has the largest number of properties.</p>
# </div>

# In[99]:


df_south_large = df_south[df_south["state"]=="Rio Grande do Sul"]
df_south_large.head()

plt.scatter(x=df_south_large["area_m2"], y=df_south_large["price_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Rio Grande do Sul: Price vs. Area")
# Don't change the code below
plt.savefig("images/1-5-18.png", dpi=150)


# In[100]:


with open("images/1-5-18.png", "rb") as file:
    wqet_grader.grade("Project 1 Assessment", "Task 1.5.18", file)


# **Task 1.5.19:** Create a dictionary `south_states_corr`, where the keys are the names of the three states in the `"South"` region of Brazil, and their associated values are the correlation coefficient between `"area_m2"` and `"price_usd"` in that state.
# 
# As an example, here's a dictionary with the states and correlation coefficients for the Southeast region. Since you're looking at a different region, the states and coefficients will be different, but the structure of the dictionary will be the same.
# 
# ```python
# {'Espírito Santo': 0.6311332554173303,
#  'Minas Gerais': 0.5830029036378931,
#  'Rio de Janeiro': 0.4554077103515366,
#  'São Paulo': 0.45882050624839366}
# ```

# In[108]:


south_states_corr={}
df_south_Santa = df_south[df_south["state"]=="Santa Catarina"]
df_south_Par = df_south[df_south["state"]=="Paraná"]

south_states_corr["Rio Grande do Sul"] = df_south_large["area_m2"].corr(df_south_large["price_usd"])
south_states_corr["Santa Catarina"] = df_south_Santa["area_m2"].corr(df_south_Santa["price_usd"])
south_states_corr["Paraná"] = df_south_Par["area_m2"].corr(df_south_Par["price_usd"])
south_states_corr
print(south_states_corr)


# In[109]:


wqet_grader.grade("Project 1 Assessment", "Task 1.5.19", south_states_corr)


# ---
# Copyright © 2022 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 

# In[ ]:




