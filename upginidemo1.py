# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:13:30 2023
env:  toppackages with upgini installed
@author: cstan
"""

import pandas as pd
import numpy as np
import duckdb
import os
from utilities import query


# def query(sql, path = os.environ['DUCKDB_DEV_PATH']):
#     with duckdb.connect(path) as db:
#         return db.sql(sql).df()

tables = query("show tables")
description = query("describe windows_domain_controller_raw")
print(tables)
print(description)
description = query("describe features")
print(description)
features_df = query("select * from system_features", to_df=True)
print(features_df.head(20))

df = query("select * from windows_domain_controller_raw limit 10")
print(df)
#df = query("select * from windows_domain_controller_events limit 10")
#print(df)
"""
Feature Table
- feature logic modules found in the features director, they are created by aggregating event tables by user and date.
- feature_creation.py is the control script that runs all the feature modules.
- All feature modules are joined together in feature_creation.py by user and date to create the final 'features' table found in duckdb.
- Example from feature table - only 1 feature now!
"""

features_df = query("select * from system_features")
print(features_df)

   # query data from duckdb
scores_df = query("select * from system_scores", path = os.environ['DUCKDB_DEV_PATH'])
full_df = pd.merge(features_df, scores_df, on=['dt', 'user_id'])
full_df = full_df.drop(['Var1_ind', "Var2_ind", 'Var3_ind', 'Var1Val_ind', 'Var2Val_ind',
                       'Var3Val_ind', "Var1Mean_ind", 'Var2Mean_ind', 'Var3Mean_ind'], axis=1)
full_df = full_df.drop(['user_id', 'severity'], axis=1)
print(full_df.columns)

UPGINI_API_KEY='6AydixX2k0I8ffR7KhpdBwqfuyD2rH-svXuni3152yc'
PATH = "/home/clay/feature-creation/data"
# read full data
df_full = pd.read_csv(PATH + "/Reviews.csv")
# convert Time to datetime column
df_full['key_date'] = pd.to_datetime(df_full['Time'], unit='s')
# re-order columns
df_full = df_full[['key_date', 'ProfileName', 'Summary', 'Text', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score' ]]

#We also filter our dataset to include reviews that have more than 10 helpfulness 
#and were published after 2011–01–01.

df_full = df_full[(df_full['HelpfulnessDenominator'] > 10) & 
                  (df_full['key_date'] >= '2011-01-01')]

#We also transform Helpfulness into a binary variable with 0.50 ratio.

df_full.loc[:, 'Helpful'] = np.where(df_full.loc[:, 'HelpfulnessNumerator'] / 
                                     df_full.loc[:, 'HelpfulnessDenominator'] > 0.50, 1, 0)

"""
Finally, we create a new column — combined — that will concatenate the summary and 
text into a single column. We also take this opportunity to drop any duplicates.
"""

#df_full["combined"] = f"Title: {df_full['Summary'].str.strip()} ; Content: {df_full['Text'].str.strip()}"
df_full["combined"] = (df_full['Summary'] + df_full['Text'])
df_full.drop(['Summary', 'Text', 'HelpfulnessNumerator', 'HelpfulnessDenominator' ], axis=1, inplace=True)
df_full.drop_duplicates(subset=['combined'], inplace=True)
df_full.reset_index(drop=True, inplace=True)

"""
Feature Search and Enrichment with Upgini
We are now ready to start searching for new features.

Following from the Upgini documentation, we can start a feature search using the 
FeaturesEnricher object. Within that FeaturesEnricher, we can specify a SearchKey 
(i.e., the column that we want to search for).

We can search for the following column types:

email, IP phone date datetime country post code
Let us import these into Python.
"""
from upgini import FeaturesEnricher, SearchKey
#enricher = FeaturesEnricher(search_keys={'Time': SearchKey.DATE})
enricher = FeaturesEnricher(
  search_keys = {
    "key_date":SearchKey.DATE,
#   "key_datetime": SearchKey.DATETIME,
#   "key_phone": SearchKey.PHONE,
#    "key_email": SearchKey.EMAIL,
#    "key_hashed_email": SearchKey.HEM,
#    "key_ip_address": SearchKey.IP
  },
  keep_input = True,
  api_key = "6AydixX2k0I8ffR7KhpdBwqfuyD2rH-svXuni3152yc"
)
enricher.fit(df_full[['key_date', 'ProfileName', 'Score', 'combined']], df_full['Helpful'])

"""
After some time, Upgini presents us with a list of search results — potentially 
relevant features to augment our dataset.  It seems that Upgini calculates the SHAP value 
for every found feature to measure the overall impact of that feature on the data and model quality.
"""

"""
Here we can see that by adding the enriched features, we managed to slightly improve 
the model’s performance. Admittedly, this performance gain is negligible.

Feature Generation using GPT models
Digging deeper into the documentation, it seems that the FeaturesEnricher also 
accepts another parameter — generate_features.

generate_features allows us to search for and generated feature embeddings for 
text columns. This sounds really promising. We do have text columns — combined and ProfileName.

Upgini has two LLMs connected to a search engine — GPT-3.5 from OpenAI and GPT-J — 
from the Upgini documentation

For every returned feature, we also can see and visit its source directly.

The package also evaluates the performance of a model on the original and enriched dataset.
"""
enricher = FeaturesEnricher(
    search_keys={'Time': SearchKey.DATE}, 
    generate_features=['combined', 'ProfileName']
    )
enricher.fit(df_full[['Time','ProfileName','Score','combined']], df_full['Helpful'])
"""
Upgini found us 222 relevant features. Again, per feature we get a report on their 
SHAP value, source, and its coverage on our data.

This time, we can also note that we have some generated features (i.e., the text GPT 
embeddings features).

With the newly generated features we see a massive boost in predictive performance — 

an uplift of 0.1. And the best part is that all of it was fully automated!

We definitely want to keep these features, given the massive performance gain that 
we observed. We can do this as follows:
"""

df_full_enrich = enricher.transform(df_full)

"""
The result is a dataset composed of 11 features. From this point onwards, we can 
proceed as we normally would with any other machine learning task.
"""


# %%
