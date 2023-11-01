#########################
# Business Problem
#########################

# Armut, Turkey's largest online service platform, connects service providers with users who need services.
# It enables users to easily access services like cleaning, renovation, and moving through their computers or smartphones.
# A product recommendation system is desired using Association Rule Learning based on the dataset containing service users,
# the services they received, and the categories of these services.


#########################
# Dataset
#########################
# The dataset consists of services purchased by customers and the categories of these services.
# It includes the date and time information for each service.

# UserId: Customer number
# ServiceId: Anonymized services for each category. (Example: upholstery cleaning service under the Cleaning category)
# A ServiceId can be found under different categories and represents different services under different categories.
# (Example: ServiceId 4 under CategoryId 7 represents radiator cleaning service,
# while ServiceId 4 under CategoryId 2 represents furniture assembly service.)
# CategoryId: Anonymized categories. (Example: Cleaning, moving, renovation categories)
# CreateDate: The date the service was purchased


#########################
# TASK 1: Data Preparation
#########################

# Step 1: Read the armut_data.csv file.
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
from datetime import datetime

armut_dt = pd.read_csv("armut_data.csv")

# Step 2: ServiceId represents a different service for each CategoryID.
# Create a new variable by concatenating ServiceID and CategoryID with "_", representing the services.
armut_dt['Service'] = armut_dt['ServiceId'].astype(str) + "_" + armut_dt['CategoryId'].astype(str)

# Step 3: The dataset consists of dates and times when services were received,
# and there is no definition of a basket (invoice, etc.).
# To apply Association Rule Learning, a basket definition (such as an invoice) needs to be created.
# Here, the basket definition is the services each customer received monthly.
# For example, customer with ID 7256 received services 9_4, 46_4 in August 2017 as one basket;
# and services 9_4, 38_4 in October 2017 as another basket. Baskets need to be identified with a unique ID.
# Create a new date variable containing only the year and month.
# Concatenate UserID and the newly created date variable with "_" to create a new variable called ID.
armut_dt['YearMonthDate'] = armut_dt['CreateDate'].str[:7]
armut_dt['BasketID'] = armut_dt['UserId'].astype(str) + '_' + armut_dt['YearMonthDate'].astype(str)

#########################
# TASK 2: Generate Association Rules
#########################

# Step 1: Create a pivot table of service baskets like the one below.

# Service         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# BasketID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

#armut_dt['Count'] = armut_dt.groupby(['BasketID', 'Service'])['Service'].count()
#print(armut_dt.groupby(['BasketID', 'Service'])['Service'].count().unstack().head())
pivot_table = armut_dt.groupby(['BasketID', 'Service'])['Service'].count().unstack().fillna(0).map(lambda x: 1 if x > 0 else 0)
print(pivot_table)

#print(pivot_table)

# Step 2: Generate association rules.

frequent_itemsets = apriori(pivot_table.astype(bool), min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets,metric='support',min_threshold=0.01)


# Step 3: Use the arl_recommender function to recommend a service to a user who last received service 2_0.

sorted_rules = rules.sort_values(by='lift', ascending=False)
print(sorted_rules.T)
