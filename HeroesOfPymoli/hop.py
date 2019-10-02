# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(),
                       r'Homework\\homework_04\pandas_challange\Instructions\HeroesOfPymoli'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython
#%%
# Dependencies and Setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'notebook')
# File to Load (Remember to Change These)
file_to_load = "Resources/purchase_data.csv"
# Read Purchasing File and store into Pandas data frame
md_df = pd.read_csv(file_to_load)
md_df.head()

#%% [Total Players]
total_players = len(md_df['SN'].unique())
print(f'Total Players: {total_players}')
#%%
# * Do Item ID's and Item Names have a 1:1 map?
print(len(md_df['Item ID'].unique()) == len(md_df['Item Name'].unique()))
## We find that some unique items have more than one Item ID
## These are not mere duplicates the prices have changed with the new ID's
md_df['Item ID'] = md_df['Item ID'].apply(str)
df = md_df.groupby('Item Name').agg(lambda x: x.unique().tolist()).reset_index()
df[df['Item ID'].str.len() > 1]
## This may be a usefull form for analysis later
#%% [markdown]
## Purchase summary 
purch_cols = ['Item Name', 'Price']
def purchase_summary(dataframe, cols):
	ui_count = len(dataframe[cols[0]].unique())
	avg_price = dataframe[cols[1]].mean()
	sales_count = len(dataframe[cols[1]])
	total_sales = dataframe[cols[1]].sum()
	summary = {'Unique Items': ui_count,
			   'Average Price': avg_price,
			   'Total Purchases': sales_count,
			   'Total Sales': total_sales}
	summary_df = pd.DataFrame(summary, index=[0])
	summary_df['Average Price'] =  summary_df['Average Price'].map('${:,.2f}'.format)
	summary_df['Total Sales'] =  summary_df['Total Sales'].map('${:,.2f}'.format)
	return summary_df
purchase_summary(md_df, purch_cols)

#%% [markdown]
## Gender Demographics
# This method is agnostic with regards to number unique genders reported
demographics = md_df.groupby(['SN', 'Gender']).agg(lambda x: x.unique().tolist()).reset_index()
gen_cols = 'Gender'
def demo_summary(dataframe, cols):
	
	demo_df = dataframe[cols]
	gen_tally = Counter(demo_df)
	demo_summary = pd.DataFrame(gen_tally, index=[0])
	demo_summary = demo_summary
	demo_percentage = demo_summary.div(demo_summary.sum(axis=1), axis=0).applymap('{:,.2%}'.format)
	demo_summary = pd.concat([demo_summary, demo_percentage], ignore_index=True)
	return demo_summary

summary_df = demo_summary(demographics, gen_cols)
summary_df
#%% [markdown]
# ## Purchasing Analysis (Gender)
#%% [markdown]
# * Run basic calculations to obtain purchase count, avg. purchase price, avg. purchase total per person etc. by gender
pag_cols = ['Gender', 'Price']
calc_cols = ['Avg Price', 'Purchase Count', 'Total Sales']
# pag_df = md_df[pag_cols].groupby('Gender').mean().applymap('${:,.2f}'.format)
# pag_df.rename(columns={'Price': 'Avg Price'}, inplace = True)
# pag_df['Purchase Count'] = md_df[pag_cols].groupby('Gender').count()
# pag_df['Total Sales'] = md_df[pag_cols].groupby('Gender').sum().applymap('${:,.2f}'.format)
# pag_df
# Here's where i started having a difficult time trying to implement this the way i wanted.
# I had a realization that at least for right now jupyter notebooks are for rapid prototyping 
# and swift data exploration. IE check out the seaborn stuff I read up on in a book later on in the notebook.
def gp_summary(dataframe, sort_cols, stat_cols):
	#summary_df = demo_summary(dataframe, 'Gender')
	gp_df = dataframe[sort_cols].groupby(sort_cols[0]).mean().applymap('${:,.2f}'.format)
	gp_df.rename(columns = {'Price':'Avg Price'}, inplace = True)
	gp_df['Purchase Count'] = dataframe[sort_cols].groupby('Gender').count()
	gp_df['Total Sales'] = dataframe[sort_cols].groupby('Gender').sum()
	#For some reason the line below doesn't calculate what it's supposed too or doesn't store what it's supposed to. I'm not sure which.
	#gp_df['Per Person Per Gender'] = (gp_df['Total Sales']/summary_df.iloc[0])
	return gp_df
# It turns out i was feeding it the wrong value
gap_summary = gp_summary(md_df, pag_cols, calc_cols)
gap_summary['Per Person Per Gender'] = gap_summary['Total Sales'].astype('float')/summary_df.iloc[0].astype('float')
gap_summary['Per Person Per Gender'] = gap_summary['Per Person Per Gender'].map('${:,.2f}'.format)
gap_summary
# After spending alot of time modularizing this I realized that at this level
# Notebooks are purpose built for prototyping and not meant to be generic.
#From here on out I followed the prompts more precisely
# ## Age Demographics
#%% [markdown]
# * Establish bins for ages
bins = [0, 9, 14, 19, 24, 29, 34, 39, 50]
labels = [f'{i+1} - {i+5}' for i in bins] 
labels[0] = '<10'
labels[-1] = '40+'
del labels[-2]
print(len(bins))
print(len(labels))
print(labels)



#%%
# * Categorize the existing players using the age bins. Hint: use pd.cut()
md_df['Age Group'] = pd.cut(md_df['Age'], bins=bins, labels=labels,include_lowest=True)
md_df.tail()
age_df = md_df[['SN', 'Age Group']].groupby('SN').first()
#age_sum = pd.DataFrame(age_df, index=[1])
age_tally = age_df['Age Group'].value_counts()
age_per = age_tally.div(age_tally.sum(), axis=0).map('{:,.2%}'.format)

age_summary = pd.DataFrame({'Total Count': age_tally,
							'Percentage': age_per})
age_summary 

#%% [markdown]
# ## Purchasing Analysis (Age)
#%% [markdown]
# * Bin the purchase_data data frame by age
pba_df = md_df[['Age Group', 'Price']].groupby('Age Group').count()
pba_df['Avg Purchase'] = md_df[['Age Group', 'Price']].groupby('Age Group').mean()
pba_df['Total Sales'] = md_df[['Age Group', 'Price']].groupby('Age Group').sum()
pba_df['Sales by Group Per Person'] = (pba_df['Total Sales'].astype('float')/age_summary['Total Count'].astype('float'))
pba_df[['Avg Purchase', 'Total Sales', 'Sales by Group Per Person']]= pba_df[['Avg Purchase', 'Total Sales', 'Sales by Group Per Person']].applymap('${:,.2f}'.format)
pba_df.rename(columns={'Price': 'Total Purchases'}, inplace=True)
pba_df

#%% [markdow
# ## Top Spenders
#%% [markdown]
# * Run basic calculations to obtain the results in the table below
top_shop_df = md_df.groupby(['SN', 'Gender', 'Age']).agg(lambda x: x.unique().tolist()).reset_index()
top_shop_df = top_shop_df[['SN', 'Price']]
top_shop_df.set_index('SN')
top_shop_df['Purchases'] = top_shop_df['Price'].apply(len)
top_shop_df['Price'] = top_shop_df['Price'].apply(np.sum)
top_shop_df['Avg Purchase Price'] = top_shop_df['Price']/top_shop_df['Purchases']
top_shop_df.nlargest(columns='Avg Purchase Price', n=10)
top_shop_df.rename(columns={'Price': 'Total Purchases'}, inplace=True)
top_ten_play = top_shop_df.nlargest(columns='Total Purchases', n = 10)
top_ten_play[['Total Purchases', 'Avg Purchase Price']] = top_shop_df[['Total Purchases', 'Avg Purchase Price']].applymap('${:,.2f}'.format)
top_ten_play
#%% [markdown]
# ## Most Popular Items
#%% [markdown]
# * Retrieve the Item ID, Item Name, and Item Price columns
top_item_df = md_df[['Item ID', 'Item Name', 'Price', 'Purchase ID']].groupby(['Item Name', 'Item ID']).agg(lambda x: x.unique().tolist()).reset_index().set_index('Item Name')
top_item_df.nlargest(columns='Price',n=10)
top_item_df['Purchase Count'] = top_item_df['Purchase ID'].apply(len)
top_item_df['Total Item Sales'] = 
# * Group by Item ID and Item Name. Perform calculations to obtain purchase count, item price, and total purchase value
#%%
top_item_df
#
# * Create a summary data frame to hold the results
#
#
# * Sort the purchase count column in descending order
#
#
# * Optional: give the displayed data cleaner formatting
#
#
# * Display a preview of the summary data frame
#
#

#%%


#%% [markdown]
# ## Most Profitable Items
#%% [markdown]
# * Sort the above table by total purchase value in descending order
#
#
# * Optional: give the displayed data cleaner formatting
#
#
# * Display a preview of the data frame
#
#

#%%
