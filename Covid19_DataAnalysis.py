#!/usr/bin/env python
# coding: utf-8

# # COVID19 Predictions using XGBOOST
# 
# - Analyze data (February 2020 - April 2020) using plotly to predict the Confirmed Cases and Fatalities for the month of May using XGBoost.
# 

# # <a id='main'>Table of Contents</a>
# - [Summary Statistics](#stat)
# - [Exploratory Data Analysis(EDA)](#eda)
#     1. [Universal growth of COVID19 over time](#world)
#     2. [Trend of COVID19 in top 10 affected countries](#top10)
#     3. [Mortality Rate](#dr)
#     4. [Country Specific growth of COVID19](#country)
#         - [United States of America](#us)
#         - [India](#in)
#         - [Italy](#it)
#         - [Spain](#sp)
# - [Feature Engineering](#fe)
# - [Preprocessing](#pp)
# - [Training and Prediction](#tp)
# - [Forecast Visualizations](#for)

# # Summary Statistics

# In[111]:


#Using country data derived from wikipedia, this package provides conversion functions between ISO country names, country-codes, and continent names.
#pip install pycountry_convert


# In[72]:


#Required Libraries
import pandas as pd
import numpy as np
import datetime as dt
import requests
import sys
from itertools import chain
import pycountry
import pycountry_convert as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV


# In[4]:


#Loading train and test Datasets downloaded from Kaggle
df_train=pd.read_csv('./Data/train.csv')
df_test = pd.read_csv('./Data/test.csv')


# In[8]:


display(df_train.head())
display(df_train.describe())
display(df_train.info())


# In[14]:


#Renaming some columns
df_train=df_train.rename(columns={"Id": "ID", "ConfirmedCases": "Confirmed_Cases"})


# In[15]:


#Understanding Data types
display(df_train.dtypes)


# In[11]:


#Converting Date from string to DateTime
df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d')
df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d')


# In[12]:


train_date_min = df_train['Date'].min()
train_date_max = df_train['Date'].max()
print('Minimum date from training set: {}'.format(train_date_min))
print('Maximum date from training set: {}'.format(train_date_max))


# In[13]:


test_date_min = df_test['Date'].min()
test_date_max = df_test['Date'].max()
print('Minimum date from test set: {}'.format(test_date_min))
print('Maximum date from test set: {}'.format(test_date_max))


# # <a id='eda'>Exploratory Data Analysis</a>

# # <a id='world'>Universal growth of COVID19 over time</a>
# - COVID19 growth throughout the world from January 22, 2020.
# - Tree maps: Worldwide COVID19
# - Chloropleth maps: Daily impact of COVID19 

# In[110]:


class country_utils():
    def __init__(self):
        self.d = {}
    
    def get_dic(self):
        return self.d
    
    def get_country_details(self,country):
        """Returns country code(alpha_3) and continent"""
        try:
            country_obj = pycountry.countries.get(name=country)
            if country_obj is None:
                c = pycountry.countries.search_fuzzy(country)
                country_obj = c[0]
            continent_code = pc.country_alpha2_to_continent_code(country_obj.alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj.alpha_3, continent
        except:
            if 'Congo' in country:
                country = 'Congo'
            elif country == 'Diamond Princess' or country == 'Laos' or country == 'MS Zaandam'            or country == 'Holy See' or country == 'Timor-Leste':
                return country, country
            elif country == 'Korea, South' or country == 'South Korea':
                country = 'Korea, Republic of'
            elif country == 'Taiwan*':
                country = 'Taiwan'
            elif country == 'Burma':
                country = 'Myanmar'
            elif country == 'West Bank and Gaza':
                country = 'Gaza'
            else:
                return country, country
            country_obj = pycountry.countries.search_fuzzy(country)
            continent_code = pc.country_alpha2_to_continent_code(country_obj[0].alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj[0].alpha_3, continent
    
    def get_iso3(self, country):
        return self.d[country]['code']
    
    def get_continent(self,country):
        return self.d[country]['continent']
    
    def add_values(self,country):
        self.d[country] = {}
        self.d[country]['code'],self.d[country]['continent'] = self.get_country_details(country)
    
    def fetch_iso3(self,country):
        if country in self.d.keys():
            return self.get_iso3(country)
        else:
            self.add_values(country)
            return self.get_iso3(country)
        
    def fetch_continent(self,country):
        if country in self.d.keys():
            return self.get_continent(country)
        else:
            self.add_values(country)
            return self.get_continent(country)


# In[40]:


df_train.Confirmed_Cases = np.abs(df_train.Confirmed_Cases)
df_tm = df_train.copy()
date = df_tm.Date.max()#get current date
df_tm = df_tm[df_tm['Date']==date]
obj = country_utils()
df_tm.Province_State.fillna('',inplace=True)
df_tm['continent'] = df_tm.apply(lambda x: obj.fetch_continent(x['Country_Region']), axis=1)
df_tm["world"] = "World" # in order to have a single root node
fig = px.treemap(df_tm, path=['world', 'continent', 'Country_Region'], values='Confirmed_Cases',
                  color='Confirmed_Cases', hover_data=['Country_Region'],
                  color_continuous_scale='dense', title='Current share of Worldwide COVID19 Cases')
fig.show()


# In[41]:


fig = px.treemap(df_tm, path=['world', 'continent', 'Country_Region'], values='Fatalities',
                  color='Fatalities', hover_data=['Country_Region'],
                  color_continuous_scale='matter', title='Current share of Worldwide COVID19 Deaths')
fig.show()


# In[26]:


#To observe a trend create a column for daily cases and deaths which will be the difference between current value and previous day's value 
def add_daily_measures(df):
    df.loc[0,'Daily Cases'] = df.loc[0,'Confirmed_Cases']
    df.loc[0,'Daily Deaths'] = df.loc[0,'Fatalities']
    for i in range(1,len(df)):
        df.loc[i,'Daily Cases'] = df.loc[i,'Confirmed_Cases'] - df.loc[i-1,'Confirmed_Cases']
        df.loc[i,'Daily Deaths'] = df.loc[i,'Fatalities'] - df.loc[i-1,'Fatalities']
    #Make the first row as 0 because we don't know the previous value
    df.loc[0,'Daily Cases'] = 0
    df.loc[0,'Daily Deaths'] = 0
    return df


# In[27]:


df_world = df_train.copy()
df_world = df_world.groupby('Date',as_index=False)['Confirmed_Cases','Fatalities'].sum()
df_world = add_daily_measures(df_world)
df_world['Cases:7-day rolling average'] = df_world['Daily Cases'].rolling(7).mean()
df_world['Deaths:7-day rolling average'] = df_world['Daily Deaths'].rolling(7).mean()


# In[35]:


fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_world['Date'], y=df_world['Daily Cases']),
    go.Bar(name='Deaths', x=df_world['Date'], y=df_world['Daily Deaths'])
])

fig.add_trace(go.Scatter(name='Cases:7-day rolling average',x=df_world['Date'],y=df_world['Cases:7-day rolling average'],marker_color='black'))
fig.add_trace(go.Scatter(name='Deaths:7-day rolling average',x=df_world['Date'],y=df_world['Deaths:7-day rolling average'],marker_color='darkred'))

# Change the bar mode
fig.update_layout(barmode='overlay', title='Worldwide daily Case and Death count',showlegend=True)
fig.show()


# In[36]:


df_map = df_train.copy()
df_map['Date'] = df_map['Date'].astype(str)
df_map = df_map.groupby(['Date','Country_Region'], as_index=False)['Confirmed_Cases','Fatalities'].sum()


# In[42]:


df_map['iso_alpha'] = df_map.apply(lambda x: obj.fetch_iso3(x['Country_Region']), axis=1)


# In[43]:


df_map['log(Confirmed_Cases)'] = np.log(df_map.Confirmed_Cases + 1)
df_map['log(Fatalities)'] = np.log(df_map.Fatalities + 1)


# >Choropleth map on logarithmic scale as cases have grown exponentially

# In[45]:


px.choropleth(df_map, 
              locations="iso_alpha", 
              color="log(Confirmed_Cases)", 
              hover_name="Country_Region", 
              hover_data=["Confirmed_Cases"] ,
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.dense, 
              title='Total Confirmed Cases growth(Logarithmic Scale)')


# # Analysis of Confirmed Cases plot:
# - Initially, dense color was present only in China because of the number of cases in Wuhan. 
# - As we proceed towards May the color intensity increases in most of the regions of the world heavily affecting the North America region with more than 1 Million confirmed cases.
# - Confirmed cases start reducing March onwards in China favoring and proving the positive effects of lockdown.
# 

# In[46]:


px.choropleth(df_map, 
              locations="iso_alpha", 
              color="log(Fatalities)", 
              hover_name="Country_Region",
              hover_data=["Fatalities"],
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.OrRd,
              title = 'Total Deaths growth(Logarithmic Scale)')


# # Analysis of Deaths Plot:
# - China was the first country to experience highest number of deaths.
# - Number of deaths started increasing in Iran, Italy, Spain and North America.
# - Strict measures taken in China resulted in the reduction of deaths from March to May.

# # Top 10:

# In[47]:


#Get the top 10 countries
last_date = df_train.Date.max()
df_countries = df_train[df_train['Date']==last_date]
df_countries = df_countries.groupby('Country_Region', as_index=False)['Confirmed_Cases','Fatalities'].sum()
df_countries = df_countries.nlargest(10,'Confirmed_Cases')
#Get the trend for top 10 countries
df_trend = df_train.groupby(['Date','Country_Region'], as_index=False)['Confirmed_Cases','Fatalities'].sum()
df_trend = df_trend.merge(df_countries, on='Country_Region')
df_trend.drop(['Confirmed_Cases_y','Fatalities_y'],axis=1, inplace=True)
df_trend.rename(columns={'Country_Region':'Country', 'Confirmed_Cases_x':'Cases', 'Fatalities_x':'Deaths'}, inplace=True)
#Add columns for studying logarithmic trends
df_trend['log(Cases)'] = np.log(df_trend['Cases']+1)# Added 1 to remove error due to log(0).
df_trend['log(Deaths)'] = np.log(df_trend['Deaths']+1)


# In[48]:


px.line(df_trend, x='Date', y='Cases', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries')


# # Growth Trend Analysis:
# - Above plot shows the top 10 countries by growth in the number of confirmed cases.
# - China is not present in this plot which implies that not a large number of new cases were confirmed through February.
# - The number of confirmed cases in US ascended.

# In[49]:


px.line(df_trend, x='Date', y='Deaths', color='Country', title='COVID19 Total Deaths growth for top 10 worst affected countries')


# # Death trend analysis:
# - The cases and deaths are increasing(almost exponentially) for top 10 countries.
# - US has shown the greatest rise in the number of Confirmed Cases and number of deaths.
# - UK, Italy have been worst affected by the number of deaths after US.
# - 6 out of the top 10 affected countries are Western European countries.

# # Daily variation of Confirmed Case,Deaths for Top 10 affected countries(Logarithmic scale)

# In[50]:


px.line(df_trend, x='Date', y='log(Cases)', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries(Logarithmic Scale)')


# In[51]:


px.line(df_trend, x='Date', y='log(Deaths)', color='Country', title='COVID19 Total Deaths growth for top 10 worst affected countries(Logarithmic Scale)')


# # Mortality Rate 
# - Number of fatalities divided by the number of confirmed cases

# In[52]:


df_map['Mortality Rate%'] = round((df_map.Fatalities/df_map.Confirmed_Cases)*100,2)


# In[56]:


px.choropleth(df_map, 
                    locations="iso_alpha", 
                    color="Mortality Rate%", 
                    hover_name="Country_Region",
                    hover_data=["Confirmed_Cases","Fatalities"],
                    animation_frame="Date",
                    color_continuous_scale=px.colors.sequential.Magma_r,
                    title = 'Worldwide Daily Variation of Mortality Rate%')


# In[54]:


df_trend['Mortality Rate%'] = round((df_trend.Deaths/df_trend.Cases)*100,2)
px.line(df_trend, x='Date', y='Mortality Rate%', color='Country', title='Variation of Mortality Rate% \n(Top 10 worst affected countries)')


# # Country Specific growth of COVID19:

# # United States of America
# 

# In[57]:


# Dictionary to get the state codes from state names for US
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}


# In[58]:


df_us = df_train[df_train['Country_Region']=='US']
df_us['Date'] = df_us['Date'].astype(str)
df_us['state_code'] = df_us.apply(lambda x: us_state_abbrev.get(x.Province_State,float('nan')), axis=1)
df_us['log(Confirmed_Cases)'] = np.log(df_us.Confirmed_Cases + 1)
df_us['log(Fatalities)'] = np.log(df_us.Fatalities + 1)


# In[59]:


px.choropleth(df_us,
              locationmode="USA-states",
              scope="usa",
              locations="state_code",
              color="log(Confirmed_Cases)",
              hover_name="Province_State",
              hover_data=["Confirmed_Cases"],
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.Darkmint,
              title = 'Total Cases growth for USA(Logarithmic Scale)')


# In[60]:


px.choropleth(df_us,
              locationmode="USA-states",
              scope="usa",
              locations="state_code",
              color="log(Fatalities)",
              hover_name="Province_State",
              hover_data=["Fatalities"],
              animation_frame="Date",
              color_continuous_scale=px.colors.sequential.OrRd,
              title = 'Total deaths growth for USA(Logarithmic Scale)')


# In[61]:


df_usa = df_train.query("Country_Region=='US'")
df_usa = df_usa.groupby('Date',as_index=False)['Confirmed_Cases','Fatalities'].sum()
df_usa = add_daily_measures(df_usa)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_usa['Date'], y=df_usa['Daily Cases']),
    go.Bar(name='Deaths', x=df_usa['Date'], y=df_usa['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(USA)')
fig.show()


# # <a id='in'>India</a>
# COVID19 outbreak has started a bit late in India as compared to other countries. But, it has started to pick up pace. With limited testing and not a well funded healthcare system, India is surely up for a challenge. Let's hope that the 21 day lockdown helps to stop or atleast slower down the spread of this dreaded virus.

# In[62]:


df_train.Province_State.fillna('NaN', inplace=True)
df_plot = df_train.groupby(['Date','Country_Region','Province_State'], as_index=False)['Confirmed_Cases','Fatalities'].sum()


# In[63]:


df = df_plot.query("Country_Region=='India'")
df.reset_index(inplace = True)
df = add_daily_measures(df)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df['Date'], y=df['Daily Cases']),
    go.Bar(name='Deaths', x=df['Date'], y=df['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(India)',
                 annotations=[dict(x='2020-03-23', y=106, xref="x", yref="y", text="First Lockdown(23rd March)", showarrow=True, arrowhead=1, ax=-100, ay=-100),
                              dict(x='2020-04-15', y=835, xref="x", yref="y", text="Second Lockdown(15th April)", showarrow=True, arrowhead=1, ax=-100, ay=-100),
                              dict(x='2020-05-04', y=3932, xref="x", yref="y", text="Third Lockdown(4th May)", showarrow=True, arrowhead=1, ax=-200, ay=0)])
fig.show()


# In[65]:


df_ind_cases = pd.read_csv('./Data/covid_19_india.csv')

df_ind_cases.dropna(how='all',inplace=True)
df_ind_cases['DateTime'] = pd.to_datetime(df_ind_cases['Date'], format = '%d/%m/%y')


# In[66]:


r = requests.get(url='https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson')
geojson = r.json()


# In[67]:


def change_state_name(state):
    if state == 'Odisha':
        return 'Orissa'
    elif state == 'Telengana':
        return 'Telangana'
    return state


# In[68]:


df_ind_cases['State/UnionTerritory'] = df_ind_cases.apply(lambda x: change_state_name(x['State/UnionTerritory']), axis=1)
last_date = df_ind_cases.DateTime.max()
df_ind_states = df_ind_cases.copy()
df_ind_cases = df_ind_cases[df_ind_cases['DateTime']==last_date]


# **Here's a state wise breakdown of cases, deaths and recoveries.**

# In[75]:


#import matplotlib
columns = ['State/UnionTerritory', 'Cured', 'Deaths','Confirmed']
df_ind_cases = df_ind_cases[columns]
df_ind_cases.sort_values('Confirmed',inplace=True, ascending=False)
df_ind_cases.reset_index(drop=True,inplace=True)
#df_ind_cases.style.background_gradient(cmap='viridis')


# In[76]:


fig = px.choropleth(df_ind_cases, geojson=geojson, color="Confirmed",
                    locations="State/UnionTerritory", featureidkey="properties.NAME_1",
                    hover_data=['Cured','Deaths'],
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title='India: Total Current cases per state'
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.update_layout(height=600,margin={"r":0,"t":30,"l":0,"b":30})
fig.show()


# > Hover on the map to view Recoveries and Deaths. 

# In[77]:


px.line(df_ind_states, x='DateTime', y='Confirmed', color='State/UnionTerritory', title='India: State-wise cases')


# # Italy

# In[78]:


df_ita = df_plot.query("Country_Region=='Italy'")
df_ita.reset_index(inplace = True)
df_ita = add_daily_measures(df_ita)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_ita['Date'], y=df_ita['Daily Cases']),
    go.Bar(name='Deaths', x=df_ita['Date'], y=df_ita['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(Italy)',
                 annotations=[dict(x='2020-03-09', y=1797, xref="x", yref="y", text="Lockdown Imposed(9th March)", showarrow=True, arrowhead=1, ax=-100, ay=-200)])
fig.show()


# # Spain

# In[89]:


df_esp = df_plot.query("Country_Region=='Spain'")
df_esp.reset_index(inplace = True)
df_esp = add_daily_measures(df_esp)
fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_esp['Date'], y=df_esp['Daily Cases']),
    go.Bar(name='Deaths', x=df_esp['Date'], y=df_esp['Daily Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='overlay', title='Daily Case and Death count(Spain)',
                 annotations=[dict(x='2020-03-15', y=1407, xref="x", yref="y", text="Lockdown Imposed(15th March)", showarrow=True, arrowhead=1, ax=-100, ay=-200)])
fig.show()


# # Feature Engineering

# In[79]:


df_pd = pd.read_csv('./Data/Population density by countries.csv') 
df_pd['iso_code3'] = df_pd.apply(lambda x: obj.fetch_iso3(x['Country (or dependent territory)'].strip()), axis=1)
df = df_train[df_train['Date']==train_date_max]
#df = df_train.copy()
df = df.groupby(['Date','Country_Region'], as_index=False)['Confirmed_Cases','Fatalities'].sum()
df['iso_code3'] = df.apply(lambda x:obj.fetch_iso3(x['Country_Region']), axis=1)
df = df.merge(df_pd, how='left', on='iso_code3')


# In[80]:


def convert(pop):
    if pop == float('nan'):
        return 0.0
    return float(pop.replace(',',''))

df['Population'].fillna('0', inplace=True)
df['Population'] = df.apply(lambda x: convert(x['Population']),axis=1)
df['Density pop./km2'].fillna('0', inplace=True)
df['Density pop./km2'] = df.apply(lambda x: convert(x['Density pop./km2']),axis=1)


# In[83]:


q3 = np.percentile(df.Confirmed_Cases,75)
q1 = np.percentile(df.Confirmed_Cases,25)
IQR = q3-q1
low = q1 - 1.5*IQR
high = q3 + 1.3*IQR
df = df[(df['Confirmed_Cases']>low) & (df['Confirmed_Cases']<high)]
df['continent'] = df.apply(lambda x: obj.fetch_continent(x['Country_Region']), axis=1)


# In[84]:


df['Date_x'] = df['Date_x'].astype(str)


# In[86]:


px.scatter(df,x='Confirmed_Cases',y='Density pop./km2', size = 'Population', color='continent',hover_data=['Country_Region'], title='Variation of Population density wrt Confirmed Cases',range_y=[0,1500])


# In[87]:


px.scatter(df,x='Fatalities',y='Density pop./km2', size = 'Population', color='continent',hover_data=['Country_Region'],title='Variation of Population density wrt Fatalities',range_y=[0,1500])


# In[88]:


df.corr()


# # Pre-processing

# In[90]:


#Add continent column to training set
df_train['Continent'] = df_train.apply(lambda X: obj.fetch_continent(X['Country_Region']), axis=1)


# In[91]:


def categoricalToInteger(df):
    #convert NaN Province State values to a string
    df.Province_State.fillna('NaN', inplace=True)
    #Define Ordinal Encoder Model
    oe = OrdinalEncoder()
    df[['Province_State','Country_Region','Continent']] = oe.fit_transform(df.loc[:,['Province_State','Country_Region','Continent']])
    return df


# Extract useful features from date.

# In[92]:


def create_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['weekofyear'] = df['Date'].dt.weekofyear
    return df


# Split the training data into train and dev set for cross-validation.

# In[93]:


def train_dev_split(df, days):
    #Last days data as dev set
    date = df['Date'].max() - dt.timedelta(days=days)
    return df[df['Date'] <= date], df[df['Date'] > date]


# In order to avoid data leakage, there should be no overlap between the data in the training and test set. Therefore, I'll remove the data from training set having dates that are already present in the test set.

# In[94]:


def avoid_data_leakage(df, date=test_date_min):
    return df[df['Date']<date]


# In[95]:


df_train = avoid_data_leakage(df_train)
df_train = categoricalToInteger(df_train)
df_train = create_features(df_train)


# In[96]:


df_train, df_dev = train_dev_split(df_train,0)


# Select all the columns that are needed for training the model.

# In[98]:


columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','Continent','Confirmed_Cases','Fatalities']
df_train = df_train[columns]
df_dev = df_dev[columns]


# # Training and Prediction
# 
# - XGBOOST model
# - Two models since Confirmed Cases, Fatalities both need to be predicted
# 
# 

# In[99]:


#Apply the same transformation to test set that were applied to the training set
df_test['Continent'] = df_test.apply(lambda X: obj.fetch_continent(X['Country_Region']), axis=1)
df_test = categoricalToInteger(df_test)
df_test = create_features(df_test)
#Columns to select
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','Continent']


# In[101]:


results = []
#Loop through all the unique countries
for country in df_train.Country_Region.unique():
    #Filter on the basis of country
    df_train1 = df_train[df_train["Country_Region"]==country]
    #Loop through all the States of the selected country
    for state in df_train1.Province_State.unique():
        #Filter on the basis of state
        df_train2 = df_train1[df_train1["Province_State"]==state]
        #Convert to numpy array for training
        train = df_train2.values
        #Separate the features and labels
        X_train, y_train = train[:,:-2], train[:,-2:]
        #model1 for predicting Confirmed Cases
        model1 = XGBRegressor(n_estimators=1000)
        model1.fit(X_train, y_train[:,0])
        #model2 for predicting Fatalities
        model2 = XGBRegressor(n_estimators=1000)
        model2.fit(X_train, y_train[:,1])
        #Get the test data for that particular country and state
        df_test1 = df_test[(df_test["Country_Region"]==country) & (df_test["Province_State"] == state)]
        #Store the ForecastId separately
        ForecastId = df_test1.ForecastId.values
        #Remove the unwanted columns
        df_test2 = df_test1[columns]
        #Get the predictions
        y_pred1 = model1.predict(df_test2.values)
        y_pred2 = model2.predict(df_test2.values)
        #Append the predicted values to submission list
        for i in range(len(y_pred1)):
            d = {'ForecastId':ForecastId[i], 'Confirmed_Cases':y_pred1[i], 'Fatalities':y_pred2[i]}
            results.append(d)


# In[102]:


df_results = pd.DataFrame(results)


# In[103]:


df_results.to_csv(r'results.csv', index=False)


# # Forecast Vizualizations
# 

# In[105]:


df_forecast = pd.concat([df_test,df_results.iloc[:,1:]], axis=1)
df_world_f = df_forecast.copy()
df_world_f = df_world_f.groupby('Date',as_index=False)['Confirmed_Cases','Fatalities'].sum()
df_world_f = add_daily_measures(df_world_f)


# In[106]:


df_world = avoid_data_leakage(df_world)


# In[107]:


fig = go.Figure(data=[
    go.Bar(name='Total Cases', x=df_world['Date'], y=df_world['Confirmed_Cases']),
    go.Bar(name='Total Cases Forecasted', x=df_world_f['Date'], y=df_world_f['Confirmed_Cases'])
])
# Change the bar mode
fig.update_layout(barmode='group', title='Worldwide Confirmed Cases + Forcasted Cases')
fig.show()


# In[108]:


fig = go.Figure(data=[
    go.Bar(name='Total Deaths', x=df_world['Date'], y=df_world['Fatalities']),
    go.Bar(name='Total Deaths Forecasted', x=df_world_f['Date'], y=df_world_f['Fatalities'])
])
# Change the bar mode
fig.update_layout(barmode='group', title='Worldwide Deaths + Forcasted Deaths')
fig.show()


# In[ ]:




