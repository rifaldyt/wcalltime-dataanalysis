# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import zipfile
import os
import seaborn as sns

#Download Dataset dari Google Drive
!wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tP5fwPaVSXRswDBCbnwqrZ1E3dA6****' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tP5fwPaVSXRswDBCbnwqrZ1E3dA6****" \
    -O fifa-wc-all-time.zip && rm -rf /tmp/cookies.txt

local_zip = 'fifa-wc-all-time.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

for dirname, _, filenames in os.walk('/tmp/fifa-wc-all-time/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Merge All Files
all_files =[file for file in os.listdir('/tmp/fifa-wc-all-time/')]
all_files

#Read kedalam Dataframe
world_cup = pd.DataFrame()

for file in all_files:
    if file != 'FIFA - World Cup Summary.csv':
        df = pd.read_csv('/tmp/fifa-wc-all-time/' + file)
        world_cup = pd.concat([world_cup, df])
world_cup

world_cup.to_csv('wc_alltime.csv',index=False)

fifa_wc = pd.read_csv('wc_alltime.csv')
fifa_wc

"""Analisis file World Cup Summary"""

wc_summary = pd.read_csv('/tmp/fifa-wc-all-time/FIFA - World Cup Summary.csv')
wc_summary.head(10)

#Rename West Germany to Germany
wc_summary['HOST'] = wc_summary['HOST'].replace("West Germany", "Germany")
wc_summary['CHAMPION'] = wc_summary['CHAMPION'].replace("West Germany", "Germany")
wc_summary['RUNNER UP'] = wc_summary['RUNNER UP'].replace("West Germany", "Germany")
wc_summary['THIRD PLACE'] = wc_summary['THIRD PLACE'].replace("West Germany", "Germany")

wc_summary

pd.pivot_table(wc_summary, index = ['YEAR', 'HOST', 'CHAMPION', 'RUNNER UP', 'THIRD PLACE'], 
               values = 'TEAMS', aggfunc = {'TEAMS' : np.sum}).sort_values('TEAMS', ascending = False)

plt.figure(figsize=(10, 7))
plt.title("Presentase Juara Terbanyak Selama Piala Dunia")
palette = sns.color_palette("viridis")
plot = sns.histplot(data=wc_summary, x="CHAMPION", hue="CHAMPION", multiple="layer", color = palette)

plt.xlabel("Negara")

plt.figure(figsize=(10, 7))
plt.title("Presentase Runner Up Terbanyak Selama Piala Dunia")
palette = sns.color_palette("viridis")
plot = sns.histplot(data=wc_summary, x = "RUNNER UP", hue = "RUNNER UP", multiple="layer", color = palette)

plt.xticks(rotation=60)
plt.xlabel("Negara")

plt.figure(figsize=(10, 7))
sns.heatmap(wc_summary.corr(), annot=True)
plt.xticks(rotation=20)
plt.title('Heatmap Korelasi antar Variabel')
plt.show()

"""Berapa presentase Tuan Rumah yang mememenangkan Piala Dunia?"""

host_won = []
for index, row in wc_summary.iterrows():
    host_won.append(row["HOST"] == row["CHAMPION"])
wc_summary["HOST WON"] = host_won

# Generate a count plot of booleans that indicates that the host country won

plt.figure(figsize=(8,6))
plt.title("Presentase Tuan Rumah Juara")
sns.countplot(data=wc_summary, x="HOST WON")

host_win_total = len(wc_summary[wc_summary["HOST WON"] == True]) 
tournament_total = len(wc_summary) 
host_win_percentage = (host_win_total/tournament_total)*100
host_win_percentage

#analyze the factor of average goals per game

plt.figure(figsize=(10,6))
sns.boxplot(data=wc_summary[['TEAMS', 'MATCHES PLAYED', 'GOALS SCORED', 'AVG GOALS PER GAME']])

plt.title('Rata-Rata Gol per Game')
plt.xlabel('Variabel')
plt.ylabel('Count')
plt.show()

"""Negara yang mencetak Gol terbanyak"""

#analyze count of goal score based on host, champion, runner up and third place team
higest_goal = pd.pivot_table(wc_summary, index = ['HOST', 'CHAMPION', 'RUNNER UP', 'THIRD PLACE'], values = 'GOALS SCORED', 
              aggfunc = {'GOALS SCORED' : np.sum}).sort_values('GOALS SCORED', ascending = False)
higest_goal

runner_up = wc_summary['RUNNER UP'].value_counts()

sns.countplot(data=wc_summary, x = 'RUNNER UP', order = runner_up.index.values)
plt.title('Presentase Runner Up Terbanyak')
plt.xticks(rotation=90);

plt.figure(figsize=(10,6))
sns.barplot(data=wc_summary, x="YEAR", y="TEAMS")
plt.title('Jumlah Tim yang mengikuti Piala Dunia setiap Eventnya')
plt.xlabel('Tahun')
plt.ylabel('Jumlah Negara')

plt.xticks(rotation=90);

plt.figure(figsize=(10,6))
sns.set_style('whitegrid')

chart = sns.barplot(x='CHAMPION', y='GOALS SCORED', data=wc_summary.groupby('CHAMPION')['GOALS SCORED'].sum().reset_index().sort_values('GOALS SCORED', ascending=False))

plt.title('Jumlah Gol berdasarkan Juara Turnamen')
plt.xlabel('Juara')
plt.ylabel('Jumlah Gol')
plt.show()

plt.figure(figsize=(10,6))

sns.barplot(data=wc_summary, x="HOST", y="AVG GOALS PER GAME")
plt.title('Rata-Rata Gol yang dihasilkan Selama Turnamen (Berdasarkan Host)')

plt.xticks(rotation=90);

plt.figure(figsize=(10,6))
sns.barplot(data=wc_summary, x="MATCHES PLAYED", y="AVG GOALS PER GAME")
plt.title('Rata-Rata Gol yang dihasilkan Selama Turnamen (Berdasarkan Host)')

plt.xticks(rotation=90);

"""

---"""

fifa_wc.columns

fifa_wc.dtypes

fifa_wc.describe()

"""Clean the Data"""

#Check for null values 
nan_wc = world_cup[world_cup.isna().any(axis = 1)]
nan_wc

world_cup.isnull().any()

world_cup.loc[world_cup["Team"] == 'West Germany', "Team"] = 'Germany'

world_cup_nation = world_cup.groupby(['Team']).sum()
world_cup

world_cup_nation['Win Rate'] = (world_cup_nation['Win']/world_cup_nation['Games Played'])*100
world_cup_nation

"""Juara Per tahunnya

Tim mana yang memainkan permainan paling banyak dan menang paling banyak?
"""

wc_most_win = world_cup_nation.sort_values(['Games Played','Win'], ascending = False)
wc_most_win.head(10)

"""Model Regressi"""

#handling categorical data
wc_summary_regression = pd.get_dummies(wc_summary, drop_first = True)
wc_summary_regression.head(10)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import metrics
import math

X = wc_summary_regression.drop('AVG GOALS PER GAME', axis = 1)
y = wc_summary_regression['AVG GOALS PER GAME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Commented out IPython magic to ensure Python compatibility.
LinReg= LinearRegression()
# %time LinReg.fit(X_train, y_train)
LinReg.score(X_test, y_test)

y_pred = LinReg.predict(X_test)
print(y_pred)

print('Mean Absolute Error (MAE) :', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE) :', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE) :', math.sqrt(metrics.mean_squared_error(y_test, y_pred)))

x = y_test
y = y_pred

sns.set_style('whitegrid')
sns.set_palette('deep')

sns.regplot(x=x, y=y, color='g', marker='o')
sns.lineplot(x=x, y=x, color='darkblue')

plt.title('Regression Model')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

"""diambil dari data summary"""