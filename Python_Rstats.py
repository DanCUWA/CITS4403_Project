pip install pandas matplotlib seaborn statsmodels scipy


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# Load Data
overall_stats = pd.read_csv("overall_stats.csv")
infected_person_stats = pd.read_csv("infected_person_stats.csv")

# Data Preparation
overall_stats = overall_stats.dropna()
print(overall_stats.info())
print(overall_stats.describe())
print(overall_stats.head())

infected_person_stats['clustering_coef'] = infected_person_stats['Uninfected.Count.Near'] / infected_person_stats['Total.Squares.Visited']

# Prepare infected DataFrame
infected_df = pd.DataFrame({
    'timeStep': infected_person_stats['Time.Step'].astype('category'),
    'xCoord': infected_person_stats['Infected.X'].astype(int),
    'yCoord': infected_person_stats['Infected.Y'].astype(int),
    'Uninfected.Count.Near': infected_person_stats['Uninfected.Count.Near'],
    'Total.Squares.Visited': infected_person_stats['Total.Squares.Visited'],
    'clustering_coef': infected_person_stats['clustering_coef']
})

infected_df = infected_df.dropna()
print(infected_df.describe())

# Calculate Average Clustering Coefficient Per Time Step
avg_clustering_coef_df = infected_df.groupby('timeStep')['clustering_coef'].mean().reset_index()
avg_clustering_coef_df.rename(columns={'clustering_coef': 'avg_clustering_coef'}, inplace=True)

# Visualising Average Clustering Coefficient Over Time
plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_clustering_coef_df, x='timeStep', y='avg_clustering_coef', marker='o', color='blue', linewidth=2.5)
plt.title('Change in Average Clustering Coefficient Over Time')
plt.xlabel('Time Step')
plt.ylabel('Average Clustering Coefficient')
plt.grid()
plt.show()

# Merging Data for Analysis
overall_stats.rename(columns={overall_stats.columns[0]: 'timeStep'}, inplace=True)
overall_stats_with_clustering = overall_stats.merge(avg_clustering_coef_df, on='timeStep', how='left')

print(overall_stats_with_clustering.info())
print(overall_stats_with_clustering.describe())
print(overall_stats_with_clustering.head())

# Pairwise Plot using Seaborn
sns.pairplot(overall_stats_with_clustering)
plt.suptitle('Pairwise Plot of All Variables', y=1.02)
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=overall_stats_with_clustering, x='avg_clustering_coef', y='Total.Deaths', color='blue', s=100)
sns.regplot(data=overall_stats_with_clustering, x='avg_clustering_coef', y='Total.Deaths', scatter=False, color='red')
plt.title('Total Deaths vs. Average Clustering Coefficient')
plt.xlabel('Average Clustering Coefficient')
plt.ylabel('Total Deaths')
plt.grid()
plt.show()

# Correlation Analysis
correlation, p_value = stats.pearsonr(overall_stats_with_clustering['avg_clustering_coef'].dropna(), 
                                       overall_stats_with_clustering['Total.Deaths'].dropna())
print(f'Correlation: {correlation}, P-value: {p_value}')

# Linear Regression Analysis
X = overall_stats_with_clustering['avg_clustering_coef']
y = overall_stats_with_clustering['Total.Deaths']
X = sm.add_constant(X)  # Adding intercept
lm_model = sm.OLS(y, X, missing='drop').fit()
print(lm_model.summary())

# ANOVA Analysis
anova_model = ols('Total.Deaths ~ C(timeStep)', data=overall_stats_with_clustering).fit()
anova_results = sm.stats.anova_lm(anova_model, typ=2)
print(anova_results)

# Results
# Correlation Analysis Result
print(f'Correlation: {correlation}, P-value: {p_value}')

# Linear Regression Analysis Result
print(lm_model.summary())

# ANOVA Analysis Result
print(anova_results)



