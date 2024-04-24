import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np # Import NumPy library 
# Load the Titanic dataset 
titanic = sns.load_dataset("titanic") 
# Display basic information about the dataset 
print("--- TITANIC ---") 
print(titanic.head()) # Display the first few rows of the dataset 
# Extract and print the 'fare' column 
x = titanic["fare"] 
print("--- FARE ---") 
print(x.head()) 
# Display summary statistics of the dataset 
print("--- SUMMARY STATISTICS ---") 
print(titanic.describe()) 
# Data cleanup: Remove unnecessary columns 
titanic_cleaned = titanic.drop(['pclass', 'embarked', 'deck', 'embark_town'], axis=1) 
print("--- CLEANED DATASET ---") 
print(titanic_cleaned.head(15)) # Display the cleaned dataset 
# Display information about the cleaned dataset 
print("--- CLEANED DATASET INFO ---") 
print(titanic_cleaned.info()) 
# Check for missing values in the cleaned dataset 
print("--- MISSING VALUES ---") 
8.
print(titanic_cleaned.isnull().sum()) 
# Exclude non-numeric columns for correlation computation 
numeric_columns = titanic_cleaned.select_dtypes(include=[np.number]).columns 
correlation_matrix = titanic_cleaned[numeric_columns].corr() 
# Display correlation matrix 
print("--- CORRELATION MATRIX ---") 
print(correlation_matrix) 
# Visualize age distribution using a histogram 
a1 = titanic['age'].dropna() # Drop missing values from the 'age' column 
sns.histplot(a1, kde=True) 
plt.title('Age Distribution') 
plt.xlabel('Age') 
plt.show() 
# Visualize fare distribution using a histogram 
a2 = titanic['fare'].dropna() # Drop missing values from the 'fare' column 
sns.histplot(a2, kde=True) 
plt.title('Fare Distribution') 
plt.xlabel('Fare') 
plt.show() 
# Visualize number of parents/children aboard using a count plot 
sns.countplot(x='parch', data=titanic) 
plt.title('Number of Parents/Children Aboard') 
plt.xlabel('Number of Parents/Children') 
plt.show()