import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
titanic = sns.load_dataset("titanic") 
print("--- TITANIC ---") 
print(titanic) 
titanic.head(10) 
titanic.info 
titanic.describe() 
titanic.loc[:,["survived", "alive"]] 
sns.boxplot(x = "sex", y = "age", data = titanic) 
plt.show() 
sns.boxplot(x = "sex", y = "age", data = titanic, hue = "survived") 
plt.show()