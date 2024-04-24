import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error,r2_score 

df=pd.read_csv(r'C:\Users\chava\dsbda\4\train.csv') 
# df=pd.read_csv(r'C:\Users\chava\dsbda\4\HousingData.csv') 

print("=======Info========") 
print(df) 
print("=======Head========") 
print(df.head(5)) 
print("=======Shape========") 
print(df.shape) 
print("=======isna========") 
print(df.isna()) 
print("=======column========") 
print(df.columns) 
print("=======isnull========") 
print(df.isnull()) 
print("=======choose features for prediction=======") 
X=df[['RM','LSTAT','CRIM']] 
Y=df['MEDV'] 
print(X) 
print(Y) 
print("==========Split the data into training and testing sets=========") 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42) 
print("=======x X_train========") 
print(X_train.shape) 
print("=======Y_train========") 
print(Y_train.shape) 
print("=======X_test========") 
print(X_test.shape) 
print("=======Y_test========") 
print(Y_test.shape) 
print("=========Create a linear regression model=========") 
model=LinearRegression() 
model.fit(X_train,Y_train) 
print("=========Predicted values=========") 
Y_pred=model.predict(X_test) 
print(Y_pred) 
print("=========Mean score=========") 
mse=mean_squared_error(Y_test,Y_pred) 
print("Mean squared error: ",mse) 
print("=========r2_score=========") 
r2=r2_score(Y_test,Y_pred) 
print("R-squared: ",r2) 
print("=========Scatter plot=========") 
plt.scatter(Y_test,Y_pred) 
plt.plot([min(Y_test),max(Y_test)],[min(Y_pred),max(Y_pred)]) 
plt.xlabel("Actual prices") 
plt.ylabel("Prediced prices") 
plt.title("Actual prices vs Prediced prices") 
plt.show()


# #2
# import numpy as np 
# import pandas as pd 
# import matplotlib.pyplot as plt 
# import seaborn as sns 
# from sklearn.model_selection import train_test_split 
# from sklearn.linear_model import LinearRegression 
# from sklearn.metrics import mean_squared_error, r2_score 

# df = pd.read_csv(r'C:\Users\chava\dsbda\4\train.csv') 

# print("=======Info========") 
# print(df) 
# print("=======Head========") 
# print(df.head(5)) 
# print("=======Shape========") 
# print(df.shape) 
# print("=======isna========") 
# print(df.isna()) 
# print("=======column========") 
# print(df.columns) 
# print("=======isnull========") 
# print(df.isnull()) 

# print("=======choose features for prediction=======") 
# # Correct column names: 'RM', 'LSTAT', 'CRIM'
# X = df[['RM', 'LSTAT', 'CRIM']] 
# Y = df['MEDV'] 
# print(X) 
# print(Y) 

# print("==========Split the data into training and testing sets=========") 
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 
# print("=======X_train========") 
# print(X_train.shape) 
# print("=======Y_train========") 
# print(Y_train.shape) 
# print("=======X_test========") 
# print(X_test.shape) 
# print("=======Y_test========") 
# print(Y_test.shape) 

# print("=========Create a linear regression model=========") 
# model = LinearRegression() 
# model.fit(X_train, Y_train) 

# print("=========Predicted values=========") 
# Y_pred = model.predict(X_test) 
# print(Y_pred) 

# print("=========Mean score=========") 
# mse = mean_squared_error(Y_test, Y_pred) 
# print("Mean squared error: ", mse) 

# print("=========r2_score=========") 
# r2 = r2_score(Y_test, Y_pred) 
# print("R-squared: ", r2) 

# print("=========Scatter plot=========") 
# plt.scatter(Y_test, Y_pred) 
# plt.plot([min(Y_test), max(Y_test)], [min(Y_pred), max(Y_pred)]) 
# plt.xlabel("Actual prices") 
# plt.ylabel("Predicted prices") 
# plt.title("Actual prices vs Predicted prices") 
# plt.show()



# import numpy as np 
# import pandas as pd 
# import matplotlib.pyplot as plt 
# import seaborn as sns 
# from sklearn.model_selection import train_test_split 
# from sklearn.linear_model import LinearRegression 
# from sklearn.metrics import mean_squared_error, r2_score 

# # Read the CSV file
# df = pd.read_csv(r'C:\Users\chava\dsbda\4\train.csv') 

# # Print information about the DataFrame
# print("=======Info========") 
# print(df.info()) 

# # Print the first few rows of the DataFrame
# print("=======Head========") 
# print(df.head(5)) 

# # Print the shape of the DataFrame
# print("=======Shape========") 
# print(df.shape) 

# # Check for missing values
# print("=======isna========") 
# print(df.isna().sum()) 

# # Check the column names
# print("=======column========") 
# print(df.columns) 

# # Check for null values
# print("=======isnull========") 
# print(df.isnull().sum()) 

# # Choose features for prediction
# # Adjust this part based on your dataset and prediction task
# # For now, we're not selecting specific columns
# # X = df[['RM', 'LSTAT', 'CRIM']] 
# # Y = df['MEDV'] 

# # Split the data into training and testing sets
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 

# # Print the shapes of training and testing sets
# # print("=======x X_train========") 
# # print(X_train.shape) 
# # print("=======Y_train========") 
# # print(Y_train.shape) 
# # print("=======X_test========") 
# # print(X_test.shape) 
# # print("=======Y_test========") 
# # print(Y_test.shape) 

# # Create a linear regression model
# # model = LinearRegression() 
# # model.fit(X_train, Y_train) 

# # Predicted values
# # Y_pred = model.predict(X_test) 

# # Compute mean squared error
# # mse = mean_squared_error(Y_test, Y_pred) 
# # print("Mean squared error: ", mse) 

# # Compute r-squared
# # r2 = r2_score(Y_test, Y_pred) 
# # print("R-squared: ", r2) 

# # Scatter plot
# # plt.scatter(Y_test, Y_pred) 
# # plt.plot([min(Y_test), max(Y_test)], [min(Y_pred), max(Y_pred)]) 
# # plt.xlabel("Actual prices") 
# # plt.ylabel("Prediced prices") 
# # plt.title("Actual prices vs Prediced prices") 
# # plt.show()
