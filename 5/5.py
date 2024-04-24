''' use assn5 dataset '''
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, classification_report 
# Load the CSV file into a DataFrame 
df = pd.read_csv(r"C:\Users\chava\dsbda\5\Social_Network_Ads.csv") 
# Display first few rows and information about the DataFrame 
print(df.head(10)) 
print(df.info()) 
# Detect outliers using IQR (assuming 'Age' column for this example) 
Score = df['Age'] 
q1, q3 = Score.quantile(0.25), Score.quantile(0.75) 
iqr = q3 - q1 
lower_fence = q1 - (1.5 * iqr) 
upper_fence = q3 + (1.5 * iqr) 
outliers = df[(df['Age'] < lower_fence) | (df['Age'] > upper_fence)] 
print("Outliers detected using IQR:") 
print(outliers) 
# Prepare features (x) and target (y) variables 
x = df[['Age', 'EstimatedSalary']] # Adjust column names as per your DataFrame 
y = df['Purchased'] # Adjust column name for target variable 
# Split data into training and testing sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0) 
# Feature scaling 
sc = StandardScaler() 
5.
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test) 
# Train a Logistic Regression model 
classifier = LogisticRegression(random_state=0) 
classifier.fit(x_train, y_train) 
# Make predictions 
y_pred = classifier.predict(x_test) 
# Evaluate the model 
cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:") 
print(cm) 
c_report = classification_report(y_test, y_pred) 
print("Classification Report:") 
print(c_report) 
# Calculate and print accuracy, precision, recall, and F1-score 
tn, fp, fn, tp = cm.ravel() 
accuracy = (tp + tn) / (tp + tn + fp + fn) 
precision = tp / (tp + fp) 
recall = tp / (tp + fn) 
f1_score = 2 * (precision * recall) / (precision + recall) 
print("Accuracy:", accuracy) 
print("Precision:", precision) 
print("Recall:", recall) 
print("F1-Score:", f1_score)