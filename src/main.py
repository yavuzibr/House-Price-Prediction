# Dependencies
# pandas for data manipulation
# scikit-learn for ml model
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# assigning dataset to arrays
data=pd.read_csv('dataset/data.csv')
train=pd.read_csv('dataset/train.csv')
test=pd.read_csv('dataset/test.csv')

# In Linear Regression our purpose is to make predictions with numbers (integers,float,etc.)
# That is why we are converting or removing columns with string type
columns_to_convert = ['elevator', 'within_the_site']
for col in columns_to_convert:
    data[col] = data[col].map({'Yes': 1, 'No': 0})
    train[col] = train[col].map({'Yes':1,'No':0})
    test[col] = train[col].map({'Yes':1,'No':0})


num_data=data[['unique_id','area_msquare','age','total_room','floor','price','distance_to_center','distance_to_hospital','within_the_site','elevator']]
train_data=train[['unique_id','area_msquare','age','total_room','floor','price','distance_to_center','distance_to_hospital','within_the_site','elevator']]
test_data=test[['unique_id','area_msquare','age','total_room','floor','price','distance_to_center','distance_to_hospital','within_the_site','elevator']]

# Creating Linear Regression Model
reg =LinearRegression()
predictors = ['area_msquare','total_room']
target='price'
reg.fit(train_data[predictors],train_data['price'])
predictions=reg.predict(test[predictors])

# Create and assign the output predictions to test dataset
test["predictions"]=predictions
# Absolute values to convert negative values
test["predictions"]=test["predictions"].abs()

# Calculate Error 
error = mean_absolute_error(test["price"],test["predictions"])



