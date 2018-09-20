from flask import Flask, jsonify
from datetime import date,datetime,timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
app = Flask(__name__)
df = pd.read_excel("Sample.xls")
dataframe = pd.DataFrame(df)
furniture = df.loc[df['Category'] == 'Furniture']
# furniture['Order Date'].min()
# furniture['Order Date'].max()
cols = ['Row ID','Order ID','Ship Date','Ship Mode','Customer ID','Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols,axis=1,inplace=True)#sort with dates
furniture = furniture.sort_values('Order Date')
dt = furniture['Order Date'].values.astype(int)
sales = furniture['Sales'].values

#print("dt: ", dt)
dt = dt.reshape(-1, 1)


#print("sales:", sales)
#dt = dataframe.iloc[:,0].values.reshape(-1,1)
#sales = dataframe.iloc[:,1].values
dt_train,dt_test,sales_train,sales_test = train_test_split(dt,sales,test_size = 0.2)
regressor = RandomForestRegressor(n_estimators=300)
regressor = regressor.fit(dt_train,sales_train)
plt.scatter(dt_train,sales_train,color='red')
plt.scatter(dt_test,regressor.predict(dt_test),color='blue')
plt.title('Date vs Sales (Training Set)')
plt.xlabel('Date')
plt.ylabel('Sale')
plt.show()
# scaler = StandardScaler()
# scaler.fit(dt_train)
# dt_train = scaler.transform(dt_train)
# dt_test = scaler.transform(dt_test)
#print("dt_test: ", dt_test)
#print("dt_train: ", dt_train)
#print("sales_train: ", sales_train)
#print("sales_test: ", sales_test)
#linear Regression
@app.route("/randomforest")
def randomforest():
   sales_pred = regressor.predict(dt_test)
   b = np.array2string(sales_pred.astype('int'))
   return ((np.array2string(regressor.score(dt_test,sales_test))) + b)

   # print("Sales Prediction:", sales_pred)
if __name__ == '__main__':
   app.run(debug=True)