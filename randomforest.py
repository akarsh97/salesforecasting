from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xlrd
app = Flask(__name__)
df = pd.read_excel("Sample.xls")
dataframe = pd.DataFrame(df)
furniture = df.loc[df['Category'] == 'Furniture']
cols = ['Row ID','Order ID','Ship Date','Ship Mode','Customer ID','Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols,axis=1,inplace=True)#sort with dates
furniture = furniture.sort_values('Order Date')
dt = furniture['Order Date'].values.astype(int)
sales = furniture['Sales'].values
dt = dt.reshape(-1, 1)
dt_train,dt_test,sales_train,sales_test = train_test_split(dt,sales,test_size = 0.2)
regressor = RandomForestRegressor(n_estimators=300)
regressor = regressor.fit(dt_train,sales_train)
sales_pred = regressor.predict(dt_test)
@app.route("/randomforest")
def randomforest():
   	data = {'SalesPredictions' : sales_pred.tolist()}
   	return jsonify(data)
if __name__ == '__main__':
	app.run(debug=True)
