from flask import Flask, jsonify
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xlrd
app = Flask(__name__)
dataset = pd.read_excel("Sample.xls")
df = pd.DataFrame(dataset)
furniture = df.loc[df['Category'] == 'Furniture']
furniture.drop(df.columns.difference(['Order Date','Sales']), 1, inplace=True)

@app.route("/rf")
def rf():
	dt = furniture['Order Date'].values.astype('int')
	sales = furniture['Sales'].values
	dt = dt.reshape(-1, 1)
	dt_train,dt_test,sales_train,sales_test = train_test_split(dt,sales,test_size = 0.2)
	regressor = RandomForestRegressor(n_estimators=300)
	regressor = regressor.fit(dt_train,sales_train)
	sales_pred = regressor.predict(dt_test)
   	data = {
   	'SalesPredictions' : list(sales_pred)
   	}
   	return jsonify(data)
if __name__ == '__main__':
	app.run(debug=True)
