from flask import Flask, jsonify, request
import requests
import json
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

	X = furniture['Order Date'].values.astype('int')
	y = furniture['Sales'].values
	data = {
	'SalesPredictions' : list(y),
	'date' : list(X)
	}
	return jsonify(data)

@app.route('/pred',methods = ['GET'])
def pred():
	url = "http://127.0.0.1:5000/rf"
	try:
		uResponse = requests.get(url)
	except requests.ConnectionError:
		return "Connection Error"  
	Jresponse = uResponse.text
	data = json.loads(Jresponse)
	list1 = [k for k in data['date']]
	list2 = [k for k in data['SalesPredictions']]

	array1 = np.asarray(list1).reshape(-1,1)
	array2 = np.asarray(list2)
	
	X_train,X_test,y_train,y_test = train_test_split(array1,array2,test_size=0.2)
	regressor = RandomForestRegressor()
	regressor = regressor.fit(X_test,y_test)
	sales_pred = regressor.predict(X_test)
	sales_pred = list(sales_pred)
	a = regressor.score(X_test,y_test)
	b = np.array2string(a)
	errors = abs(sales_pred - y_test)    #average absolute error-https://towardsdatascience.com/improving-random-forest-in-python-part-1-893916666cd
	return jsonify({'Sales Predictions(in Dollars)':sales_pred,'accuracy':b,'Average absolute error(in Dollars):': round(np.mean(errors), 2)})
if __name__ == '__main__':
	app.run(debug=True)
