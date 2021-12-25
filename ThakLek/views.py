from django.http import HttpResponse
from django.shortcuts import render
from django.contrib import messages
from tablib import Dataset
import joblib
import pandas as pd
from pandas import concat
from pandas import DataFrame
import json 

def index(request):
    return  render(request,"home.html")

def result1(request):
    #Code lan1
    cls=joblib.load('finalized_model.sav')
    lis=[]
    lis.append(request.GET['w7'])
    lis.append(request.GET['w5'])
    lis.append(request.GET['w3'])
    lis.append(request.GET['hnk2'])
    lis.append(request.GET['hnk1'])
    lis.append(request.GET['hnk'])
    lis.append(request.GET['htk2'])
    lis.append(request.GET['htk1'])
    lis.append(request.GET['htk'])
    sumation=cls.predict([lis])
    return render(request,"result1.html",{'something':True,'sum':sumation,'lis':lis})

# convert series to supervised learning

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def simple_upload(request):
    if  request.method == "POST":
        file= request.FILES["myFile"]
        # dataset=Dataset()usecols=[1,2,3,4,5,6,7,8,9]
        imported_data = pd.read_excel(file,usecols=[1,2,3,4,5,6,7,8,9],engine='openpyxl')     
        values = imported_data.values
        # print(values)
        # transform the time series data into supervised learning9,10,11,12,13, 14, 15, 16, 17
        data1 = series_to_supervised(values,1,1)
        data1.drop(data1.columns[[9,10,11,12,13, 14, 15, 16,17]], axis=1, inplace=True)
        json_records = imported_data.reset_index().to_json(orient ='records') 
        data2 = [] 
        # context = {'d': data2} 
        if not file.name.endswith("xlsx"):
            messages.info(request,'wrong format')
            return  render(request,"result.html")
        data1 = data1.values
        x1 = [x for x in data1]
        cls=joblib.load('finalized_model.sav')
        predict=cls.predict(x1)
        data2 = json.loads(json_records) 
        return render(request,"result.html",{'something':True,'predict':predict,'d':data2})
    else:
        return render(request,"result.html")

def upload(request):
    return  render(request,"fileupload.html")