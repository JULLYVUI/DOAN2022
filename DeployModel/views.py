from django.http import HttpResponse
from django.shortcuts import render
from django.contrib import messages
from tablib import Dataset
import joblib
import pandas as pd


def home(request):
    return  render(request,"home.html")

def result(request):
    if request.method =="POST":
        file= request.FILES["myFile"]
        csv=pd.read_csv(file)
        print(csv.head())
        arr=csv["sum"]
        sumation=sum(arr)
        return render(request,"result.html",{'something':True,'sum':sumation})
    else:
        return render(request,"result.html")

def simple_upload(request):
    if  request.method == "POST":
        file= request.FILES["myFile"]
        dataset=Dataset()
        if not file.name.endswith("csv"):
            messages.info(request,'wrong format')
            return  render(request,"result.html")
        imported_data = dataset.load(file.read().decode('utf-8'),format='csv')
        # imported_data = dataset.load(file.read().decode('utf-8'),format='csv')
        x1 = [x for x in imported_data]
        cls=joblib.load('finalized_model.sav')
        predict=cls.predict(x1)
        return render(request,"result.html",{'something':True,'predict':predict})
    else:
        return render(request,"result.html")

def upload(request):
    return  render(request,"fileupload.html")