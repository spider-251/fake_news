from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.utils import timezone

import pandas as pd 
#import the models here
from .models import News

#App Functions
df=pd.read_csv("fapp\\static\\data\\sample_data (2).csv")

from .fake_news_main import *
model,device,tokenizer = model_load()

# Create your views here.
def homepage(request):
    """ Show the homepage with input search key  and validate the News"""
    try:
        ns=request.POST["news"]
        text_path = write_to_file(ns)
        valids_dataloader = valids_dl(text_path,tokenizer)
        logits, prediction_labels = validation(valids_dataloader,device,model)
        lg,fk = sigmoid(logits[0])
        #lg,fk= 69.8,42.95
        if(lg<fk):
            lbl="Fake"
        else :
            lbl="Legit"

        n=News(input=ns,search_date=timezone.now(),fk_prob=fk,lg_prob=lg,label=lbl)
        #n.save()
        args = {"news" : ns, "submitted":1,"lg_prob":"{:.2f}".format(lg) ,"fk_prob":"{:.2f}".format(fk)}
        return render(request, "homepage.html", args)
    except Exception as e:
        print("Exception Raised",(e))
        args = {"df":df,"submitted":5}
        return render(request, "homepage.html", args)

#page not found function
def page_not_found(request, exception):
    return render(request, '404.html', status=404)

def handler505(request):
    return render(request, "505.html", status=505)
    