from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from .models import PredResults
import datetime


def predict(request):
    return render(request, 'predict.html')


def about_page(request):
    return render(request, 'about.html')


def predict_chances(request):

    if request.POST.get('action') == 'post':

        # Receive data from client
        age = float(request.POST.get('age'))
        sex = float(request.POST.get('sex'))
        cp = float(request.POST.get('cp'))
        trestbps = float(request.POST.get('trestbps'))
        chol = float(request.POST.get('chol'))
        fbs = float(request.POST.get('fbs'))
        restecg = float(request.POST.get('restecg'))
        thalach = float(request.POST.get('thalach'))
        exang = float(request.POST.get('exang'))
        oldpeak = float(request.POST.get('oldpeak'))
        slope = float(request.POST.get('slope'))
        ca = float(request.POST.get('ca'))
        thal = float(request.POST.get('thal'))

        # Unpickle model
        model = pd.read_pickle(r"machine_learning_stuff/new_model.pickle")
        # Make prediction
        result = model.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        classification = int(result[0])

        PredResults.objects.create(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs,
                                   restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal, classification=classification)

        now = datetime.datetime.now()
        return JsonResponse({'result': classification, 'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}, safe=False)


def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}
    return render(request, "results.html", data)
