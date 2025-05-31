from django.shortcuts import render, HttpResponse
from django.contrib import messages
from users.models import UserRegistrationModel

# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request,'admins/viewregisterusers.html',{'data':data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/viewregisterusers.html',{'data':data})


def BlockUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'waiting'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/viewregisterusers.html',{'data':data})



def AdminViewResults(request):
    import pandas as pd
    from users.utility import Ransomware_Classification
    rf_report = Ransomware_Classification.process_randomForest()
    dt_report = Ransomware_Classification.process_decisionTree()
    nb_report = Ransomware_Classification.process_naiveBayes()
    lg_report = Ransomware_Classification.process_logisticRegression()
    nn_report = Ransomware_Classification.process_neuralNetwork()
    rf_report = pd.DataFrame(rf_report).transpose()
    rf_report = pd.DataFrame(rf_report)
    dt_report = pd.DataFrame(dt_report).transpose()
    dt_report = pd.DataFrame(dt_report)
    nb_report = pd.DataFrame(nb_report).transpose()
    nb_report = pd.DataFrame(nb_report)
    lg_report = pd.DataFrame(lg_report).transpose()
    lg_report = pd.DataFrame(lg_report)
    nn_report = pd.DataFrame(nn_report).transpose()
    nn_report = pd.DataFrame(nn_report)
    # # report_df.to_csv("rf_report.csv")
    return render(request, 'admins/ml_reports.html',
                  {'rf': rf_report.to_html, 'dt': dt_report.to_html, 'nb': nb_report.to_html, 'lg': lg_report.to_html,
                   'nn': nn_report.to_html})




