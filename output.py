import csv 
import numpy as np
import os
import random
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
import math

def read_csv_list(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)
    return np.array(csv_list)[1:]
def write_csv_list(filename, list_rows):
    np.savetxt(filename, list_rows, delimiter =",",fmt ='% s')#

satisfaction = read_csv_list('satisfaction.csv')


Test_IDs = read_csv_list('Test_IDs.csv')
status = read_csv_list('status.csv')
demographics = read_csv_list('demographics_filled.csv')
location = read_csv_list('location_filled.csv')
population = read_csv_list('population.csv')
satisfaction = read_csv_list('satisfaction.csv')
services = read_csv_list('services_filled.csv')

def float_or_nan(d):
    if d != '':
        return float(d)
    else:
        return float("nan") 

print(demographics)
to_num = {'Male':0,'Female':1,'Yes':0,'No':1,'None':0, 'Offer A':1, 'Offer B':2, 'Offer C':3, \
    'Offer D':4, 'Offer E':5, 'DSL':1, 'Fiber Optic':2, 'Cable':3, 'Month-to-Month':0, 'One Year':1, \
        'Two Year':2, 'Bank Withdrawal':0, 'Credit Card':1, 'Mailed Check':2,'':float("nan") }

population_dict = {d[1]:float_or_nan(d[2]) for d in population if d[2] != ''}
demographics_dict = {d[0]:[to_num[d[2]],float_or_nan(d[3]),to_num[d[6]],to_num[d[7]],float_or_nan(d[8])]  for d in demographics}
location_dict = {d[0]:[float_or_nan(d[7]),float_or_nan(d[8])] for d in location if d[7] != '' and d[8] != ''}
services_dict = {d[0]:[to_num[d[3]],float_or_nan(d[4]),float_or_nan(d[5]), to_num[d[6]],to_num[d[7]], \
    float_or_nan(d[8]), to_num[d[9]],to_num[d[10]],to_num[d[11]],float_or_nan(d[12]), to_num[d[13]],\
    to_num[d[14]],to_num[d[15]],to_num[d[16]],to_num[d[17]],to_num[d[18]],to_num[d[19]],\
    to_num[d[20]],to_num[d[21]],to_num[d[22]],to_num[d[23]], float_or_nan(d[24]),\
    float_or_nan(d[25]),float_or_nan(d[26]),float_or_nan(d[27]),float_or_nan(d[28]),float_or_nan(d[29])] for d in services}




#Train:
label = {}
label_map = {'No Churn':0, 'Competitor':1, 'Dissatisfaction':2, 'Attitude':3, 'Price':4, 'Other':5}
for s in status:
    label[s[0]] = label_map[s[1]]
X_train, y_train = [], []
# X_val, y_val = [], []
for l in label:
    x = []
    if l in demographics_dict:
        x += demographics_dict[l]
    else:
        x += [math.nan]*5

    if l in population_dict:
        x += population_dict[l]
    else:
        x += [math.nan]

    if l in location_dict:
        x += location_dict[l]
    else:
        x += [math.nan]*2

    if l in services_dict:
        x += services_dict[l]
        if len(services_dict[l]) != 27:
            print(services_dict[l],l)
            input()
    else:
        x += [math.nan]*27
    X_train.append(x) 
    y_train.append(label[l])   
X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train,y_train)
print(X_train.shape,y_train.shape)

X_train, y_train = np.array(X_train),np.array(y_train)
len_x = len(X_train)//10
X_train, X_val = X_train[:-len_x], X_train[-len_x:]
y_train, y_val = y_train[:-len_x], y_train[-len_x:]

xgb_estimator = xgb.XGBClassifier(objective='count:poisson',max_depth=15)
clf_multilabel = OneVsRestClassifier(xgb_estimator)
clf_multilabel.fit(X_train, y_train)

print('Accuracy on val data: {:.1f}%'.format(accuracy_score(y_val, clf_multilabel.predict(X_val))*100))
# print(f'y_val:{y_val}')

#Predict:

X_test = []
test = [t0 for [t0] in Test_IDs]

for l in test:
    x = []
    if l in demographics_dict:
        x += demographics_dict[l]
    else:
        x += [math.nan]*5

    if l in population_dict:
        x += population_dict[l]
    else:
        x += [math.nan]

    if l in location_dict:
        x += location_dict[l]
    else:
        x += [math.nan]*2

    if l in services_dict:
        x += services_dict[l]
        if len(services_dict[l]) != 27:
            print(services_dict[l],l)
            input()
    else:
        x += [math.nan]*27
    X_test.append(x) 
X_test = np.array(X_test)
y_test = clf_multilabel.predict(X_test)
print(f'y_test:{y_test},len:{len(y_test)}')
# for i in range(len(y_test)//100):
#     print(y_test[i*100:(i+1)*100])
#     input()

sa_dict = {s[0]:s[1] for s in satisfaction if not s[1] == '' }
submit = np.array([[tt[0],''] for tt in Test_IDs])
t = 0
s = 0
for i,test in enumerate(Test_IDs[1:]):
    test_id = test[0]
    if test_id in sa_dict:
        score = float_or_nan(sa_dict[test_id])
        # print(score,test_id)
        # input()
        if score >= 4:
            submit[i][1] = 0
            s+=1
        else:#not sure with satisfaction
            pass
        t+=1
    else: #no satisfaction data
        pass
print(submit)
print(len(Test_IDs),t,s)
write_csv_list('Test_IDs_filled.csv', submit)
Test_IDs = read_csv_list('Test_IDs_filled.csv')


