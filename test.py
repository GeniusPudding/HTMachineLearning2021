import csv 
import numpy as np
from libsvm.svmutil import *
import os
def read_csv_list(filename):
    f = open(filename)
    reader = csv.reader(f)
    csv_list = list(reader)[1:]
    # csv_list = [i[0] for i in csv_list]
    # print(len(csv_list))
    f.close()
    return np.array(csv_list)

Test_IDs = read_csv_list('Test_IDs.csv')
Train_IDs = read_csv_list('Train_IDs.csv')#Test_IDs intersection Train_IDs = NULL
status = read_csv_list('status.csv')
demographics = read_csv_list('demographics_filled.csv')
location = read_csv_list('location_filled.csv')
population = read_csv_list('population.csv')
satisfaction = read_csv_list('satisfaction.csv')
services = read_csv_list('services.csv')

label = {}
label_map = {'No Churn':0, 'Competitor':1, 'Dissatisfaction':2, 'Attitude':3, 'Price':4, 'Other':5}
for s in status:
    label[s[0]] = label_map[s[1]]
# keys = label.keys()
ts = 0
td = 0
tl = 0
tsa = 0
tse = 0 
#num of train id not in csvs:1408 698 686 693 698 
#num of test id not in csvs:1409 182 194 187 182
#num of (label)status id not in csvs:524 514 513 535
nonelabeled_id = [] 
labeled_id = []
test = [t0 for [t0] in Test_IDs]
train = [t0 for [t0] in Train_IDs]
for tr in test:
    b = False
    
    t = ts+td+tl+tsa+tse
    if tr not in status:
        ts +=1

    if tr not in demographics:
        td +=1

    if tr not in location:
        tl +=1

    if tr not in satisfaction:
        tsa +=1

    if tr not in services:
        tse +=1
    if ts+td+tl+tsa+tse - t == 5:
        input(f'id:{tr} not exist')

print(ts,td,tl,tsa,tse)
input()

population_dict = {d[1]:d[2] for d in population}
location_dict = {d[0]:[d[5],d[7],d[8]] for d in location }

    # input(d)
y = []
x = []
n = 0
for k in label: 
    if k not in location_dict:
        continue
    if '' in location_dict[k]:
        continue
    n+=1
    y.append(label[k])
    d = location_dict[k] 
    x.append([float(d[0]),float(d[1]),float(d[2])])

print(f'n:{n}')
print(f'x:{x}')
y = [0 if yi == 0 else 1 for yi in y]
print(f'y:{y}')

prob  = svm_problem(y[:-350], x[:-350])
param = svm_parameter('-s 1')
m = svm_train(prob, param)

p_label, p_acc, p_val = svm_predict(y[-350:], x[-350:], m)
input(f'p_val:{p_val},\np_label:{p_label}, len(p_label):{len(p_label)} \np_acc:{p_acc}, \n')

#satisfaction:
ratio = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
for s in satisfaction:
    # print(f's:{s}')
    if s[1] == '' or s[0] not in label:
        continue
    score = int(float(s[1]))-1
    # ratio[score][1] += 1
    # if label[s[0]] == 0: #'No Churn'
    ratio[score][label[s[0]]] += 1        
input(f'ratio:{ratio}')#ratio:[[0, 405], [0, 232], [1020, 1229], [843, 843], [520, 520]]

#demographics:
# Gender (2): (Male) 1, (Female) 0
# Married (6): 1, 0
# Age (3): int (19~80)
# Number of Dependents (-1): int (0~9)
# blank: maybe random?

# max_age, min_age = 0, 100
# for a in demographics.T[-1]:
#     if a == '':
#         continue
#     if int(float(a)) > max_age:
#         max_age = int(float(a))
#     if int(float(a)) < min_age:
#         min_age = int(float(a)) 
# print(max_age, min_age )

# s = np.zeros((81,2))
#s = np.zeros((10,2))
# for data in demographics:
#     i = data[0]
#     a = data[3]
#     if a == '' or i not in keys:
#         continue
#     a = int(float(a))
#     if label[i] == 0:
#         s[a][0]+=1
#     s[a][1]+=1
# for a,ss in enumerate(s):
#     print(f'p:{a}, ratio:{ss[0]}/{ss[1]}={ss[0]/ss[1]}')


demographics_dict = {}
demographics_map = {'Male':1, 'Female':0, 'Yes':1, 'No':0}
for d in demographics:
    demographics_dict[d[0]] = (d[2],d[3],d[4],d[5],d[6],d[7],d[8])
    # input(d)
y = []
x = []
n = 0
for k in label: 
    if k not in demographics_dict:
        continue
    if '' in demographics_dict[k]:
        continue
    n+=1
    y.append(label[k])
    d = demographics_dict[k] 
    # d1,d2,d3,d4 = demographics_map[d[0]],float(d[1]),demographics_map[d[2]],float(d[3])
    # x.append([d1,d2,d3,d4])
    d1,d2,d3,d4,d5,d6,d7 = demographics_map[d[0]],float(d[1]),demographics_map[d[2]],demographics_map[d[3]],demographics_map[d[4]],demographics_map[d[5]],float(d[6])
    x.append([d1,d2,d3,d4,d5,d6,d7])
    # d1,d2 = demographics_map[d[0]],demographics_map[d[2]]
    # x.append([d1,d2])
print(f'n:{n}')
print(f'x:{x}')
y = [0 if yi == 0 else 1 for yi in y]
print(f'y:{y}')

prob  = svm_problem(y[:-350], x[:-350])
param = svm_parameter('-s 1')
m = svm_train(prob, param)
svm_save_model('test_svm',m)
# m = svm_load_model('test_svm')
# test_x = []
# for i in Test_IDs:
#     test_id = i[0]
#     if test_id not in demographics_dict:
#         continue
#     if '' in demographics_dict[test_id]:
#         continue    
#     d = demographics_dict[test_id] 
#     # d1,d2,d3,d4 = demographics_map[d[0]],float(d[1]),demographics_map[d[2]],float(d[3])
#     # test_x.append([d1,d2,d3,d4])        
#     d1,d2,d3,d4,d5,d6,d7 = demographics_map[d[0]],float(d[1]),demographics_map[d[2]],demographics_map[d[3]],demographics_map[d[4]],demographics_map[d[5]],float(d[6])
#     test_x.append([d1,d2,d3,d4,d5,d6,d7])

p_label, p_acc, p_val = svm_predict(y[-350:], x[-350:], m)
# p_label, p_acc, p_val = svm_predict([], test_x, m)
print(f'p_val:{p_val},\np_label:{p_label}, \np_acc:{p_acc}, \n')
# print(f'test_x:{test_x}')


