import csv 
import numpy as np
import os
import random


def read_csv_list(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)
    return np.array(csv_list)
def write_csv_list(filename, list_rows):
    np.savetxt(filename, list_rows, delimiter =",",fmt ='% s')#

satisfaction = read_csv_list('satisfaction.csv')[1:]
services = read_csv_list('services.csv')
status = read_csv_list('status.csv')[1:]
Test_IDs = read_csv_list('Test_IDs.csv')
label = {}
label_map = {'No Churn':0, 'Competitor':1, 'Dissatisfaction':2, 'Attitude':3, 'Price':4, 'Other':5}
for s in status:
    label[s[0]] = label_map[s[1]]

# sa_id = [s[0] for s in satisfaction if not s[1] == '' ]
sa_dict = {s[0]:s[1] for s in satisfaction if not s[1] == '' }
t = 0
for i,test in enumerate(Test_IDs[1:]):
    test_id = test[0]
    if test_id in sa_dict:
        score = float(sa_dict[test_id])
        # print(score,test_id)
        # input()
        if score >= 4:
            print(Test_IDs[i])
            Test_IDs[i] = np.concatenate(np.array(Test_IDs[i]), np.array(['0']))

        t+=1
print(Test_IDs)
print(len(Test_IDs),t)

    # if test in satisfaction:
    #     input()

print(f'Test_IDs:{Test_IDs}\n')
write_csv_list('Test_IDs_filled.csv', Test_IDs)
