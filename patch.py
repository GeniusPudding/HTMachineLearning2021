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
    #list_rows = [ ['ABC', 'COE', '2', '9.0'],['TUV', 'COE', '2', '9.1'], ['XYZ', 'IT', '2', '9.3'],['PQR', 'SE', '1', '9.5']]  
    np.savetxt(filename, list_rows, delimiter =",",fmt ='% s')

boolstr = ['Yes', 'No']
gender = ['Female', 'Male']
demographics = read_csv_list('demographics.csv')
location = read_csv_list('location.csv')
population = read_csv_list('population.csv')[1:]
satisfaction = read_csv_list('satisfaction.csv')
services = read_csv_list('services.csv')
population_dict = {}
for d in population:
    population_dict[d[1]] = d[2]
input(population_dict)
for i, l in enumerate(location[1:]):
    # print(l)
    # input()
    if (l[-2] == '' or l[-1] == '') and l[-3] != '':
        location[i+1][-2] = l[-3][0]
        location[i+1][-1] = l[-3][1]
    elif l[-3] == ''  and l[-2] != '' and l[-1] != '' :
        location[i+1][-3] = l[-2]+', '+l[-1]
print(location)
# for i, d in enumerate(demographics[1:]):
#     if d[-2] == 'No' and d[-1] == '':
#         demographics[i+1][-1] = '0.0'
#     elif d[-2] == 'Yes' and d[-1] == '':
#         demographics[i+1][-1] = str(random.randint(1, 4))
#     elif d[-2] == '' and d[-1] == '':
#         demographics[i+1][-1] = str(0) 
#         demographics[i+1][-2] = 'No' 
#     elif d[-2] == '' and float(d[-1]) == 0:
#         demographics[i+1][-2] = 'No'
#     elif d[-2] == '':
#         demographics[i+1][-2] = 'Yes'

#     if d[-3] == '':
#         demographics[i+1][-3] = boolstr[random.randint(0, 1)]
#     if d[2] == '':
#         demographics[i+1][2] = gender[random.randint(0, 1)]
#     if d[4] == '' and d[3] == '':
#         ran = random.randint(19, 80)
#         demographics[i+1][3] = str(ran)
#         if ran < 30:
#             demographics[i+1][4] = 'Yes'      
#         else:
#             demographics[i+1][4] = 'No'
#     elif d[4] == '':
#         if float(d[3]) < 30:
#             demographics[i+1][4] = 'Yes'      
#         else:
#             demographics[i+1][4] = 'No'
#     elif d[3] == '':
#         if d[4] == 'Yes':
#             demographics[i+1][3] = str(random.randint(19, 29) )    
#         else:
#             demographics[i+1][3] = str(random.randint(30, 80) )

# print(f'demographics:{demographics}\n')
# write_csv_list('demographics_filled.csv', demographics)