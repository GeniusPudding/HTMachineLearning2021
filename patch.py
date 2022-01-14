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

boolstr = ['Yes', 'No']
gender = ['Female', 'Male']
demographics = read_csv_list('demographics.csv')
location = read_csv_list('location.csv')
population = read_csv_list('population.csv')[1:]
satisfaction = read_csv_list('satisfaction.csv')[1:]
services = read_csv_list('services.csv')
status = read_csv_list('status.csv')[1:]
Test_IDs = read_csv_list('Test_IDs.csv')[1:]
Train_IDs = read_csv_list('Train_IDs.csv')[1:]
label = {}
label_map = {'No Churn':0, 'Competitor':1, 'Dissatisfaction':2, 'Attitude':3, 'Price':4, 'Other':5}
for s in status:
    label[s[0]] = label_map[s[1]]


# print(label)
yes = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
no = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]

offer = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]] # None, Offer A, Offer B, Offer C, Offer D, and Offer E
internet = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]] # None, DSL,  Fiber Optic, Cable
contract = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]] # Month-to-Month, One Year, Two Year
payment = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]] # Bank Withdrawal, Credit Card, Mailed Check
offer_map = {'None':0, 'Offer A':1, 'Offer B':2, 'Offer C':3, 'Offer D':4, 'Offer E':5}
internet_map = {'None':0, 'DSL':1, 'Fiber Optic':2, 'Cable':3}
contract_map = {'Month-to-Month':0, 'One Year':1, 'Two Year':2}
payment_map = {'Bank Withdrawal':0, 'Credit Card':1, 'Mailed Check':2}
bool_index = [3,7,9,10,13,14,15,16,17,18,19,20,22]
choice_index = [6,11,21,23]
for s in services[1:]:
    # print(s)
    # input()
    s_id = s[0] 
    if s_id not in Train_IDs or s_id not in label:
        continue 
    l_s = label[s_id]  

    if s[6] != '':
        offer[offer_map[s[6]]][l_s] += 1
    if s[11] != '':
        internet[internet_map[s[11]]][l_s] += 1
    if s[21] != '':
        contract[contract_map[s[21]]][l_s] += 1    
    if s[23] != '':
        payment[payment_map[s[23]]][l_s] += 1    
    for bi,b in enumerate(bool_index):
        if s[b] == 'Yes':
            yes[bi][l_s] += 1
        elif s[b] == 'No':
            no[bi][l_s] += 1


# print(f'offer:{offer}')
# print(f'internet:{internet}')
# print(f'contract:{contract}')
# print(f'payment:{payment}')
# offer:[[1329, 201, 75, 71, 55, 56], [222, 7, 2, 2, 1, 1], [338, 21, 12, 5, 7, 1], [145, 18, 6, 8, 10, 5], [200, 34, 20, 21, 9, 5], [167, 87, 17, 42, 21, 20]]
# internet:[[672, 10, 5, 16, 15, 0], [618, 51, 28, 28, 15, 22], [805, 247, 80, 95, 59, 59], [279, 50, 14, 19, 8, 11]]
# contract:[[917, 313, 108, 145, 84, 74], [664, 32, 14, 13, 8, 11], [836, 10, 4, 3, 4, 1]]
# payment:[[1193, 254, 87, 109, 64, 63], [1080, 74, 32, 39, 28, 23], [116, 30, 11, 8, 8, 3]]

# for i in range(13):
#     print(f'yes:{yes[i]}')
#     print(f'no:{no[i]}\n')
# 12 features in services.csv
# yes:[1178, 124, 55, 46, 28, 26]
# no:[1212, 242, 83, 105, 80, 59]

# yes:[2164, 340, 106, 123, 96, 75]
# no:[235, 25, 11, 18, 5, 9]

# yes:[949, 181, 59, 60, 48, 40]
# no:[1423, 182, 71, 94, 50, 49]

# yes:[1729, 359, 125, 143, 81, 79]
# no:[692, 6, 5, 16, 12, 1]

# yes:[790, 49, 24, 22, 22, 11]
# no:[1603, 329, 104, 127, 76, 75]

# yes:[850, 104, 34, 35, 24, 25]
# no:[1528, 251, 91, 116, 77, 68]

# yes:[836, 117, 47, 37, 22, 20]
# no:[1563, 248, 89, 115, 74, 69]

# yes:[788, 51, 22, 21, 16, 18]
# no:[1619, 309, 102, 134, 82, 62]

# yes:[876, 154, 50, 55, 44, 45]
# no:[1540, 209, 73, 92, 57, 46]

# yes:[865, 163, 49, 72, 43, 35]
# no:[1526, 198, 85, 84, 57, 56]

# yes:[803, 147, 42, 57, 33, 28]
# no:[1590, 216, 82, 90, 64, 56]

# yes:[1479, 308, 106, 117, 71, 77]
# no:[926, 59, 22, 34, 30, 9]

# yes:[1257, 276, 93, 109, 66, 64]
# no:[1134, 90, 32, 48, 28, 27]
bool_choice = ['Yes','No'] 
total = 0
blank = 0
for i,s in enumerate(services[1:]):
    # print(s)
    # input()
    
    s_id = s[0] 
    if s_id not in Train_IDs or s_id not in label:
        continue 
    l_s = label[s_id]  
    # print(f'before services[i+1]:{services[i+1]}')

    #fill boolean values:
    for bi,b in enumerate(bool_index):
        if s[b] == '':
            y_p = yes[bi][l_s]
            n_p = no[bi][l_s]
            services[i+1][b] = random.choices(bool_choice,weights=[y_p,n_p])[0]

    #fill choice values:
    if s[6] == '':
        weight_offer = [offer[v][l_s]  for v in offer_map.values()]
        services[i+1][6] = random.choices(list(offer_map.keys()),weights=weight_offer)[0]
    if s[11] == '':
        weight_internet = [internet[v][l_s]  for v in internet_map.values()]
        services[i+1][11] = random.choices(list(internet_map.keys()),weights=weight_internet)[0]
    if s[21] == '':
        weight_contract = [contract[v][l_s]  for v in contract_map.values()]
        services[i+1][21] = random.choices(list(contract_map.keys()),weights=weight_contract)[0]
    if s[23] == '':
        payment_offer = [payment[v][l_s]  for v in payment_map.values()]
        services[i+1][23] = random.choices(list(payment_map.keys()),weights=payment_offer)[0]

    # input(f'after services[i+1]:{services[i+1]}\n')
    total += 1 
    if ''  in s:
        blank += 1
print(f'blank:{blank},total:{total}')
#
write_csv_list('services_filled.csv', services)
print(services)

# population_dict = {}
# for d in population:
#     population_dict[d[1]] = d[2]
# # input(population_dict)
# for i, l in enumerate(location[1:]):
#     # print(l)
#     # input()
#     if (l[-2] == '' or l[-1] == '') and l[-3] != '':
#         location[i+1][-2] = l[-3].split(',')[0].strip(' ')
#         location[i+1][-1] = l[-3].split(',')[1].strip(' ')
#     elif l[-3] == ''  and l[-2] != '' and l[-1] != '' :
#         location[i+1][-3] = l[-2]+' '+l[-1]
#     location[i+1][-3] = location[i+1][-3].replace(',', '')
#     # print(location[i+1])
#     # input()
# write_csv_list('location_filled.csv', location)
# print(location)


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