import numpy as np

def gen_X_y(data):#data: train_data, test_data
    return data.T[:-1].T, data.T[-1]    

def fi(x,Q):
    fix = np.array([])
    fix = np.concatenate((np.ones((len(x),1)), x),axis=1)
    
    for i in range(2,Q+1):
        fix = np.concatenate((fix, x**i),axis=1)

    return fix


train_data = []
with open('hw4_train.dat.txt') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split(" ")
        train_data.append(k)
test_data = []
with open('hw4_test.dat.txt') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split(" ")
        test_data.append(k)

train_data = np.array(train_data,dtype=float)
test_data = np.array(test_data,dtype=float)
print('train_data len:',len(train_data))
print('test_data:',test_data)