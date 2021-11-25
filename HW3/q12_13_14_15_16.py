import numpy as np
def lin_reg_alg(X, y):
    # y = train_data.T[0]
    # X = train_data.T[1:].T
    # x_pinv = []
    try:
        x_pinv = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)
    except np.linalg.LinAlgError as e:#no inverse of XTX
        x_pinv = np.linalg.pinv(X)
    w_lin = np.matmul(x_pinv , y )
    # print('w_lin:',w_lin)
    return w_lin
def cal_01_err(w_lin, X, y, size):
    E_01_sqr = np.sum(np.abs(np.sign(np.matmul(X,w_lin)) - y)/2)/size
    # print('E_01_sqr:',E_01_sqr)    
    return E_01_sqr
def gen_X_y(data):#data: train_data, test_data
    return data.T[:-1].T, data.T[-1]    
def fi(x,Q):
    fix = np.array([])
    fix = np.concatenate((np.ones((len(x),1)), x),axis=1)
    
    for i in range(2,Q+1):
        fix = np.concatenate((fix, x**i),axis=1)

    return fix
def fi_full2(x):
    fix = np.array([])
    fix = np.concatenate((np.ones((len(x),1)), x),axis=1)
    dim = len(x[0])
    for i in range(dim):
        for j in range(i+1):   
            print()
            mul = np.multiply(x.T[i],x.T[j])
            mul = np.resize(mul,(mul.size,1))
            print(mul.shape)
            fix = np.concatenate((mul,fix),axis=1) 
    print(fix.shape)
    return fix
def fi_lower(x,i):
    fix = np.concatenate((np.ones((len(x),1)), x.T[:i].T),axis=1)
    return fix
def fi_ran_dims(x,dims):
    fix = np.ones((len(x),1))
    for d in dims:
        xd = x.T[d].T
        xd = np.resize(xd,(xd.size,1))
        # print('xd.shape:',xd.shape)  
        fix = np.concatenate((xd,fix),axis=1)
    # print('fix.shape:',fix.shape)
    return fix

train_data = []
with open('hw3_train.dat.txt') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split("\t")
        train_data.append(k)
test_data = []
with open('hw3_test.dat.txt') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split("\t")
        test_data.append(k)

train_data = np.array(train_data,dtype=float)
test_data = np.array(test_data,dtype=float)
print('train_data len:',len(train_data))
# print('test_data:',test_data)

X, y = gen_X_y(train_data)
Q = 2# q13 : Q = 8
test_y = test_data.T[-1]
test_X = test_data.T[:-1].T    

#q12~14
# # fi_X = fi(X,Q) #q12,q13
# fi_X = fi_full2(X) #q14
# print('fi_X:',fi_X)
# w_lin = lin_reg_alg(fi_X, y)
# zero_one_in_err = cal_01_err(w_lin, fi_X, y,len(train_data))
# # fi_X_test = fi(test_X,Q) #q12,q13
# fi_X_test = fi_full2(test_X) #q14
# zero_one_out_err = cal_01_err(w_lin, fi_X_test, test_y,len(fi_X_test))
# # x_1 = np.array(train[1][:-1],dtype=float)
# print('abs:',np.abs(zero_one_in_err-zero_one_out_err))
# print('zero_one_in_err,zero_one_out_err:',zero_one_in_err,zero_one_out_err)

#q15
# for i in range(1,11):
#     fi_X = fi_lower(X,i)
#     w_lin = lin_reg_alg(fi_X, y)
#     zero_one_in_err = cal_01_err(w_lin, fi_X, y,len(train_data))
#     fi_X_test = fi_lower(test_X,i)    
#     zero_one_out_err = cal_01_err(w_lin, fi_X_test, test_y,len(fi_X_test))
#     print(f'{i}-th abs err:{np.abs(zero_one_in_err-zero_one_out_err)}')

#q16
err_t = 0
rounds = 20000
for i in range(rounds):
    np.random.seed(np.random.randint(1126))
    dims = np.random.choice(10, 5, replace=False, p=None)
    # print('dims:',dims)
    fi_X = fi_ran_dims(X,dims)
    w_lin = lin_reg_alg(fi_X, y)
    zero_one_in_err = cal_01_err(w_lin, fi_X, y,len(train_data))
    fi_X_test = fi_ran_dims(test_X,dims)  
    zero_one_out_err = cal_01_err(w_lin, fi_X_test, test_y,len(fi_X_test))
    err_t +=  np.abs(zero_one_in_err-zero_one_out_err)
print('avr err:',err_t/rounds)


#q12:0.3263333333333333
#q13:0.4576666666666667
#q14:0.33866666666666667
#q15
# 1-th abs err:0.13666666666666666
# 2-th abs err:0.13433333333333336
# 3-th abs err:0.1323333333333333 
# 4-th abs err:0.1443333333333333 
# 5-th abs err:0.2523333333333333 
# 6-th abs err:0.3223333333333333 
# 7-th abs err:0.26466666666666666
# 8-th abs err:0.2653333333333333 
# 9-th abs err:0.2483333333333333 
# 10-th abs err:0.3226666666666666
#q16 avr err:0.19414191666669055