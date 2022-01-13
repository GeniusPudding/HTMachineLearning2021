import numpy as np
from liblinear.liblinearutil import *

def gen_X_y(data):#data: train_data, test_data
    origin_X = data.T[:-1].T
    y = data.T[-1]   
    X = fi_full3(origin_X)

    return X,y

def fi_full3(x):
    fix = np.array([])
    fix = np.concatenate((np.ones((len(x),1)), x),axis=1)
    dim = len(x[0])
    for i in range(dim):
        for j in range(i+1):   
            mul = np.multiply(x.T[i],x.T[j])
            mul = np.resize(mul,(mul.size,1))
            # print(str(i)+str(j))
            fix = np.concatenate((mul,fix),axis=1) 

    for i in range(dim):
        for j in range(i+1):   
            for k in range(j+1):  
                # print(str(i)+str(j)+str(k)) 
                mul = np.multiply(x.T[i],x.T[j])
                mul = np.multiply(mul,x.T[k])
                mul = np.resize(mul,(mul.size,1))
                # print(mul.shape)
                fix = np.concatenate((mul,fix),axis=1)    
    # input(fix.shape)
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
X,y = gen_X_y(train_data)
testX,testy = gen_X_y(test_data)

# #q12
model1 = train(y,X,'-s 0 -e 0.000001 -c 0.00005')
model2 = train(y,X,'-s 0 -e 0.000001 -c 0.005')
model3 = train(y,X,'-s 0 -e 0.000001 -c 0.5')
model4 = train(y,X,'-s 0 -e 0.000001 -c 50')
model5 = train(y,X,'-s 0 -e 0.000001 -c 5000')

# predict(testy,testX,model1)
# predict(testy,testX,model2)
# predict(testy,testX,model3)
# predict(testy,testX,model4)
# predict(testy,testX,model5)
# # q13
# predict(y,X,model1)
# predict(y,X,model2)
# predict(y,X,model3)
# predict(y,X,model4)
# predict(y,X,model5)


#q14
# train_data_, val_data = train_data[:120],train_data[120:] 
# X,y = gen_X_y(train_data_)
# valX,valy = gen_X_y(val_data)
# # model1 = train(y,X,'-s 0 -e 0.000001 -c 0.00005')
# # model2 = train(y,X,'-s 0 -e 0.000001 -c 0.005')
# # model3 = train(y,X,'-s 0 -e 0.000001 -c 0.5')
# # model4 = train(y,X,'-s 0 -e 0.000001 -c 50')
# # model5 = train(y,X,'-s 0 -e 0.000001 -c 5000')
# 

# predict(valy,valX,model1)
# predict(valy,valX,model2)
# predict(valy,valX,model3)
# predict(valy,valX,model4)
# predict(valy,valX,model5)
# predict(testy,testX,model2)

# #q15
# model2_plus = train(y,X,'-s 0 -e 0.000001 -c 0.005')
# predict(testy,testX,model2_plus)

#q16
fold1, train1 = train_data[:40], train_data[40:]
fold2, train2 = train_data[40:80], np.concatenate((train_data[:40], train_data[80:]))
fold3, train3 = train_data[80:120], np.concatenate((train_data[:80], train_data[120:])) 
fold4, train4 = train_data[120:160], np.concatenate((train_data[:120], train_data[160:])) 
fold5, train5 = train_data[160:], train_data[:160]

x1,y1 = gen_X_y(train1)
x2,y2 = gen_X_y(train2)
x3,y3 = gen_X_y(train3)
x4,y4 = gen_X_y(train4)
x5,y5 = gen_X_y(train5)
vx1,vy1 = gen_X_y(fold1)
vx2,vy2 = gen_X_y(fold2)
vx3,vy3 = gen_X_y(fold3)
vx4,vy4 = gen_X_y(fold4)
vx5,vy5 = gen_X_y(fold5)

err1 = 0
model1 = train(y1,x1,'-s 0 -e 0.000001 -c 0.00005')
p_label,p_accm,p_val = predict(vy1,vx1,model1)
ACC,MSE,SCC = evaluations(vy1,p_label)
err1 += MSE
model1 = train(y2,x2,'-s 0 -e 0.000001 -c 0.00005')
p_label,p_accm,p_val = predict(vy2,vx2,model1)
ACC,MSE,SCC = evaluations(vy2,p_label)
err1 += MSE
model1 = train(y3,x3,'-s 0 -e 0.000001 -c 0.00005')
p_label,p_accm,p_val = predict(vy3,vx3,model1)
ACC,MSE,SCC = evaluations(vy3,p_label)
err1 += MSE
model1 = train(y4,x4,'-s 0 -e 0.000001 -c 0.00005')
p_label,p_accm,p_val = predict(vy4,vx4,model1)
ACC,MSE,SCC = evaluations(vy4,p_label)
err1 += MSE
model1 = train(y5,x5,'-s 0 -e 0.000001 -c 0.00005')
p_label,p_accm,p_val = predict(vy5,vx5,model1)
ACC,MSE,SCC = evaluations(vy5,p_label)
err1 += MSE

err2 = 0
model2 = train(y1,x1,'-s 0 -e 0.000001 -c 0.005')
p_label,p_accm,p_val = predict(vy1,vx1,model2)
ACC,MSE,SCC = evaluations(vy1,p_label)
err2 += MSE
model2 = train(y2,x2,'-s 0 -e 0.000001 -c 0.005')
p_label,p_accm,p_val = predict(vy2,vx2,model2)
ACC,MSE,SCC = evaluations(vy2,p_label)
err2 += MSE
model2 = train(y3,x3,'-s 0 -e 0.000001 -c 0.005')
p_label,p_accm,p_val = predict(vy3,vx3,model2)
ACC,MSE,SCC = evaluations(vy3,p_label)
err2 += MSE
model2 = train(y4,x4,'-s 0 -e 0.000001 -c 0.005')
p_label,p_accm,p_val = predict(vy4,vx4,model2)
ACC,MSE,SCC = evaluations(vy4,p_label)
err2 += MSE
model2 = train(y5,x5,'-s 0 -e 0.000001 -c 0.005')
p_label,p_accm,p_val = predict(vy5,vx5,model2)
ACC,MSE,SCC = evaluations(vy5,p_label)
err2 += MSE

err3 = 0
model3 = train(y1,x1,'-s 0 -e 0.000001 -c 0.5')
p_label,p_accm,p_val = predict(vy1,vx1,model3)
ACC,MSE,SCC = evaluations(vy1,p_label)
err3 += MSE
model3 = train(y2,x2,'-s 0 -e 0.000001 -c 0.5')
p_label,p_accm,p_val = predict(vy2,vx2,model3)
ACC,MSE,SCC = evaluations(vy2,p_label)
err3 += MSE
model3 = train(y3,x3,'-s 0 -e 0.000001 -c 0.5')
p_label,p_accm,p_val = predict(vy3,vx3,model3)
ACC,MSE,SCC = evaluations(vy3,p_label)
err3 += MSE
model3 = train(y4,x4,'-s 0 -e 0.000001 -c 0.5')
p_label,p_accm,p_val = predict(vy4,vx4,model3)
ACC,MSE,SCC = evaluations(vy4,p_label)
err3 += MSE
model3 = train(y5,x5,'-s 0 -e 0.000001 -c 0.5')
p_label,p_accm,p_val = predict(vy5,vx5,model3)
ACC,MSE,SCC = evaluations(vy5,p_label)
err3 += MSE

err4 = 0
model4 = train(y1,x1,'-s 0 -e 0.000001 -c 50')
p_label,p_accm,p_val = predict(vy1,vx1,model4)
ACC,MSE,SCC = evaluations(vy1,p_label)
err4 += MSE
model4 = train(y2,x2,'-s 0 -e 0.000001 -c 50')
p_label,p_accm,p_val = predict(vy2,vx2,model4)
ACC,MSE,SCC = evaluations(vy2,p_label)
err4 += MSE
model4 = train(y3,x3,'-s 0 -e 0.000001 -c 50')
p_label,p_accm,p_val = predict(vy3,vx3,model4)
ACC,MSE,SCC = evaluations(vy3,p_label)
err4 += MSE
model4 = train(y4,x4,'-s 0 -e 0.000001 -c 50')
p_label,p_accm,p_val = predict(vy4,vx4,model4)
ACC,MSE,SCC = evaluations(vy4,p_label)
err4 += MSE
model4 = train(y5,x5,'-s 0 -e 0.000001 -c 50')
p_label,p_accm,p_val = predict(vy5,vx5,model4)
ACC,MSE,SCC = evaluations(vy5,p_label)
err4 += MSE

err5 = 0
model5 = train(y1,x1,'-s 0 -e 0.000001 -c 5000')
p_label,p_accm,p_val = predict(vy1,vx1,model5)
ACC,MSE,SCC = evaluations(vy1,p_label)
err5 += MSE
model5 = train(y2,x2,'-s 0 -e 0.000001 -c 5000')
p_label,p_accm,p_val = predict(vy2,vx2,model5)
ACC,MSE,SCC = evaluations(vy2,p_label)
err5 += MSE
model5 = train(y3,x3,'-s 0 -e 0.000001 -c 5000')
p_label,p_accm,p_val = predict(vy3,vx3,model5)
ACC,MSE,SCC = evaluations(vy3,p_label)
err5 += MSE
model5 = train(y4,x4,'-s 0 -e 0.000001 -c 5000')
p_label,p_accm,p_val = predict(vy4,vx4,model5)
ACC,MSE,SCC = evaluations(vy4,p_label)
err5 += MSE
model5 = train(y5,x5,'-s 0 -e 0.000001 -c 5000')
p_label,p_accm,p_val = predict(vy5,vx5,model5)
ACC,MSE,SCC = evaluations(vy5,p_label)
err5 += MSE

print('err1:',err1)
print('err2:',err2)
print('err3:',err3)
print('err4:',err4)
print('err5:',err5)