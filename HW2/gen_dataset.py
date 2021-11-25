from __future__ import print_function
import numpy as np

mean_pos = [2, 3]
cov_pos = [[0.6,0],[0,0.6]]
mean_neg = [0, 4]
cov_neg = [[0.4,0],[0,0.4]]

mean_out = [6, 0]
cov_out = [[0.3,0],[0,0.1]]
def gen_dataset():
    np.random.seed(np.random.randint(1126))
    y = np.random.randint(2, size=5200) #0->-1
    y_x = []
    for i in y:
        if i:
            x = np.random.multivariate_normal(mean_pos, cov_pos, 1)
            x1,x2 = x[0][0],x[0][1]
            y_x.append([1,1,x1,x2])
        else:
            x = np.random.multivariate_normal(mean_neg, cov_neg, 1)
            x1,x2 = x[0][0],x[0][1]
            y_x.append([-1,1,x1,x2])  

    y_x = np.array(y_x)      
    train_data = y_x[:200] 
    test_data = y_x[200:]
    # print("train_data.shape:",train_data.shape)
    # print("test_data.shape:",test_data.shape)
    # print("train_data:",train_data)

    return train_data, test_data 
def gen_dataset_outlier():
    np.random.seed(np.random.randint(1126))
    y = np.random.randint(2, size=5200) #0->-1
    y_x = []
    for i in y:
        if i:
            x = np.random.multivariate_normal(mean_pos, cov_pos, 1)
            x1,x2 = x[0][0],x[0][1]
            y_x.append([1,1,x1,x2])
        else:
            x = np.random.multivariate_normal(mean_neg, cov_neg, 1)
            x1,x2 = x[0][0],x[0][1]
            y_x.append([-1,1,x1,x2])  

    y_x = np.array(y_x)      
    train_data = y_x[:200] 
    test_data = y_x[200:]

    outlier = np.ones(20)
    y_x = []
    for i in outlier:
        x = np.random.multivariate_normal(mean_out, cov_out, 1)
        x1,x2 = x[0][0],x[0][1]
        y_x.append([1,1,x1,x2])    
    y_x = np.array(y_x)     
    # print(train_data.shape,y_x.shape)
    train_data = np.concatenate((train_data,y_x))

    return train_data, test_data     
def gen_X_y(data):#data: train_data, test_data
    return data.T[1:].T, data.T[0]

if __name__ == "__main__":
    gen_dataset()