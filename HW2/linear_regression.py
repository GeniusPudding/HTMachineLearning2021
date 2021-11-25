from gen_dataset import *

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

def cal_sqr_err(w_lin, X, y, size):
    vec = np.matmul(X,w_lin) - y
    E_in_sqr = np.dot(vec, vec)/size
    # print('E_in_sqr:',E_in_sqr)
    return E_in_sqr

def cal_01_err(w_lin, X, y, size):
    E_01_sqr = np.sum(np.abs(np.sign(np.matmul(X,w_lin)) - y)/2)/size
    # print('E_01_sqr:',E_01_sqr)    
    return E_01_sqr

