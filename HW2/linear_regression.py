from gen_dataset import *

def lin_reg_alg(train_data):
    y = train_data.T[0]
    X = train_data.T[1:].T
    # x_pinv = []
    try:
        x_pinv = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)
    except np.linalg.LinAlgError as e:#no inverse of XTX
        x_pinv = np.linalg.pinv(X)
    w_lin = np.matmul(x_pinv , y )
    # print('w_lin:',w_lin)
    return w_lin, X, y

def cal_sqr_err(w_lin, X, y):
    vec = np.matmul(X,w_lin) - y
    E_in_sqr = np.dot(vec, vec)/200
    # print('E_in_sqr:',E_in_sqr)
    return E_in_sqr

def cal_01_err(w_lin, X, y):
    vec = np.sign(np.matmul(X,w_lin))
    E_01_sqr = np.dot(vec, vec)/200
    return E_01_sqr

def problem13():
    s = 0
    for i in range(100):
        train_data, test_data = gen_dataset()
        w_lin, X, y = lin_reg_alg(train_data)
        s += cal_sqr_err(w_lin, X, y)
    print('av_err:',s/100)  

def problem14():
    zero_one_in_err = 0
    zero_one_out_err = 0
    for i in range(100):
        train_data, test_data = gen_dataset()
        w_lin, X, y = lin_reg_alg(train_data)
        zero_one_in_err += cal_01_err(w_lin, X, y)

        test_y = test_data.T[0]
        test_X = test_data.T[1:].T    
        zero_one_out_err += cal_01_err(w_lin, test_X, test_y)
    print('av_01_err:',zero_one_in_err/100,zero_one_out_err/100)  

if __name__ == "__main__":
    #p13:
    # problem13()

    #p14:  
    problem14()