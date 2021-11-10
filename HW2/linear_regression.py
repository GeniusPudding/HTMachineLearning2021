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

def cal_sqr_err(w_lin, X, y, size):
    vec = np.matmul(X,w_lin) - y
    E_in_sqr = np.dot(vec, vec)/size
    # print('E_in_sqr:',E_in_sqr)
    return E_in_sqr

def cal_01_err(w_lin, X, y, size):
    E_01_sqr = np.sum(np.abs(np.sign(np.matmul(X,w_lin)) - y)/2)/size
    # print('E_01_sqr:',E_01_sqr)    
    return E_01_sqr

def problem13(times):
    s = 0
    for i in range(times):
        train_data, test_data = gen_dataset()
        w_lin, X, y = lin_reg_alg(train_data)
        s += cal_sqr_err(w_lin, X, y,200)
    print('av_err:',s/times)  

def problem14(times):
    diff = 0
    for i in range(times):
        train_data, test_data = gen_dataset()
        w_lin, X, y = lin_reg_alg(train_data)
        zero_one_in_err = cal_01_err(w_lin, X, y,200)

        test_y = test_data.T[0]
        test_X = test_data.T[1:].T    
        zero_one_out_err = cal_01_err(w_lin, test_X, test_y,5000)
        print("zero_one_in_err, zero_one_out_err:",zero_one_in_err, zero_one_out_err)
        diff += np.abs(zero_one_in_err-zero_one_out_err)
    print('av_01_err:',diff/times)  
def problem15(times):
    E_01_out = 0
    for i in range(times):
        train_data, test_data = gen_dataset()
        w_lin, X, y = lin_reg_alg(train_data)
        test_y = test_data.T[0]
        test_X = test_data.T[1:].T    
        E_01_out += cal_01_err(w_lin, test_X, test_y,5000)
    print('av_01_out_err:',E_01_out/times)  

if __name__ == "__main__":
    #p13:
    # problem13(100)

    #p14:  
    # problem14(1000)#av_01_err: 0.011751800000000017
    
    #p15:  
    problem15(100)

    #p16: 
    # problem16(100)  