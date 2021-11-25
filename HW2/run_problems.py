from linear_regression import *
from logistic_regression import *

def problem13(times):
    s = 0
    for i in range(times):
        train_data, test_data = gen_dataset()
        X, y = gen_X_y(train_data)
        w_lin = lin_reg_alg(X, y)
        s += cal_sqr_err(w_lin, X, y,200)
    print('av_err:',s/times)  

def problem14(times):
    diff = 0
    for i in range(times):
        train_data, test_data = gen_dataset()
        X, y = gen_X_y(train_data)
        w_lin = lin_reg_alg(X, y)
        zero_one_in_err = cal_01_err(w_lin, X, y,200)

        test_y = test_data.T[0]
        test_X = test_data.T[1:].T    
        zero_one_out_err = cal_01_err(w_lin, test_X, test_y,5000)
        print("zero_one_in_err, zero_one_out_err:",zero_one_in_err, zero_one_out_err)
        diff += np.abs(zero_one_in_err-zero_one_out_err)
    print('av_01_err:',diff/times)  
def problem15(times):
    E_01_out_A = 0
    E_01_out_B = 0
    for i in range(times):
        train_data, test_data = gen_dataset()
        test_y = test_data.T[0]
        test_X = test_data.T[1:].T    
        X, y = gen_X_y(train_data)

        w_lin = lin_reg_alg(X, y)
        E_01_out_A += cal_01_err(w_lin, test_X, test_y,5000)

        w_t = log_reg_alg(X, y)
        E_01_out_B += cal_01_err(w_t, test_X, test_y,5000)
        # print('01_out_err:',E_01_out_A,E_01_out_B)  

    print('av_01_out_err:',E_01_out_A/times,E_01_out_B/times)  

def problem16(times):
    E_01_out_A = 0
    E_01_out_B = 0
    for i in range(times):
        train_data, test_data = gen_dataset_outlier()
        test_y = test_data.T[0]
        test_X = test_data.T[1:].T    
        X, y = gen_X_y(train_data)

        w_lin = lin_reg_alg(X, y)
        E_01_out_A += cal_01_err(w_lin, test_X, test_y,5000)

        w_t = log_reg_alg(X, y)
        E_01_out_B += cal_01_err(w_t, test_X, test_y,5000)
        # print('01_out_err:',E_01_out_A,E_01_out_B)  

    print('av_01_out_err:',E_01_out_A/times,E_01_out_B/times)  



if __name__ == "__main__":
    #p13:
    problem13(100)#av_err: 0.2803833863514997

    #p14:  
    problem14(100)#problem14(1000) av_01_err: 0.011751800000000017
    
    # #p15:  
    problem15(100)#av_01_out_err: 0.057588000000000014 0.058557999999999985

    #p16: 
    problem16(100)  #av_01_out_err: 0.08943200000000001 0.05737799999999999  