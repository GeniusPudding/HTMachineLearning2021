from gen_dataset import *

def lin_reg_alg():

    train_data, test_data = gen_dataset()

    y = train_data.T[0]
    X = train_data.T[1:].T
    # x_pinv = []
    try:
        x_pinv = np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T)
    except np.linalg.LinAlgError as e:#no inverse of XTX
        x_pinv = np.linalg.pinv(X)
    w_lin = np.matmul(x_pinv , y )
    # print('w_lin:',w_lin)
    vec = np.matmul(X,w_lin) - y
    E_in_sqr = np.dot(vec, vec)/200
    # print('E_in_sqr:',E_in_sqr)
    return E_in_sqr

if __name__ == "__main__":
    s = 0
    for i in range(100):
        s += lin_reg_alg()
    print('av_err:',s/100)    