from gen_dataset import *
eta = 0.1
T = 500

def logistic(x):#x: np.array
    return 1 / (1 + np.exp(-x))

def get_gradient_Ein(w, x, y):#w,x,y: np.array
    N = np.size(y)
    grad_Ein = np.zeros(np.size(x[0]))
    for i in range(N):

        # print('y[i],y[i].shape:',y[i],y[i].shape)
        # print('w,w.shape:',w,w.shape)
        # print('x[i],x[i].shape:',x[i],x[i].shape)
        # a = logistic(-y[i]*np.dot(w,x[i]))
        # b = (y[i]*x[i])
        # print('a,a.shape:',a,a.shape)
        # print('b,b.shape:',b,b.shape)
        grad_Ein += (logistic(-y[i]*np.dot(w,x[i]))* (-y[i]*x[i]))
    grad_Ein /= N
    return grad_Ein

def gradient_descent( X, y):
    w_t = np.zeros(np.size(X[0]))
    for i in range(T):
        grad_Ein = get_gradient_Ein(w_t, X, y)
        w_t = w_t - eta* grad_Ein /(np.linalg.norm(grad_Ein))
        # print("w_t gd:",w_t)
    return w_t


def log_reg_alg(X, y):
    w_t = gradient_descent(X,y)
    return w_t
