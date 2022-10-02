from telnetlib import OUTMRK
from matplotlib.projections import projection_registry
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation
import sys
from os.path import join, isfile

def get_mean(x):
    return np.sum(x,axis=1)/x[0].shape[0]

def get_std(x):
    return np.std(x,axis =1)

def get_normalized_data(x):
    x_mean = get_mean(x)
    # print(type(x_mean))
    x_dif = np.copy(x)
    x_dif[0]=x_dif[0]-x_mean[0]
    x_dif[1]=x_dif[1]-x_mean[1]
    # print(x_dif)
    sigma = get_std(x)
    x_dif[0]=x_dif[0]/sigma[0]
    x_dif[1]=x_dif[1]/sigma[1]
    return x_dif

def heuristic(theta, x):
    return 1 / (1 + np.exp(-np.dot(theta,x)))

def hessian(theta, x):
    h = np.zeros((x.shape[0], x.shape[0]))
    # print(x[:,1])
    for i in range(x[0].shape[0]):
        x_cur = x[:,i]
        # print(x_cur)
        e = np.exp(-np.dot(theta, x_cur))
        h -= e / (1 + e)**2 * np.outer(x_cur, x_cur)
    return h

def derivative(theta, x, y):
    der = np.zeros(x.shape[0])
    m = x[0].shape[0]
    for i in range(m):
        x_cur, y_cur = x[:,i], y[i]
        e = np.exp(-np.dot(theta, x_cur))
        der += (heuristic(theta,x_cur)*(y_cur*e-(1-y_cur)))*x_cur
    return der

def logistic_regression(x, y):
    n = x.shape[0]
    theta = np.zeros(n)
    stopping_fac = 1e-7
    cur_diff=1
    # print(hessian(theta,x))
    while cur_diff>stopping_fac:
        der= derivative(theta, x, y)
        h = hessian(theta, x)
        # print(np.linalg.inv(h))
        diff = np.matmul(np.linalg.inv(h), der)
        theta -= diff
        cur_diff= np.linalg.norm(diff)
    return theta

def test(theta,test_data,x_mean,x_std):
    x = np.array(np.genfromtxt(join(test_data, 'X.csv'), delimiter=',')).T
    x[0]=x[0]-x_mean[0]
    x[1]=x[1]-x_mean[1]
    x[0]=x[0]/x_std[0]
    x[1]=x[1]/x_std[1]
    # print(x_mean)
    # print(x_std)
    # print(theta)
    m = x[0].shape[0]
    y= theta[0]+(theta[1]*x[0])+theta[2]*x[1]
    outtxt = open('result_3.txt', mode='w')
    for i in range(m):
        if(1/(1+np.exp(-y[i]))>0.5):
            outtxt.write(str(1)+'\n')
        else:
            outtxt.write("0\n")
    outtxt.close()
    return

def main():
    data_dir = sys.argv[1]
    test_data = sys.argv[2]
    out_dir = "output"
    file = open(join(data_dir,'X.csv'))
    lines = file.readlines()
    data =[]
    for line in lines:
        data.append(np.fromstring(line, dtype=float, sep=','))
    data = np.array(data)
    data = data.T
    x = np.array(data)
    y = np.array(np.genfromtxt(join(data_dir,"Y.csv")))
    # print(y)
    x_mean = get_mean(x)
    x_std = get_std(x)

    x_norm = get_normalized_data(x)

    
    m = x[0].shape[0]
    x_norm = np.vstack((np.ones(m),x_norm))
    fin_theta = logistic_regression(x_norm,y)
    outtxt_file = open(join(out_dir,"theta(a).txt"),mode='w')
    outtxt_file.write("value of theta obtained is: "+ str(fin_theta))
    outtxt_file.close()
    ##part 2
    x_y0=[]
    x_y1=[]
    # print(get_mean(x))
    for i in range(m):
        if y[i]==1:
            x_y1.append(x[:,i])
        else:
            x_y0.append(x[:,i])

    x_y0=np.array(x_y0).T
    x_y1=np.array(x_y1).T
    figb = plt.figure()
    axb = plt.axes()
    axb.scatter(x_y0[0],x_y0[1],c='red')
    axb.scatter(x_y1[0],x_y1[1],c='blue')
    
    x_line =np.array([-2.5,2])

    y_line = -(x_line*fin_theta[1]+fin_theta[0])/fin_theta[2]
    
    axb.plot(x_line*x_std[0]+x_mean[0],y_line*x_std[1]+x_mean[1])
    figb.savefig(join(out_dir, 'regression_plot.png'))
    plt.close(figb)
    ##output
    test(fin_theta,test_data,get_mean(x),get_std(x))

main()
