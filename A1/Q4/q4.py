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
    #  (type(x_mean))
    x_dif = np.copy(x)
    x_dif[0]=x_dif[0]-x_mean[0]
    x_dif[1]=x_dif[1]-x_mean[1]
    # print(x_dif)
    sigma = get_std(x)
    x_dif[0]=x_dif[0]/sigma[0]
    x_dif[1]=x_dif[1]/sigma[1]
    return x_dif

    
def get_parameters_same_sigma(x, y):

    phi=0
    x_mean = np.array([0., 0.])
    x_sigma =np.array([[0.,0.],[0.,0.]])
    tot =np.array([0,0])
    m = y.shape[0]

    tot[1] = np.sum(y)
    tot[0] = m - tot[1]

    phi = tot[1] / m

    x_mean = np.array([np.sum(np.array([x[:,j] for j in range(m) if y[j] == i]), axis=0) / tot[i] for i in range(2)])

    for i in range(m):
        x_sigma=np.add(x_sigma,np.outer((x[:,i]-x_mean[y[i]]),(x[:,i]-x_mean[y[i]])))
    x_sigma=x_sigma/m


    return phi, x_mean, x_sigma

def get_parameters_diff_sigma(x, y):

    phi=0
    x_mean = np.array([0., 0.])
    x_sigma =np.array([[[0.,0.],[0.,0.]],[[0.,0.],[0.,0.]]])
    tot =np.array([0.,0.])
    m = y.shape[0]

    tot[1] = np.sum(y)
    tot[0] = m - tot[1]

    phi = tot[1] / m

    x_mean = np.array([np.sum(np.array([x[:,j] for j in range(m) if y[j] == i]), axis=0) / tot[i] for i in range(2)])

    for i in range(m):
        k = y[i]
        x_sigma[k]= x_sigma[k]+ np.outer((x[:,i]-x_mean[k]),(x[:,i]-x_mean[k]))

    x_sigma[0]=x_sigma[0]/(tot[0])
    x_sigma[1]=x_sigma[1]/(tot[1])

    return phi, x_mean, x_sigma

def test(test_data,x_mean,x_std,phi,mu,sigma):
    x = np.array(np.genfromtxt(join(test_data, 'X.csv'), delimiter=',')).T
    x[0]=x[0]-x_mean[0]
    x[1]=x[1]-x_mean[1]
    x[0]=x[0]/x_std[0]
    x[1]=x[1]/x_std[1]
    m = x[0].shape[0]
    mu0sinv= np.matmul(mu[0],np.linalg.inv(sigma[0]))
    mu1sinv= np.matmul(mu[1],np.linalg.inv(sigma[1]))
    # print(mu0sinv)
    constant= np.log(phi/(1-phi))  + np.log(np.linalg.det(sigma[0])/np.linalg.det(sigma[1]))/2 +(np.dot(mu0sinv,mu[0])-np.dot(mu1sinv,mu[1]))/2
    invsigmadif = np.linalg.inv(sigma[0])- np.linalg.inv(sigma[1])    
    
    outtxt = open('result_4.txt', mode='w')
    for i in range(m):
        X= x[0][i]
        Y = x[1][i]
        Z=(((X**2)*invsigmadif[0,0])+((X*Y)*(invsigmadif[0,1]+invsigmadif[1,0]))+((Y**2)*invsigmadif[1,1]))/2 +(mu1sinv-mu0sinv)[0]*X + (mu1sinv-mu0sinv)[1]*Y + constant
        if Z>0 :
            outtxt.write('Canada\n')
        else:
           outtxt.write('Alaska\n') 
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
    # print(x)
    y = np.array([1 if yi == 'Canada' else 0 for yi in np.loadtxt(join(data_dir,"Y.csv"),dtype=str)])
    # print(x)
    m = x[0].shape[0]
    x_norm = get_normalized_data(x)
    phi_a,x_mean_a,x_sigma_a=get_parameters_same_sigma(x_norm,y)
    outtxt_file = open(join(out_dir,"4a.txt"),mode='w')
    outtxt_file.write("mu0 = " + str(x_mean_a[0])+'\n')
    outtxt_file.write("mu1 = " + str(x_mean_a[1])+'\n')
    outtxt_file.write("sigma = \n" + str(x_sigma_a)+'\n')
    outtxt_file.close()
    ##part 2 and part 3
    x_y0b=[]
    x_y1b=[]
    for i in range(m):
        if y[i]==1:
            x_y1b.append(x[:,i])
        else:
            x_y0b.append(x[:,i])

    x_y0b=np.array(x_y0b).T
    x_y1b=np.array(x_y1b).T
    figb = plt.figure()
    ax = plt.axes()
    ax.scatter(x_y0b[0],x_y0b[1],c='red')
    ax.scatter(x_y1b[0],x_y1b[1],c='blue')

    sigma_inverse = np.linalg.inv(x_sigma_a)
    theta = np.array([0., 0., 0.])

    theta[1:] = np.matmul( sigma_inverse,np.transpose( np.array([x_mean_a[1] - x_mean_a[0]]))).T
    theta[0] = np.log(phi_a / (1 - phi_a))
    cur_mean0 = np.array([x_mean_a[0]])
    theta[0] +=  np.matmul(np.matmul(cur_mean0, sigma_inverse), cur_mean0.T)/2
    cur_mean1 = np.array([x_mean_a[1]])
    theta[0] -=  np.matmul(np.matmul(cur_mean1, sigma_inverse), cur_mean1.T)/2

    x_line =np.array([-2,2])

    y_line = -(x_line*theta[1]+theta[0])/theta[2]
    
    ax.plot(x_line*get_std(x)[0]+get_mean(x)[0],y_line*get_std(x)[1]+get_mean(x)[1],c='black')
    figb.savefig(join(out_dir, 'linearseparator.png'))
    ## part 4
    
    X, Y = np.meshgrid(np.linspace(-3, 3, 1000), np.linspace(-3, 3, 1000))
    phi_d,x_mean_d,x_sigma_d=get_parameters_diff_sigma(x_norm,y)
    # print("4: ")
    # print(phi_d)
    # print(x_mean_d)
    # print(x_sigma_d)
    outtxt_file = open(join(out_dir,"4d.txt"),mode='w')
    outtxt_file.write("mu0 = " + str(x_mean_d[0])+'\n')
    outtxt_file.write("mu1 = " + str(x_mean_d[1])+'\n')
    outtxt_file.write("sigma0 = \n" + str(x_sigma_d[0])+'\n')
    outtxt_file.write("sigma1 = \n" + str(x_sigma_d[1])+'\n')
    outtxt_file.close()
    ##part 5
    mu0sinv= np.matmul(x_mean_d[0],np.linalg.inv(x_sigma_d[0]))
    mu1sinv= np.matmul(x_mean_d[1],np.linalg.inv(x_sigma_d[1]))
    # print(mu0sinv)
    constant= np.log(phi_d/(1-phi_d))  + np.log(np.linalg.det(x_sigma_d[0])/np.linalg.det(x_sigma_d[1]))/2 +(np.dot(mu0sinv,x_mean_d[0])-np.dot(mu1sinv,x_mean_d[1]))/2
    invsigmadif = np.linalg.inv(x_sigma_d[0])- np.linalg.inv(x_sigma_d[1])
    Z = (((X**2)*invsigmadif[0,0])+((X*Y)*(invsigmadif[0,1]+invsigmadif[1,0]))+((Y**2)*invsigmadif[1,1]))/2 +(mu1sinv-mu0sinv)[0]*X + (mu1sinv-mu0sinv)[1]*Y + constant
    ax.contour(X * get_std(x)[0] + get_mean(x)[0], Y * get_std(x)[1] +get_mean(x)[1], Z, 0)
    figb.savefig(join(out_dir, 'quadraticseparator.png'))
    plt.close(figb)    
    ##output
    test(test_data,get_mean(x),get_std(x),phi_d,x_mean_d,x_sigma_d)
main()