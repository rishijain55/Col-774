from matplotlib.projections import projection_registry
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation
from matplotlib.animation import PillowWriter
import sys
from os.path import join, isfile
def get_mean(x):
    return np.sum(x,axis=0)/x.shape[0]

def get_std(x):
    return np.std(x,axis =0)

def get_normalized_data(x):
    x_mean = get_mean(x)
    x_dif = x-x_mean
    sigma = get_std(x)
    return x_dif/sigma

def grad_desc(x,y,eta):
    learning_rate =eta
    stopping_fac= 1e-15
    norm_x = get_normalized_data(x)
    m = norm_x.shape[0]
    norm_x = np.vstack((np.ones(m), norm_x))
    theta = np.array([0., 0.])
    iterations = 0
    cur_cost =1
    prev_cost =-1
    theta_vs_cost=[]
    while (abs(cur_cost-prev_cost)>stopping_fac):
        prev_cost=cur_cost
        dif=(y-np.matmul(theta,norm_x))
        cur_cost=np.sum(dif ** 2) / (2 * m)
        theta_vs_cost.append([theta[0],theta[1],cur_cost])
        
        theta = theta + (learning_rate/m)*(np.sum(dif*norm_x,axis=1))
        iterations+=1

    return theta,iterations,np.array(theta_vs_cost),learning_rate,stopping_fac

def test(theta,test_data,mean,std):
    x = np.array(np.genfromtxt(join(test_data, 'X.csv')))
    norm_x = (x-mean)/std
    m = x.shape[0]
    y= theta[0]+(theta[1]*norm_x)
    outtxt = open('result_1.txt', mode='w')
    for i in range(m):
        outtxt.write(str(y[i])+'\n')
    outtxt.close()
    return
  
def main():
    data_dir = sys.argv[1]
    test_data = sys.argv[2]
    out_dir = "output"
    x = np.array(np.genfromtxt(join(data_dir, 'X.csv')))
    y = np.array(np.genfromtxt(join(data_dir, 'Y.csv')))
    x_norm = get_normalized_data(x)

    fin_theta,iterations,theta_vs_cost,learning_rate,stopping_factor=grad_desc(x,y,0.001)
    outtxt = open(join(out_dir, '1a.txt'), mode='w')
    outtxt.write('learning_rate = ' + str(learning_rate) + '\n')
    outtxt.write('stopping_factor = ' + str(stopping_factor) + '\n')
    outtxt.write('theta_0 = ' + str(fin_theta[0]) + '\n')
    outtxt.write('theta_1 = ' + str(fin_theta[1]) + '\n')
    outtxt.write('total_iterations = ' + str(iterations) + '\n')
    outtxt.close()

    #part b

    x_line = np.array([-2,5])
    figb = plt.figure()
    axb = plt.axes()
    axb.scatter(x,y)
    axb.set_xlabel('Acidity')
    axb.set_ylabel('Density')
    axb.plot(x_line*get_std(x)+get_mean(x),fin_theta[0]+fin_theta[1]*x_line)
    figb.savefig(join(out_dir, 'regression_plot.png'))
    plt.close(figb)

    (X, Y), Z = np.meshgrid(np.linspace(-0.1, 2, 1000), np.linspace(-0.5, 0.5, 1000)), 0
    for i in range(x.shape[0]):
        Z += ((X + Y * x_norm[i]) - y[i]) ** 2
    Z /= 2 * x_norm.shape[0]
    figc = plt.figure()
    axc = plt.axes(projection='3d')
    axc.plot_surface(X, Y, Z)

    x_data_c=[]
    y_data_c=[]
    z_data_c=[]
    axc.set_xlabel('Theta_0')
    axc.set_ylabel('Theta_1')
    axc.set_zlabel('Error_function')
    # x_data_c.extend([theta_vs_cost[i][0] for i in range(0,theta_vs_cost.shape[0],200)])
    # y_data_c.extend([theta_vs_cost[i][1] for i in range(0,theta_vs_cost.shape[0],200)])
    # z_data_c.extend([theta_vs_cost[i][2] for i in range(0,theta_vs_cost.shape[0],200)])
    # print(theta_vs_cost.shape[0])
    # print(x_data_c)
    # y_data_c.extend(theta_vs_cost[:,1])
    # z_data_c.extend(theta_vs_cost[:,2])
    # plotc = axc.plot(x_data_c,y_data_c,z_data_c)
    # axc.scatter(x_data_c,y_data_c,z_data_c,c='red')
    # def animate(i):
    #     x_data_c.append(theta_vs_cost[i][0])
    #     y_data_c.append(theta_vs_cost[i][1])
    #     z_data_c.append(theta_vs_cost[i][2])
    #     plotc = axc.plot(x_data_c,y_data_c,z_data_c)
    #     return plotc


    # anim1c = FuncAnimation.FuncAnimation(figc, animate, theta_vs_cost.shape[0], interval=10, blit=True)
    # animate(0)
    # plt.show()
    # writervideo = PillowWriter(fps=60)
    # anim1c.save('error_function_surface.gif', writer="imagemagick")
    

    figc.savefig(join(out_dir, 'error_function_surface.png'))
    plt.close(figc)

    #part d
    figd = plt.figure()
    axd = plt.axes()
    axd.contour(X, Y, Z,100)

    x_data_d=[]
    y_data_d=[]
    axd.set_xlabel('Theta_0')
    axd.set_ylabel('Theta_1')
    # x_data_d.extend([theta_vs_cost[i][0] for i in range(0,theta_vs_cost.shape[0],200)])
    # y_data_d.extend([theta_vs_cost[i][1] for i in range(0,theta_vs_cost.shape[0],200)])
    # plotd = axd.plot(x_data_d,y_data_d)
    # axd.scatter(x_data_d,y_data_d,c='red')
    #for animation
    
    def animated(i):
        # print(i)
        x_data_d.append(theta_vs_cost[i][0])
        y_data_d.append(theta_vs_cost[i][1])
        plotd = axd.plot(x_data_d,y_data_d)
        return plotd


    animd = FuncAnimation.FuncAnimation(figd, animated, theta_vs_cost.shape[0], interval=2, blit=True)
    animated(0)
    plt.show()
    
    figd.savefig(join(out_dir, 'contour.png'))
    plt.close(figd)
    #part e
    theta_vs_cost=[]
    fin_theta,iterations,theta_vs_cost,learning_rate,stopping_factor=grad_desc(x,y,0.001)
    fige = plt.figure()
    axe = plt.axes()
    axe.contour(X, Y, Z,100)

    axe.set_xlabel('Theta_0')
    axe.set_ylabel('Theta_1')
    x_data_e=[]
    y_data_e=[]
    # print(theta_vs_cost.shape[0])
    x_data_e.extend([theta_vs_cost[i][0] for i in range(0,theta_vs_cost.shape[0],5)])
    y_data_e.extend([theta_vs_cost[i][1] for i in range(0,theta_vs_cost.shape[0],5)])
    # plotd = axe.plot(x_data_d,y_data_d)
    axe.scatter(x_data_e,y_data_e,c='red')

    fige.savefig(join(out_dir, 'contour_for_eta_0.001.png'))
    plt.close(fige)
    #eta 0,025
    theta_vs_cost=[]
    fin_theta,iterations,theta_vs_cost,learning_rate,stopping_factor=grad_desc(x,y,0.025)
    fige = plt.figure()
    axe = plt.axes()
    axe.contour(X, Y, Z,100)

    axe.set_xlabel('Theta_0')
    axe.set_ylabel('Theta_1')
    x_data_e=[]
    y_data_e=[]
    # print(theta_vs_cost.shape[0])
    x_data_e.extend([theta_vs_cost[i][0] for i in range(0,theta_vs_cost.shape[0],5)])
    y_data_e.extend([theta_vs_cost[i][1] for i in range(0,theta_vs_cost.shape[0],5)])
    # plotd = axe.plot(x_data_e,y_data_e)
    axe.scatter(x_data_e,y_data_e,c='red')
    fige.savefig(join(out_dir, 'contour_for_eta_0.025.png'))
    plt.close(fige)
    # eta = 0.1
    theta_vs_cost=[]
    fin_theta,iterations,theta_vs_cost,learning_rate,stopping_factor=grad_desc(x,y,0.1)
    fige3 = plt.figure()
    axe = plt.axes()
    axe.contour(X, Y, Z,100)

    axe.set_xlabel('Theta_0')
    axe.set_ylabel('Theta_1')
    x_data_e=[]
    y_data_e=[]
    # print(theta_vs_cost.shape[0])
    x_data_e.extend([theta_vs_cost[i][0] for i in range(0,theta_vs_cost.shape[0],5)])
    y_data_e.extend([theta_vs_cost[i][1] for i in range(0,theta_vs_cost.shape[0],5)])
    # plotd = axe.plot(x_data_e,y_data_e)
    axe.scatter(x_data_e,y_data_e,c='red')
    fige3.savefig(join(out_dir, 'contour_for_eta_0.1.png'))
    plt.close(fige3)
    #output
    fin_theta,iterations,theta_vs_cost,learning_rate,stopping_factor=grad_desc(x,y,0.001)
    fin_theta= np.array([0.9966191017815839,0.0013401946761656392])
    test(fin_theta,test_data,get_mean(x),get_std(x))
main()