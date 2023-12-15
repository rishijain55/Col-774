from matplotlib.projections import projection_registry
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation
import sys
from os.path import join, isfile

def grad_desc(x,y,b):
    eta= 0.01
    stopping_fac= 0.001
    # print(x)
    m = x.shape[1]
    # print(m)
    theta = np.array([0., 0.,0.])
    iterations = 0
    checkpoint_iteration=1000
    cur_error = 1

    prev_cum_error =-1
    cum_error=0
    thetas=[]
    i =0
    while (True):
        # print(i)
        # print(iterations)
        thetas.append(theta)
        x_cur_batch = x[:,i:i+b]
        y_cur_batch = y[i: i+b]
        dif=(y_cur_batch-np.matmul(theta,x_cur_batch))
        cur_error=np.sum(dif ** 2) / (2 * b)
        theta = theta + (eta*(np.sum(dif*x_cur_batch,axis=1))/b)         
        iterations+=1
        cum_error+= cur_error
        if(iterations%checkpoint_iteration==0):
            cum_error/=checkpoint_iteration
            if(abs(cum_error-prev_cum_error)<stopping_fac):
                break
            prev_cum_error=cum_error
            cum_error=0
        i= (i+b)%m
    return theta,iterations,thetas

def test(theta,test_data):
    x = np.array(np.genfromtxt(join(test_data, 'X.csv'), delimiter=',')).T
    # print(x)
    m = x[0].shape[0]
    y= theta[0]+(theta[1]*x[0])+theta[2]*x[1]
    outtxt = open('result_2.txt', mode='w')
    for i in range(m):
        outtxt.write(str(y[i])+'\n')
    outtxt.close()
    return

def main():

    test_dir = sys.argv[1]
    # out_dir ="output"
    # ##part 1
    # x_mean = np.array([3., -1.])
    # x_sigma = np.array([2., 2.])
    # m = 1000000
    # x = [None]* 3
    # x[0] = np.ones(m)
    # for i in range(2):
    #     x[i + 1] = np.random.normal(x_mean[i], x_sigma[i], m)
    # theta = np.array([3., 1., 2.])
    # y = np.matmul(theta,x)
    # y = y + np.random.normal(0,np.sqrt(2),m)
    # out_file = open(join(out_dir,"q2a.txt"), 'w')
    # out_file.write('x:\n' + str(x) + '\n')
    # out_file.write('y:\n' + str(y) + '\n')
    # out_file.close()

    # ##part 2
    # x = np.array(x)
    # y = np.array(y)
    # fin_thetas=[[3,1,2]]
    # iterations=[]
    # all_thetas=[]
    # out_txt = open(join(out_dir,"q2bc.txt"), 'w')
    # for b in [1,100,10000,1000000]:
    #     fin_theta,iteration,thetas=grad_desc(x,y,b)
    #     out_txt.write("for r = "+str(b)+" theta is: "+"["+str(fin_theta[0])+", "+str(fin_theta[1])+", "+str(fin_theta[2])+"]"+'\n')
    #     out_txt.write("for r = "+str(b)+" iterations are: "+str(iteration)+'\n')
    #     all_thetas.append(thetas)
    #     fin_thetas.append(fin_theta)
    #     iterations.append(iteration)
    #     # print(fin_theta)
    # ##part 3
    # data = []
    # file = open('q2test.csv')
    # lines = file.readlines()[1:]
    # for line in lines:
    #     data.append(np.fromstring(line, dtype=float, sep=','))
    # data = np.array(data)
    # data = data.T
    # x_c = np.array(data[:2])
    # # print(x_c)
    # y_c = np.array(data[2])
    # # print(y_c)
    # x_c = np.vstack((np.ones((x_c.shape[1])),x_c))
    # # print(x_c)
    # errors=[]
    # m_example = y_c.shape[0]
    # turn =0
    # for t in fin_thetas:
    #     dif = ( y_c - np.matmul(t,x_c) )
    #     error_t = np.sum((dif ** 2))/(2*m_example)
    #     if turn ==0:
    #         out_txt.write("for original model error is: "+str(error_t)+'\n')
    #     else:
    #         out_txt.write("for theta ="+ "["+str(t[0])+", "+str(t[1])+", "+str(t[2])+"]"+" error is: "+str(error_t)+'\n')
    #     errors.append(error_t)
    #     turn+=1
    # out_txt.close()

    # ##part 4
    # # print(all_thetas[0][0])
    # for t in range(4):
    #     figd = plt.figure()
    #     axd = plt.axes(projection='3d')

    #     theta0_d=[]
    #     theta1_d=[]
    #     theta2_d=[]
    #     axd.set_xlabel('Theta_0')
    #     axd.set_ylabel('Theta_1')
    #     axd.set_zlabel('Theta_2')
    #     axd.set_xlim(0.0,4.0)
    #     axd.set_ylim(0.0,1.5)
        
    #     axd.set_zlim(0.0,2.5)
    #     # print(len(all_thetas))
    #     for i in range(len(all_thetas[t])):
    #         theta0_d.append(all_thetas[t][i][0])
    #         theta1_d.append(all_thetas[t][i][1])
    #         theta2_d.append(all_thetas[t][i][2])

    #     plot2d = axd.plot(theta0_d,theta1_d,theta2_d)
    #     outpath ="theta_variation"+str(t)+".png"

    #     figd.savefig(join(out_dir,outpath))
    #     plt.close(figd)
        
    #     # def animate(i):
    #     #     theta0_d.append(all_thetas[t][i][0])
    #     #     theta1_d.append(all_thetas[t][i][1])
    #     #     theta2_d.append(all_thetas[t][i][2])
    #     #     # print("thetas")
    #     #     # print(theta0_d)
    #     #     plot2d = axd.plot(theta0_d,theta1_d,theta2_d)
    #     #     return plot2d


    #     # anim1c = FuncAnimation.FuncAnimation(figd, animate, len(all_thetas[t]), interval=20, blit=True)
    #     # animate(0)
    #     # plt.show()

    #     # plt.close(figd)

    #     ##output
    # error_min = 10000
    # minind=1
    # for i in range(4):
    #     if(errors[i+1]<error_min):
    #         errormin = errors[i+1]
    #         minind = i+1
    # # print(minind)
    best_theta = np.array([3.0093846132073776, 0.9986790266007011, 2.000214536840162])
    test( best_theta,test_dir)

main()