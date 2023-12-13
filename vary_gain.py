import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

def S(theta):
    return np.array([[np.cos(theta),0],[np.sin(theta),0],[0,1]])

def Sbd(thetas):
    matrices = [S(theta) for theta in thetas]
    return block_diag(*matrices)

def l2_norm(vector):
    return np.sqrt(np.sum(np.square(vector)))

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def sat(u, u_max):
    if abs(u) > u_max:
        if u < 0:
            return -u_max
        else: return u_max
    else: return u

num_robot = 6
num_state = 3
num_input = 2

max_v = 2.0
max_w = 13.3

# adjacency_matrix
a = np.array([
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0]
        ])

dt= 0.01
f = int(1/dt)
T = 10
epsilon = 0.1

K_p_max = 6
K_h_max = 6
dK_p = 0.05
dK_h = 0.05

array_K_p = np.arange(0, K_p_max+dK_p, dK_p)
array_K_h = np.arange(0, K_h_max+dK_h, dK_h)
array_t   = np.zeros((len(array_K_p), len(array_K_h)))
n_K_p = 0
for K_p in array_K_p:
    n_K_h = 0
    for K_h in array_K_h:
        x = np.zeros(num_state*num_robot) # robots' states (1x3*num_robot)
        p = np.zeros(2*num_robot) # robots' position (1x2*num_robot)
        u = np.zeros(num_input*num_robot) # robots' inputs (1x2*num_robot)

        # initial position
        x[0] = -4.0; x[1] = 2.0; x[2] = 1.57
        x[3] = 0.0 ; x[4] = -4.0;x[5] = 1.0
        x[6] = 4.5 ; x[7] = 2.1; x[8] = 3.1
        x[9] = 4.0 ; x[10] =-4.0;x[11] = 2.0
        x[12] = -1.0;x[13]=-3.4; x[14] = 1.7
        x[15] = -4.0;x[16] =-4.0;x[17] = 1.2

        p = x.reshape(num_robot, num_state)[:, :2].flatten()

        traj_x = np.zeros((T*f, num_state*num_robot))
        traj_p = np.zeros((T*f, 2*num_robot))

        traj_x[0] = x
        traj_p[0] = p

        # initial input u
        for i in range(num_robot):
            where = np.where(a[i]==1)[0]
            ep = np.zeros((2,1))
            for j in where:
                ep += np.array([[p[j*num_input]-p[i*num_input]],
                                [p[j*num_input+1]-p[i*num_input+1]]])
            u[i*num_input] = sat(K_p*l2_norm(ep), max_v)
            u[i*num_input+1] = sat(K_h*wrap_to_pi(np.arctan2(ep[1],ep[0])-x[2::num_state][i]),
                                max_w)

        t = 0
        for k in range(1, T*f):
            # undate state
            x_new = x + np.matmul(Sbd(x[2::num_state]),u)*dt
            #traj_x[k] = x_new
            x = x_new

            p = x_new.reshape(num_robot, num_state)[:, :2].flatten()
            #traj_p[k] = p

            sum_ep = np.zeros((2,1))
            
            # update input u
            for i in range(num_robot):
                where = np.where(a[i]==1)[0]
                ep = np.zeros((2,1))
                for j in where:
                    ep += np.array([[p[j*num_input]-p[i*num_input]],
                                    [p[j*num_input+1]-p[i*num_input+1]]])
                u[i*num_input] = sat(K_p*l2_norm(ep), max_v)
                u[i*num_input+1] = sat(K_h*wrap_to_pi(np.arctan2(ep[1],ep[0])-x[2::num_state][i]),
                                    max_w)
                sum_ep += ep
            
            if (l2_norm(sum_ep/num_robot) >= epsilon):
                t+=dt
            else: break

        array_t[n_K_p, n_K_h] = t
        n_K_h += 1
    n_K_p += 1
    print(n_K_p)

K_p_grid, K_h_grid = np.meshgrid(array_K_p, array_K_h)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(K_p_grid, K_h_grid, array_t.T, cmap='viridis')

ax.set_xlabel('K_p')
ax.set_ylabel('K_h')
ax.set_zlabel('Time (t)')
ax.set_title('3D Plot of Time vs K_p and K_h')

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.show()