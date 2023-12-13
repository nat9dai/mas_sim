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

dt= 0.01
f = int(1/dt)
T = 15
epsilon = 0.05

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
V = np.zeros(T*f)

traj_x[0] = x
traj_p[0] = p

# adjacency_matrix
a = np.array([
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0]
        ])
       
K_p = 5.0
K_h = 0.5
max_v = 2.0
max_w = 13.3

# initial input u
for i in range(num_robot):
    where = np.where(a[i]==1)[0]
    ep = np.zeros((2,1))
    for j in where:
        # epi
        ep += np.array([[p[j*num_input]-p[i*num_input]],
                        [p[j*num_input+1]-p[i*num_input+1]]])
    u[i*num_input] = sat(K_p*l2_norm(ep), max_v)
    u[i*num_input+1] = sat(K_h*wrap_to_pi(np.arctan2(ep[1],ep[0])-x[2::num_state][i]),
                           max_w)
    V[0] += ep.T @ ep # dot product for 2D array

t = 0
for k in range(1, T*f):
    # undate state
    x_new = x + np.matmul(Sbd(x[2::num_state]),u)*dt
    traj_x[k] = x_new
    x = x_new

    p = x_new.reshape(num_robot, num_state)[:, :2].flatten()
    traj_p[k] = p

    sum_ep = np.zeros((2,1))
    
    # update input u
    for i in range(num_robot):
        where = np.where(a[i]==1)[0]
        ep = np.zeros((2,1))
        for j in where:
            # epi
            ep += np.array([[p[j*num_input]-p[i*num_input]],
                            [p[j*num_input+1]-p[i*num_input+1]]])
        u[i*num_input] = sat(K_p*l2_norm(ep), max_v)
        u[i*num_input+1] = sat(K_h*wrap_to_pi(np.arctan2(ep[1],ep[0])-x[2::num_state][i]),
                            max_w)
        sum_ep += ep
        V[k] += ep.T @ ep
    
    #if (l2_norm((sum_ep/num_robot) >= epsilon) and (l2_norm(u[0::2]) >= 0.1) and (l2_norm(u[1::2]) >= 1.0)):
    if (l2_norm((sum_ep/num_robot) >= epsilon)):
        t+=dt

# Splitting traj_p into p_x and p_y for each robot
p_x = traj_p[:, 0::2]  # even columns are x positions
p_y = traj_p[:, 1::2]  # odd columns are y positions

print(t)
print(np.mean(V[-100:]))

time = np.linspace(0, T, T*f)

# Create a figure and a set of subplots
plt.figure(figsize=(12, 10))

# Subplot for p_x vs t
plt.subplot(3, 1, 1)  # 2 rows, 1 column, first subplot
for i in range(num_robot):
    plt.plot(time, p_x[:, i], label=f'Robot {i+1} p_x')
plt.title('p_x vs Time')
plt.xlabel('Time (seconds)')
plt.ylabel('p_x')
plt.legend()

# Subplot for p_y vs t
plt.subplot(3, 1, 2)  # 2 rows, 1 column, second subplot
for i in range(num_robot):
    plt.plot(time, p_y[:, i], label=f'Robot {i+1} p_y')
plt.title('p_y vs Time')
plt.xlabel('Time (seconds)')
plt.ylabel('p_y')
plt.legend()

# Subplot for V(x) vs t
plt.subplot(3, 1, 3)  # 2 rows, 1 column, second subplot
plt.plot(time, V/2, label='V(x)')
plt.title('V(x) vs Time')
plt.xlabel('Time (seconds)')
plt.ylabel('V(x)')
plt.legend()

plt.tight_layout()
plt.show()