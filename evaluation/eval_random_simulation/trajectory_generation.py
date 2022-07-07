import numpy as np
import math
import random

def is_saw(x, y, n):
    '''
    Checks if walk of length n is self-avoiding
    :param (x,y)(list,list): walk of length n
    :param n(int): length of the walk
    :return: True if the walk is self-avoiding
    '''
    # creating a set removes duplicates, so it suffices to check the size of the set
    return n+1 == len(set(zip(x, y)))

def in_map(x, y):
    '''
    Checks if walk is in map
    :param (x,y)(list,list): the walk
    :return judge: True if the walk is in map
    '''
    judge = False
    x, y = np.array(x), np.array(y)
    if (abs(x) < 9.5).all() and (abs(y) < 9.5).all():
        judge = True
    return judge

def myopic_saw(n, x_init, y_init):
    '''
    Tries to generate a SAW of length n using myopic algorithm
    :param n(int): length of walk
    :param (x_init, y_init)(float, float): initial coordinate of saw
    :return (x,y)(list,list): the walk (length <= n)
    '''
    theta = [i * math.pi / 2 for i in range(0, 4)]
    r = 5    # original one: 0.75
    x, y = [x_init], [y_init]
    positions = set([(x_init,y_init)])    # positions is a set(no same element) that stores all sites visited
    for i in range(n):
        t = random.choice(theta)
        x_new = x[-1] + r * round(math.cos(t), 2)
        y_new = y[-1] + r * round(math.sin(t), 2)
        if (x_new, y_new) not in positions:
            x.append(x_new)
            y.append(y_new)
            positions.add((x_new, y_new))
        else:
            continue
    return x, y

def dimer(n, x_init, y_init):
    '''
    Generates a SAW by dimerization
    :param n(int): length of walk
    :param (x_init, y_init)(float, float): initial coordinate of saw
    :return (x_concat,y_concat)(list,list): walk of length n
    '''
    if n <= 3:
        x, y = myopic_saw(n, x_init, y_init)
        return x, y
    else:
        not_saw = True
        while not_saw:
            x_1, y_1 = dimer(n//2, x_init, y_init)
            x_2, y_2 = dimer(n - n//2, x_init, y_init)
            x_2 = [(x_1[-1] - x_1[0] + x) for x in x_2]
            y_2 = [(y_1[-1] - y_1[0] + y) for y in y_2]
            x_concat, y_concat = x_1 + x_2[1:], y_1 + y_2[1:]
            if is_saw(x_concat, y_concat, n) and in_map(x_concat, y_concat):
                not_saw = False
        return x_concat, y_concat

def interpolation(x, y, n_trajectory):
    '''
    Generates continous trajectory by whittaker-shannon interpolation formula
    :param (x,y)(list,list): discrete self-avoiding walk
    :param n_trajectory(int): length of continous trajectory
    :return (x_contin,y_contin)(list,list): continous trajectory
    '''
    interval = 1
    N = len(x)
    t = np.linspace(0, (N - 1) * interval, n_trajectory)
    x_contin, y_contin = 0, 0
    for n in range(N):
        x_contin += x[n] * np.sinc(t / interval - n)
        y_contin += y[n] * np.sinc(t / interval - n)
    return x_contin, y_contin

def gen_saw_track(x_init, y_init, trajectory_n=1100, n_discre=69):
    '''
    Generates continous self-avoiding trajectory
    :param (x_init, y_init)(float, float): initial coordinate of saw
    :param n_discre(int): length of discrete self-avoiding walk
    :param n_trajectory(int): length of continous trajectory
    :return trajectory(list): continous trajectory
    '''
    x, y = dimer(n_discre, x_init, y_init)
    x_contin, y_contin = interpolation(x, y, trajectory_n)
    trajectory = list(zip(x_contin, y_contin))
    return trajectory

def gen_rose_track(x_init, y_init, trajectory_n=1200):
    x, y = [], []
    for i in range(trajectory_n):
        k = i * np.pi / 400
        r = 5 * np.sin(4 * k)
        x.append(0.5 * x_init + r * np.cos(k))
        y.append(0.5 * y_init + r * np.sin(k))
    trajectory = list(zip(x, y))
    return trajectory

def gen_spiral_track(x_init, y_init, trajectory_n=1200):
    x, y = [], []
    for i in range(trajectory_n):
        k = i * math.pi / 90
        r = 0.12 * k
        x.append(0.5 * x_init + r * math.cos(k))
        y.append(0.5 * y_init + r * math.sin(k))
    trajectory = list(zip(x, y))
    return trajectory