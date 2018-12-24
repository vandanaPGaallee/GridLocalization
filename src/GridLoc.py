#!/usr/bin/env python
import rospy
import math
import random
import time
from math import radians

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import rosbag
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import numpy as np
from tf.transformations import euler_from_quaternion

iterations = 0
threshold = 0.001
disc_size = 30
theta = 360/disc_size
occ_grid = np.zeros((35, 35, theta)) 
orig_grid = []
occ_grid[11][27][int(200.52/disc_size)] = 1
cell_size = 20
sigma = cell_size/2
sigma_r = disc_size/2
tags = np.array([[125, 525],[125, 325], [125, 125], [425, 125], [425, 325], [425, 525]])
fi = open("/home/first/catkin_ws/src/lab4/src/trajectory.txt", "w")


def ComputeNorm(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def getCM(x, c_size):
    return (x)*c_size + c_size/2

def ComputeSMParams(x, y, z, x1, y1):
    r = math.degrees(math.atan2((y1-y),(x1-x)))
    trans = math.sqrt((x1 - x)**2 + (y1 -y)**2)
    r1 = roundDegree(r - z)
    return trans, r1
    
def SensorModel(trans, rot, tagnum):
    global occ_grid, tags, iterations
    # print("Tags")
    # print(tags[tagnum][0],tags[tagnum][1],tagnum)
    for i in range(35):
        for j in range(35):
            for k in range(theta):
                c_trans, c_r1 = ComputeSMParams(getCM(i, cell_size), getCM(j, cell_size), getCM(k, disc_size) - 180, tags[tagnum][0], tags[tagnum][1])
                step1 = ComputeNorm(c_trans, trans, sigma) * ComputeNorm(c_r1, rot, sigma_r)
                step2 = step1 * occ_grid[i][j][k]
                occ_grid[i][j][k] = step2
    occ_grid = occ_grid / np.sum(occ_grid)
    iterations +=1
    print("Result = " + str(iterations))
    # print(tags[tagnum][0]/cell_size, tags[tagnum][1]/cell_size)
    x,y,z = np.unravel_index(occ_grid.argmax(), occ_grid.shape)
    max_prob = np.amax(occ_grid)
    fi.write("Iteration = " + str(iterations) + "\n")
    fi.write("Index = " + str(x+1) + "," + str(y+1) + "," + str(z+1) + "   " + "Prob = " +  str(max_prob) + "\n\n")
    print(x+1,y+1,z+1)
    print(max_prob)

def roundDegree(r):
    if(r < -180):
        r = r + 360
    elif(r > 180):
        r = r - 360
    return r

def ComputeParams(x, y, z, x1, y1, z1):
    r = math.degrees(math.atan2((y1-y), (x1-x)))
    trans = math.sqrt((x1 - x)**2 + (y1 -y)**2)
    r1 = roundDegree(r - z)
    r2 = roundDegree(z1 - z - r1)
    return trans, r1, r2

def ComputeProbabilities(x, y, z, r1, trans, r2):
    global orig_grid, occ_grid
    for i in range(35):
        for j in range(35):
            for k in range(theta):
                c_trans, c_r1, c_r2 = ComputeParams(getCM(x, cell_size), getCM(y, cell_size), getCM(z, disc_size) - 180, getCM(i, cell_size), getCM(j, cell_size), getCM(k, disc_size) - 180)
                step1 = ComputeNorm(c_trans, trans, sigma) * ComputeNorm(c_r1, r1, sigma_r) * ComputeNorm(c_r2, r2, sigma_r)
                step2 = step1 * occ_grid[x][y][z]
                orig_grid[i][j][k] += step2

def ComputePrior(r1, trans, r2):
    global orig_grid, occ_grid
    orig_grid = np.zeros((35, 35, theta)) 
    for i in range(35):
        for j in range(35):
            for k in range(theta):
                if(occ_grid[i][j][k] >= threshold):
                    ComputeProbabilities(i, j, k, r1, trans, r2)    
    occ_grid = orig_grid

if __name__ == '__main__':
    rospy.init_node('lab4', anonymous=True)
    bag = rosbag.Bag('/home/first/catkin_ws/src/lab4/bag/grid.bag')
    for topic, msg, t in bag.read_messages(topics=['Movements', 'Observations']):
        if( topic == "Movements"):
            r1 = euler_from_quaternion([ msg.rotation1.x, msg.rotation1.y, msg.rotation1.z, msg.rotation1.w])[2]
            r2 = euler_from_quaternion([ msg.rotation2.x, msg.rotation2.y, msg.rotation2.z, msg.rotation2.w])[2]
            # print(math.degrees(r1) + 180,math.degrees(r2)+180)
            # print(math.degrees(r1),math.degrees(r2))
            ComputePrior(math.degrees(r1), msg.translation * 100, math.degrees(r2)) 
        elif(topic == "Observations"):
            rot = euler_from_quaternion([ msg.bearing.x, msg.bearing.y, msg.bearing.z, msg.bearing.w])[2]
            SensorModel(msg.range * 100, math.degrees(rot), msg.tagNum) 
    bag.close()  
    fi.close()                                                    

