# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:26:44 2020

@author: xen
"""
from shapely.geometry import Polygon
import random
import math
import numpy as np



def L_shape_room(max_side = 4, min_side = 2, height = 2.2):
    a = np.random.uniform(min_side, max_side, 4)
    
    
    polygon = Polygon([(0,0), (0, a[1]), (a[0], a[1]), (a[0], 0)])
    # print(list(polygon.exterior.coords))
    
    one = list(polygon.exterior.coords)
    
    poly2 = Polygon([(0, a[1]), (0, a[1]+a[2]), (a[0]+a[3], a[1]+a[2]), (a[0]+a[3], a[1])])
    
    
    two = list(poly2.exterior.coords)
    
    total = [one[0], two[1], two[2], two[3], one[2], one[3]]
    
    total_floor = []
    total_ceil = []
    centroid = Polygon(total).centroid

    for ii in range(len(total)):
        total_floor.append((*total[ii],0.0))
        total_ceil.append((*total[ii], height))

    # print(total)
    return total_floor, total_ceil, centroid

def S_shape_room(max_side = 3, min_side = 2, height = 2.2):
    
    a = np.random.uniform(min_side, max_side, 6)
    
    
    polygon = Polygon([(0,0), (0, a[1]), (a[0], a[1]), (a[0], 0)])
    # print(list(polygon.exterior.coords))
    
    one = list(polygon.exterior.coords)
    
    poly2 = Polygon([(0, a[1]), (0, a[1]+a[2]), (a[0]+a[3], a[1]+a[2]), (a[0]+a[3], a[1])])
    
    
    two = list(poly2.exterior.coords)
    
    poly3 =  Polygon([(a[0]+a[3], a[1]+a[2]), (a[0]+a[3], a[1]+a[2]+a[4]), (a[0]+a[3]+a[5], a[1]+a[2]+a[4]), (a[0]+a[3]+a[5], a[1])])
    
    three = list(poly3.exterior.coords)
    
    total = [one[0], two[1], two[2], three[1], three[2], three[3], one[2], one[3]]
  
    total_floor = []
    total_ceil = []
    
    centroid = Polygon(total).centroid
    
    for ii in range(len(total)):
        total_floor.append((*total[ii],0.0))
        total_ceil.append((*total[ii], height))

    return total_floor, total_ceil, centroid

def shoebox(max_side = 7, min_side = 3, height = 2.5):
    
    a = np.random.uniform(min_side, max_side, 2)
    
    polygon = Polygon([(0,0), (0, a[1]), (a[0], a[1]), (a[0], 0)])

    total = list(polygon.exterior.coords)
    
    centroid = polygon.centroid
    total_floor = []
    total_ceil = []
    
    for ii in range(len(total)):
        total_floor.append((*total[ii],0.0))
        total_ceil.append((*total[ii], height))

    return total_floor, total_ceil, centroid




def generatePolygon( ctrX, ctrY, z_axis, aveRadius, irregularity, spikeyness, numVerts ) :
    """ Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Random noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order."""
    
    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius
    
    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp
    
    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k
    
    # now generate the points
    points_floor = []
    points_ceil = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points_floor.append( (x,y,0) )
        points_ceil.append( (x,y,z_axis) )

        angle = angle + angleSteps[i]
    
    return points_floor, points_ceil

def clip(x, min, max) :
    if( min > max ) :  return x    
    elif( x < min ) :  return min
    elif( x > max ) :  return max
    else :             return x
    
 
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return abs(ang)
 
#print(getAngle((5, 0), (0, 0), (0, 5)))

def source_position(points, circ_radius):
    
    dom_ind = np.random.randint(0,len(points))
    source_corner =  points[dom_ind]
        
    for ii in range(0,len(points)):
        if sum(points[ii]) == sum(source_corner):
            ind1 = ii - 1
            ind2 = ii + 1
            if ind2 >= len(points):
                ind2 -= len(points)
            
    
    c = points[ind1]  
    a = points[ind2]
    b = source_corner
    
    theta = getAngle(a, b, c)
    theta = np.deg2rad(theta)
    circle_x = source_corner[0]
    circle_y = source_corner[1]
    r = 2
    
    
    x = r * math.cos(theta) + circle_x
    y = r * math.sin(theta) + circle_y

    return x,y