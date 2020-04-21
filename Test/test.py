#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:49:02 2020

@author: xen
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pyroomacoustics as pra

import random
from room_geometry import L_shape_room, S_shape_room, generatePolygon, shoebox, getAngle, source_position
import pyny3d.geoms as pyny
import math



# L shape room

# wall_len = np.zeros((4,))
# for ii in range(0,4):
#     wall_len[ii] = np.random.uniform(low= 0.5, high = 2)
    
# # Create the 2D L-shaped room from the floor polygon
# pol = np.random.randint(1,4) * np.array([[0,0], [0, wall_len[0]], [wall_len[1],wall_len[0]], [wall_len[1],wall_len[2]], [wall_len[3],wall_len[2]], [wall_len[3],0]]).T

# alpha = np.random.uniform(0.1, 0.9)
# alpha = alpha**2
# room = pra.Room.from_corners(pol, fs=16000, max_order=25, absorption=alpha, air_absorption = True, sigma2_awgn = 0.2)



# # Create the 3D room by extruding the 2D by 3 meters
# room.extrude(3.)

# # Add a source somewhere in the room
# room.add_source([1.5, 1.2, 0.5])

# R = np.array([[3.],   [2.2], [1.6]])

# room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

# room.image_source_model()

# room.plot_rir()

              
              
# room.plot()



# room height
room_height = random.uniform(2.4, 5.5)

# Polygon generator function - set irregularity and spikeyness of polygon
poly_points_floor, poly_points_ceil, cent = L_shape_room(height = room_height)

# Room center (not important so far)
room_centre  = [cent.x, cent.y]

# v_vec = np.array([0.,0.,1.])

poly1 = pyny.Polygon(np.array(poly_points_floor), make_ccw = True)
poly2 = pyny.Polygon(np.array(poly_points_ceil), make_ccw = True)
polyhedron = pyny.Polyhedron.by_two_polygons(poly1, poly2)
polys = polyhedron.polygons

poly_coords = []

for pgns in polys:
    poly_coords.append(poly1)
    
print(poly_coords)
alpha = np.random.uniform(0.1, 0.9)
# m = pra.Material(energy_axbsorption=0.03)
alpha = alpha**2
room = pra.Room.from_corners(np.array(poly1.points)[:,:2].T, fs=16000, max_order=12, absorption = alpha, sigma2_awgn = 0.2)

room.extrude(3.)
room.plot()

# # Add a source somewhere in the room
room.add_source([1.5, 1.2, 0.5])



R = np.array([[3.],   [2.2], [1.6]])

room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

room.image_source_model()

room.plot_rir()

              
