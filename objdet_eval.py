# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import matplotlib
#matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools

def transform_bev_to_bev_corners(bev):

    [x, y, w, l, yaw] = bev

    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw # front left
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw 
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw # rear left
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw # rear right
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw # front right
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners

def transform_real_to_bev_corners(label, configs):

    _x, _y, _w, _l, _yaw = label

    # convert from metric into pixel coordinates
    x = (_y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
    y = (_x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
    w = _w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
    l = _l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
    yaw = _yaw
    
    bev = [x, y, w, l, yaw]

    # get object corners within bev image
    bev_corners = transform_bev_to_bev_corners(bev)

    return bev_corners

# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, configs_det, min_iou=0.5):
    
     # find best detection for each valid label 
    true_positives = 0 # no. of correctly detected objects
    center_devs = []
    ious = []
    #print(configs_det)
    #{'lim_x': [0, 50], 'lim_y': [-25, 25], 'lim_z': [-1, 3], 'lim_r': [0, 1.0], 'bev_width': 608, 'bev_height': 608, 'model_path':
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid: # exclude all labels from statistics which are not considered valid
            
            # compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######     
            #######
            print("student task ID_S4_EX1 ")

            ## step 1 : extract the four corners of the current label bounding-box
            print(label)
            box = label.box
            label_box = [box.center_x, box.center_y, box.width, box.length, box.heading]
            label_corners = transform_real_to_bev_corners(label_box, configs_det)
            #print(label_corners)
            polygon_label = Polygon(label_corners)
            #print(polygon_label)
            z0_label = box.center_z-box.height/2
            z1_label = box.center_z+box.height/2
            ## step 2 : loop over all detected objects
            for detection in detections:
                ## step 3 : extract the four corners of the current detection
                print(detection)
                detection_bev = [detection[1], detection[2], detection[5], detection[6], detection[7]]
                detection_corners = transform_bev_to_bev_corners(detection_bev)
                #print(detection_corners)
                
                ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z
                #x = (_y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
                #y = (_x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
                #w = _w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
                #l = _l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
                x_detection = detection[2]/configs_det.bev_height*(configs_det.lim_x[1]-configs_det.lim_x[0])+configs_det.lim_x[0]
                y_detection = detection[1]/configs_det.bev_width*(configs_det.lim_y[1]-configs_det.lim_y[0])+configs_det.lim_y[0]
                z_detection = detection[3]+detection[4]/2
                dist_x = float(abs(box.center_x-x_detection))
                dist_y = float(abs(box.center_y-y_detection))
                dist_z = float(abs(box.center_z-z_detection))

                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                z0_detection = detection[3]
                z1_detection = detection[4]

                zA = max(z0_detection, z0_label)
                zB = min(z1_detection, z1_label)
                # compute the area of intersection rectangle
                interArea = max(0, zB - zA)
                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = z1_detection - z0_detection
                boxBArea = z1_label - z0_label
                unionArea = boxAArea+boxBArea - interArea
                iou_z = interArea/unionArea
                #print(iou_z)

                polygon_detection = Polygon(detection_corners)
                #print(polygon_detection)

                area_i = polygon_label.intersection(polygon_detection).area
                area_u = polygon_label.union(polygon_detection).area
                iou_area = area_i/area_u
                #print(iou_area)
                
                #iou=iou_z*iou_area
                iou=iou_area
                #print(iou)

                ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                if (iou > min_iou):
                    #
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z])
                    true_positives = true_positives+1
                    print("MATCHED")
                    print(matches_lab_det)
                
            #######
            ####### ID_S4_EX1 END #######     
            
        # find best match and compute metrics
        if matches_lab_det:
            best_match = max(matches_lab_det,key=itemgetter(1)) # retrieve entry with max iou in case of multiple candidates   
            ious.append(best_match[0])
            center_devs.append(best_match[1:])


    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    # compute positives and negatives for precision/recall
    
    ## step 1 : compute the total number of positives present in the scene
    all_positives = len(detections)

    ## step 2 : compute the number of false negatives
    false_negatives = labels_valid.sum()-true_positives

    ## step 3 : compute the number of false positives
    false_positives = all_positives-true_positives
    
    #######
    ####### ID_S4_EX2 END #######     
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    print(pos_negs)
    det_performance = [ious, center_devs, pos_negs]
    
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])  
    pos_negs_arr = np.asarray(pos_negs)
    
    ####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')
    
    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    positives = sum(pos_negs_arr[:,0])
    true_positives = sum(pos_negs_arr[:,1])
    false_negatives = sum(pos_negs_arr[:,2])
    false_positives = sum(pos_negs_arr[:,3])
    print("TP = " + str(true_positives) + ", FP = " + str(false_positives) + ", FN = " + str(false_negatives))
    
    ## step 2 : compute precision
    precision = true_positives / (true_positives + false_positives)

    ## step 3 : compute recall 
    recall = true_positives / (true_positives + false_negatives)

    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall))   

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)
    

    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    #std_dev_x = np.std(devs_x)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

