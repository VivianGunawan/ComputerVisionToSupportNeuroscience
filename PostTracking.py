import pandas as pd
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import deeplabcut
import cv2
import math
from collections import namedtuple
from scipy.spatial import distance

deeplabcut.utils.plotting.PlottingResults

#put video paths here!
videopaths =["pursuit-14_days_post_TN_C57-5-1.mp4", "pursuit-14_days_post_TN_C57-5-2.mp4", "pursuit-14_days_post_TN_C57-5-3.mp4", "pursuit-14_days_post_TN_C57-5-4.mp4","pursuit-14_days_post_TN_C57-5-5.mp4"]
#CSV save paths/filenames here
savenames =["results/csv/collision1.csv", "results/csv/collision2.csv", "results/csv/collision3.csv", "results/csv/collision4.csv", "results/csv/collision5.csv"]

for i in range (len(videopaths)):
    video = videopaths[i]
    savename = savenames[i]
    # Vivian's (MICKEY)
    DLCscorer_Mickey='DLC_resnet50_MGH-mouseApr6shuffle1_220500'
    dataname_Mickey = 'MGH-mouse-Vivian-Gunawan-2020-04-06/videos/'  + str(Path(video).stem) + DLCscorer_Mickey + '.h5'
    # Beatrice's (JERRY)
    DLCscorer_Jerry='DLC_resnet50_MGH-mouseApr23shuffle1_316500'
    dataname_Jerry = 'MGH-mouse-Beatrice-Tanaga-2020-04-23/videos/'  + str(Path(video).stem) + DLCscorer_Jerry + '.h5'
    #loading output of DLC
    dfMickey = pd.read_hdf(os.path.join(dataname_Mickey))
    dfJerry = pd.read_hdf(os.path.join(dataname_Jerry))


    print(dfJerry[DLCscorer_Jerry]["snout"].iloc[5]["x"])

    #using metric of euclidean distance from left ear to right ear as gauge whether the rats are in direct contact
    leftx = dfJerry[DLCscorer_Jerry]["leftear"]["x"][1]
    lefty = dfJerry[DLCscorer_Jerry]["leftear"]["y"][1]
    rightx = dfJerry[DLCscorer_Jerry]["rightear"]["x"][1]
    righty = dfJerry[DLCscorer_Jerry]["rightear"]["y"][1]

    contact_thresh = math.sqrt((leftx -rightx)**2 + (lefty - righty)**2)
    # print(contact_thresh)
    # absolute distance between any point is less than 

    bpt='snout'
    Jvel =calc_distance_between_points_in_a_vector_2d(np.vstack([dfJerry[DLCscorer_Jerry][bpt]['x'].values.flatten(), dfJerry[DLCscorer_Jerry][bpt]['y'].values.flatten()]).T)

    fps=30 # frame rate of camera in those experiments
    time=np.arange(len(Jvel))*1./fps #notice the units of vel are relative pixel distance [per time step]

    # store in other variables:
    Jxsnout=dfJerry[DLCscorer_Jerry][bpt]['x'].values
    Jysnout=dfJerry[DLCscorer_Jerry][bpt]['y'].values
    Jvsnout=Jvel

    Mvel =calc_distance_between_points_in_a_vector_2d(np.vstack([dfMickey[DLCscorer_Mickey][bpt]['x'].values.flatten(), dfMickey[DLCscorer_Mickey][bpt]['y'].values.flatten()]).T)

    Mxsnout=dfMickey[DLCscorer_Mickey][bpt]['x'].values
    Mysnout=dfMickey[DLCscorer_Mickey][bpt]['y'].values
    Mvsnout=Mvel

    bp_tracking_Jerry = np.array((Jxsnout, Jysnout, Jvsnout))
    bp_tracking_Mickey = np.array((Mxsnout, Mysnout, Mvsnout))
    flag = False

    collision_record =[]
    for i in range(dfJerry.shape[0]):
        m1 = getpoints(dfJerry[DLCscorer_Jerry].iloc[i])
        m2 = getpoints(dfMickey[DLCscorer_Mickey].iloc[i])

        distances = get_distances(m1,m2)
        
        if(detect_collisions(distances, contact_thresh) and flag == False):
            flag =True
            collision_roi = get_roi_of_collision(distances, m1,m2)
            entry = most_recent_entry(dfJerry,dfMickey,collision_roi,i)
            if entry == 1:
                collision_record += [[i,1]]
            elif entry ==2:
                collision_record +=[[i,2]]
            else:
                flag = False
        elif (detect_collisions(distances, contact_thresh) and flag == True):
            continue
        
        elif (flag ==True and not detect_collisions(distances, contact_thresh)):
            flag = False


    df=  pd.DataFrame(collision_record, columns = ["Frame","Approacher"])
    df.to_csv(savename)
    extract_frames(video, savename)

# helper methods created for this file.

def euclidean_distance(pt1, pt2):
    (x1,y1) = pt1
    (x2,y2) = pt2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

# Creates a numpy array(4x4) of the distance from each tracked feature to another
def get_distances(m1, m2):
    distances= np.zeros(shape=(4,4))
    for i in range(4):
        for j in range(4):
            distances[i][j]= euclidean_distance(m1[i],m2[j])
    return distances

#checks if any of the featurs are within collision distance of each other
def detect_collisions(distances, threshold):
    for k in distances:
        for j in k:
            if j <threshold:
                return True
    return False

#Gets centroid of a mouse
def get_centroid(m):
    (s,l,r,t) = m
    x = (s[0] + l[0] + r[0] + t[0])//4
    y = (s[1] + l[1] + r[1] + t[1])//4
    return(x,y)

#creates ROI of collision, we identify the center of the 2 points where the mice are closes, then we make the distance  
def get_roi_of_collision(distances, m1 ,m2):
    # print(distances)
    ind = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    pt1 = m1[ind[0]]
    pt2 = m2[ind[1]]
    collisioncenter = ((pt1[0] + pt2[0])//2, (pt1[1] +pt2[1])//2)
    m1_dist = euclidean_distance(get_centroid(m1), collisioncenter)
    m2_dist = euclidean_distance(get_centroid(m2), collisioncenter)
    dimension = min(m1_dist, m2_dist)
    (x,y,w,h) = (collisioncenter[0]-(dimension//2), collisioncenter[1]-(dimension//2), dimension, dimension)
    return (x,y,w,h)

def getpoints(tablerow):
    return ((tablerow["snout"]["x"],tablerow["snout"]["y"]), (tablerow["leftear"]["x"], tablerow["leftear"]["y"]), (tablerow["rightear"]["x"],tablerow["rightear"]["y"]), (tablerow["tailbase"]["x"], tablerow["tailbase"]["y"]))

def in_roi(m, roi):
    for i in m:
        x1,y1 = i
        (x,y,w,h) = roi
        if ( x1> x and x1< x+w and y1> y and y1< y+h):
            return True
    return False

def most_recent_entry(dfJerry, dfMickey, roi, i):
    for j in range(i, 0,-1):
        if(in_roi(m1, collision_roi)and in_roi(m2,collision_roi)):
            continue
        elif (in_roi(m1, collision_roi) == True and in_roi(m2, collision_roi)==False):
            return 2
        else:
            return 1
    return 0



   
def extract_frames(video, csv_file):
    """
    Method to extract desired frames based off collision record, the second if else clause can be tweaked
    params:
        video : string name/path of the video

        csv_file : string name/path of csv with the records of the frames with interactions
    output:
        saves frames that are notated with the region of collision and 
    """
    cap = cv2.VideoCapture(video)
    framecount = 0
    collisions = pd.read_csv(csv_file)
    nextframe = collisions.iloc[0]
    entrycount= 0
    # print(nextframe)
    # For use if we want to extract a particular frame and x amount of frames before
    # targetframe = PUT TARGET FRAME HERE

    while(True):
        # print("loopstarted")
        # Capture frame-by-frame
        ret, frame = cap.read()
        if (framecount == nextframe["Frame"]):
            
            m1 = getpoints(dfJerry[DLCscorer_Jerry].iloc[nextframe["Frame"]])
            m2 = getpoints(dfMickey[DLCscorer_Mickey].iloc[nextframe["Frame"]])
            
            distances = get_distances(m1,m2)
            collision_roi = get_roi_of_collision(distances, m1,m2)
            # print(collision_roi)
            for i in range(len(m1)):
                cv2.circle(frame, (int(m1[i][0]),int(m1[i][1])), 5, (255,0,0),1)
                cv2.circle(frame, (int(m2[i][0]),int(m2[i][1])), 5, (0,0,255),1)
            cv2.rectangle(frame,(int(collision_roi[0]),int(collision_roi[1])),(int(collision_roi[0]+collision_roi[2]),int(collision_roi[1]+collision_roi[3])),(0,255,0), 1)
            cv2.imwrite("results/frames/frame"+str(framecount)+".jpg", frame )
            entrycount +=1
            nextframe = collisions.iloc[entrycount]
        # since video is 30fps we want to extract 3 secondswe extract 90 frames leading up to the target frame.
        # UNCOMMENT THIS BLOCK IF YOU WANT TO EXTRACT MORE FRAMES THAN THE COLLISION FRAME 
        # elif (nextframe["Frame"] - framecount < 90):
        #     m1 = getpoints(dfJerry[DLCscorer_Jerry].iloc[framecount])
        #     m2 = getpoints(dfMickey[DLCscorer_Mickey].iloc[framecount])
            
        #     distances = get_distances(m1,m2)
        #     for i in range(len(m1)):
        #         cv2.circle(frame, (int(m1[i][0]),int(m1[i][1])), 5, (255,0,0),2)
        #         cv2.circle(frame, (int(m2[i][0]),int(m2[i][1])), 5, (0,0,255),2)
        #     cv2.imwrite("results/precollision/preframe"+str(framecount)+".jpg",frame)



#helper methods taken from a mix of sources.

def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def Histogram(vector,color,bins):
    dvector=np.diff(vector)
    dvector=dvector[np.isfinite(dvector)]
    plt.hist(dvector,color=color,histtype='step',bins=bins)

def PlottingResults(Dataframe,bodyparts2plot,alphavalue=.2,pcutoff=.5,colormap='jet',fs=(4,3)):
    ''' Plots poses vs time; pose x vs pose y; histogram of differences and likelihoods.'''
    plt.figure(figsize=fs)
    colors = get_cmap(len(bodyparts2plot),name = colormap)
    scorer=Dataframe.columns.get_level_values(0)[0] #you can read out the header to get the scorer name!

    for bpindex, bp in enumerate(bodyparts2plot):
        Index=Dataframe[scorer][bp]['likelihood'].values > pcutoff
        plt.plot(Dataframe[scorer][bp]['x'].values[Index],Dataframe[scorer][bp]['y'].values[Index],'.',color=colors(bpindex),alpha=alphavalue)

    plt.gca().invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot)-1))
    sm._A = []
    cbar = plt.colorbar(sm,ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    #plt.savefig(os.path.join(tmpfolder,"trajectory"+suffix))
    plt.figure(figsize=fs)
    Time=np.arange(np.size(Dataframe[scorer][bodyparts2plot[0]]['x'].values))

    for bpindex, bp in enumerate(bodyparts2plot):
        Index=Dataframe[scorer][bp]['likelihood'].values > pcutoff
        plt.plot(Time[Index],Dataframe[scorer][bp]['x'].values[Index],'--',color=colors(bpindex),alpha=alphavalue)
        plt.plot(Time[Index],Dataframe[scorer][bp]['y'].values[Index],'-',color=colors(bpindex),alpha=alphavalue)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot)-1))
    sm._A = []
    cbar = plt.colorbar(sm,ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.xlabel('Frame index')
    plt.ylabel('X and y-position in pixels')
    #plt.savefig(os.path.join(tmpfolder,"plot"+suffix))

    plt.figure(figsize=fs)
    for bpindex, bp in enumerate(bodyparts2plot):
        Index=Dataframe[scorer][bp]['likelihood'].values > pcutoff
        plt.plot(Time,Dataframe[scorer][bp]['likelihood'].values,'-',color=colors(bpindex),alpha=alphavalue)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot)-1))
    sm._A = []
    cbar = plt.colorbar(sm,ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.xlabel('Frame index')
    plt.ylabel('likelihood')

    #plt.savefig(os.path.join(tmpfolder,"plot-likelihood"+suffix))

    plt.figure(figsize=fs)
    bins=np.linspace(0,np.amax(Dataframe.max()),100)

    for bpindex, bp in enumerate(bodyparts2plot):
        Index=Dataframe[scorer][bp]['likelihood'].values < pcutoff
        X=Dataframe[scorer][bp]['x'].values
        X[Index]=np.nan
        Histogram(X,colors(bpindex),bins)
        Y=Dataframe[scorer][bp]['x'].values
        Y[Index]=np.nan
        Histogram(Y,colors(bpindex),bins)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=plt.Normalize(vmin=0, vmax=len(bodyparts2plot)-1))
    sm._A = []
    cbar = plt.colorbar(sm,ticks=range(len(bodyparts2plot)))
    cbar.set_ticklabels(bodyparts2plot)
    plt.ylabel('Count')
    plt.xlabel('DeltaX and DeltaY')
    
    #plt.savefig(os.path.join(tmpfolder,"hist"+suffix))


"""
    Functions to extract time spent by the mouse in each of a list of user defined ROIS 
    Contributed by Federico Claudi
    https://github.com/FedeClaudi
    Example usage:
    rois        -->  a dictionary with name and position of each roi
    tracking    -->  a pandas dataframe with X,Y,Velocity for each bodypart
    bodyparts   -->  a list with the name of all the bodyparts
    
    -----------------------------------------------------------------------------------
    results = {}
    for bp in bodyparts:
        bp_tracking = np.array((tracking.bp.x.values, tracking.bp.y.values, tracking.bp.Velocity.values))
        res = get_timeinrois_stats(bp_tracking, roi, fps=30)
        results[bp] = res
    
    ------------------------------------------------------------------------------------
    if Velocity is not know, it can be calculated using "calc_distance_between_points_in_a_vector_2d":
        vel = calc_distance_between_points_in_a_vector_2d(np.array(tracking.bp.x.values, tracking.bp.y.values))
    which returns a 1d vector with the velocity in pixels/frame [effectively the number pixels a tracked point moved
    from one frame to the next]
"""

def calc_distance_between_points_in_a_vector_2d(v1):
    '''calc_distance_between_points_in_a_vector_2d [for each consecutive pair of points, p1-p2, in a vector, get euclidian distance]
    This function can be used to calculate the velocity in pixel/frame from tracking data (X,Y coordinates)
    
    Arguments:
        v1 {[np.array]} -- [2d array, X,Y position at various timepoints]
    
    Raises:
        ValueError
    
    Returns:
        [np.array] -- [1d array with distance at each timepoint]
    >>> v1 = [0, 10, 25, 50, 100]
    >>> d = calc_distance_between_points_in_a_vector_2d(v1)
    '''
    # Check data format
    if isinstance(v1, dict) or not np.any(v1) or v1 is None:
            raise ValueError(
                'Feature not implemented: cant handle with data format passed to this function')

    # If pandas series were passed, try to get numpy arrays
    try:
        v1, v2 = v1.values, v2.values
    except:  # all good
        pass
    # loop over each pair of points and extract distances
    dist = []
    for n, pos in enumerate(v1):
        # Get a pair of points
        if n == 0:  # get the position at time 0, velocity is 0
            p0 = pos
            dist.append(0)
        else:
            p1 = pos  # get position at current frame

            # Calc distance
            dist.append(np.abs(distance.euclidean(p0, p1)))

            # Prepare for next iteration, current position becomes the old one and repeat
            p0 = p1

    return np.array(dist)


def get_roi_at_each_frame(bp_data, rois, check_inroi):
    """
    Given position data for a bodypart and the position of a list of rois, this function calculates which roi is
    the closest to the bodypart at each frame
    :param bp_data: numpy array: [nframes, 3] -> X,Y,Speed position of bodypart at each frame
                    [as extracted by DeepLabCut] --> df.bodypart.values. 
    :param rois: dictionary with the position of each roi. The position is stored in a named tuple with the location of
                    two points defyining the roi: topleft(X,Y) and bottomright(X,Y).
    :param check_inroi: boolean, default True. If true only counts frames in which the tracked point is inside of a ROI. 
                Otherwise at each frame it counts the closest ROI.
    :return: tuple, closest roi to the bodypart at each frame
    """

    def sort_roi_points(roi):
        return np.sort([roi.topleft[0], roi.bottomright[0]]), np.sort([roi.topleft[1], roi.bottomright[1]])

    if not isinstance(rois, dict): raise ValueError('rois locations should be passed as a dictionary')

    if not isinstance(bp_data, np.ndarray):
        if not isinstance(bp_data, tuple): raise ValueError('Unrecognised data format for bp tracking data')
        else:
            pos = np.zeros((len(bp_data.x), 2))
            pos[:, 0], pos[:, 1] = bp_data.x, bp_data.y
            bp_data = pos

    # Get the center of each roi
    centers = []
    for points in rois.values():
        center_x = (points.topleft[0] + points.bottomright[0]) / 2
        center_y = (points.topleft[1] + points.bottomright[1]) / 2
        center = np.asarray([center_x, center_y])
        centers.append(center)

    roi_names = list(rois.keys())

    # Calc distance to each roi for each frame
    data_length = bp_data.shape[0]
    distances = np.zeros((data_length, len(centers)))
    for idx, center in enumerate(centers):
        cnt = np.tile(center, data_length).reshape((data_length, 2))
        
        dist = np.hypot(np.subtract(cnt[:, 0], bp_data[:, 0]), np.subtract(cnt[:, 1], bp_data[:, 1]))
        distances[:, idx] = dist

    # Get which roi is closest at each frame
    sel_rois = np.argmin(distances, 1)
    roi_at_each_frame = tuple([roi_names[x] for x in sel_rois])

    # Check if the tracked point is actually in the closest ROI
    if not check_inroi: 
        cleaned_rois = []
        for i, roi in enumerate(roi_at_each_frame):
            x,y = bp_data[i, 0], bp_data[i, 1]
            X, Y = sort_roi_points(rois[roi]) # get x,y coordinates of roi points
            if not X[0] <= x <= X[1] or not Y[0] <= y <= Y[1]:
                cleaned_rois.append('none')
            else:
                cleaned_rois.append(roi)
        return cleaned_rois
    else:
        print("Warning: you've set check_inroi=False, so data reflect which ROI is closest even if tracked point is not in any given ROI.")
        return roi_at_each_frame


def get_timeinrois_stats(data, rois, fps=None, returndf=False, check_inroi=True):
    """
    Quantify number of times the animal enters a roi, cumulative number of frames spend there, cumulative time in seconds
    spent in the roi and average velocity while in the roi.
    In which roi the mouse is at a given frame is determined with --> get_roi_at_each_frame()
    Quantify the ammount of time in each  roi and the avg stay in each roi
    :param data: trackind data is a numpy array with shape (n_frames, 3) with data for X,Y position and Speed. If [n_frames, 2]
                array is passed, speed is calculated automatically.
    :param rois: dictionary with the position of each roi. The position is stored in a named tuple with the location of
                two points defyining the roi: topleft(X,Y) and bottomright(X,Y).
    :param fps: framerate at which video was acquired
    :param returndf: boolean, default False. If true data are returned as a DataFrame instead of dict.
    :param check_inroi: boolean, default True. If true only counts frames in which the tracked point is inside of a ROI. 
                Otherwise at each frame it counts the closest ROI.
    :return: dictionary or dataframe
    # Testing
    >>> position = namedtuple('position', ['topleft', 'bottomright'])
    >>> rois = {'middle': position((300, 400), (500, 800))}
    >>> data = np.zeros((23188, 3))
    >>> res = get_timeinrois_stats(data, rois, fps=30)
    >>> print(res)
    """

    def get_indexes(lst, match):
        return np.asarray([i for i, x in enumerate(lst) if x == match])

    # Check arguments
    if data.shape[1] == 2:  # only X and Y tracking data passed, calculate speed
        speed = calc_distance_between_points_in_a_vector_2d(data)
        data = np.hstack((data, speed.reshape((len(speed), 1))))

    elif data.shape[1] != 3:
        raise ValueError("Tracking data should be passed as either an Nx2 or Nx3 array. Tracking data shape was: {}. Maybe you forgot to transpose the data?".format(data.shape))

    roi_names = [k.lower() for k in list(rois.keys())]
    if "none" in roi_names:
        raise ValueError("No roi can have name 'none', that's reserved for the code to use, please use a different name for your rois.")

    if "tot" in roi_names:
        raise ValueError("No roi can have name 'tot', that's reserved for the code to use, please use a different name for your rois.")

    # get roi at each frame of data
    data_rois = get_roi_at_each_frame(data, rois, check_inroi)
    # print(data_rois)
    data_time_inrois = {name: data_rois.count(name) for name in set(data_rois)}  # total time (frames) in each roi

    # number of enters in each roi
    transitions = [n for i, n in enumerate(list(data_rois)) if i == 0 or n != list(data_rois)[i - 1]]
    transitions_count = {name: transitions.count(name) for name in transitions}

    # avg time spend in each roi (frames)
    avg_time_in_roi = {transits[0]: time / transits[1]
                       for transits, time in zip(transitions_count.items(), data_time_inrois.values())}

    # avg time spend in each roi (seconds)
    if fps is not None:
        data_time_inrois_sec = {name: t / fps for name, t in data_time_inrois.items()}
        avg_time_in_roi_sec = {name: t / fps for name, t in avg_time_in_roi.items()}
    else:
        data_time_inrois_sec, avg_time_in_roi_sec = None, None

    # get avg velocity in each roi
    avg_vel_per_roi = {}
    for name in set(data_rois):
        indexes = get_indexes(data_rois, name)
        vels = data[indexes, 2]
        avg_vel_per_roi[name] = np.average(np.asarray(vels))

    # get comulative
    transitions_count['tot'] = np.sum(list(transitions_count.values()))
    data_time_inrois['tot'] = np.sum(list(data_time_inrois.values()))
    data_time_inrois_sec['tot'] = np.sum(list(data_time_inrois_sec.values()))
    avg_time_in_roi['tot'] = np.sum(list(avg_time_in_roi.values()))
    avg_time_in_roi_sec['tot'] = np.sum(list(avg_time_in_roi_sec.values()))
    avg_vel_per_roi['tot'] = np.sum(list(avg_vel_per_roi.values()))

    if returndf:
        roinames = sorted(list(data_time_inrois.keys()))
        results = pd.DataFrame.from_dict({
                    "ROI_name": roinames, 
                    "transitions_per_roi": [transitions_count[r] for r in roinames],
                    "cumulative_time_in_roi": [data_time_inrois[r] for r in roinames],
                    "cumulative_time_in_roi_sec": [data_time_inrois_sec[r] for r in roinames],
                    "avg_time_in_roi": [avg_time_in_roi[r] for r in roinames],
                    "avg_time_in_roi_sec": [avg_time_in_roi_sec[r] for r in roinames],
                    "avg_vel_in_roi": [avg_vel_per_roi[r] for r in roinames],
                    })
    else:
        results = dict(transitions_per_roi=transitions_count,
                cumulative_time_in_roi=data_time_inrois,
                cumulative_time_in_roi_sec=data_time_inrois_sec,
                avg_time_in_roi=avg_time_in_roi,
                avg_time_in_roi_sec=avg_time_in_roi_sec,
                avg_vel_in_roi=avg_vel_per_roi)

    return results
from collections import namedtuple


def calc_distance_between_points_in_a_vector_2d(v1):
    '''calc_distance_between_points_in_a_vector_2d [for each consecutive pair of points, p1-p2, in a vector, get euclidian distance]
    This function can be used to calculate the velocity in pixel/frame from tracking data (X,Y coordinates)
    
    Arguments:
        v1 {[np.array]} -- [2d array, X,Y position at various timepoints]
    
    Raises:
        ValueError
    
    Returns:
        [np.array] -- [1d array with distance at each timepoint]
    >>> v1 = [0, 10, 25, 50, 100]
    >>> d = calc_distance_between_points_in_a_vector_2d(v1)
    '''
    # Check data format
    if isinstance(v1, dict) or not np.any(v1) or v1 is None:
            raise ValueError(
                'Feature not implemented: cant handle with data format passed to this function')

    # If pandas series were passed, try to get numpy arrays
    try:
        v1, v2 = v1.values, v2.values
    except:  # all good
        pass
    # loop over each pair of points and extract distances
    dist = []
    for n, pos in enumerate(v1):
        # Get a pair of points
        if n == 0:  # get the position at time 0, velocity is 0
            p0 = pos
            dist.append(0)
        else:
            p1 = pos  # get position at current frame

            # Calc distance
            dist.append(np.abs(distance.euclidean(p0, p1)))

            # Prepare for next iteration, current position becomes the old one and repeat
            p0 = p1

    return np.array(dist)





