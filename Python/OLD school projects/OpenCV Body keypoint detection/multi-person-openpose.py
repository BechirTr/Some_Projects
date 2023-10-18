import cv2
import time
import numpy as np
from random import randint
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import os
LID=[0,1,2,3,4,5,6,7,8,9,10,11]
Xcoord=[]
Ycoord=[]
Lankle=[]
angle1=[]
angle2=[]
Rankle=[]

def angle(V1,V2):
    x1=V1[0]
    y1=V1[1]
    x2=V2[0]
    y2=V2[1]
    ps=x1*x2+y1*y2
    norm=((x1**2)+(y1**2))*((x2**2)+(y2**2))
    c=ps/m.sqrt(norm)
    return(np.arccos(c))
def getvector(x1,x2,y1,y2):
    x=x2-x1
    y=y2-y1
    V=[x,y]
    return V
ch1=input("where is the data ?")
#C:/Users/Bechir\Desktop/learnopencv-master/OpenPose-Multi-Person/saut 5/

ch2=input("what is the image ID?")
#429-4_20190830_154344_C001H001S00010000
#429-2_20190830_150227_C001H001S00010000
#429-1_20190830_142648_C001H001S00010000
#429-2_20190830_150227_C001H001S00010000
dataloc=input("where to save local data for each image ?")
path=dataloc
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
#C:/Users/Bechir/Desktop/data PIE/
ch4=input("where to save final data ?")
path=ch4
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
#C:/Users/Bechir/Desktop/data PIE/
start=int(input("from which step does the data start?"))
step=int(input("what is the value of the sampling step ?"))
final=int(input("what is the index of the last image ?"))
im_id=start

while im_id<final+1:


    ch=ch1+ch2+str(im_id)+".jpg"
    if im_id>=100:
        ch3=ch2[0:38]
        ch=ch1+ch3+str(im_id)+".jpg"
    image1 = cv2.imread(ch)
    print("proccessing "+str(im_id)+".jpg")
    protoFile = "C:/Users/Bechir\Desktop/learnopencv-master/OpenPose-Multi-Person/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "C:/Users/Bechir/Desktop/learnopencv-master/OpenPose-Multi-Person/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    # COCO Output Format
    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

    POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
                [1,0], [0,14], [14,16], [0,15], [15,17],
                [2,17], [5,16] ]

    # index of pafs correspoding to the POSE_PAIRS
    # e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
            [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
            [47,48], [49,50], [53,54], [51,52], [55,56],
            [37,38], [45,46]]

    colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
            [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
            [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


    def getKeypoints(probMap, threshold=0.1):

        mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

        mapMask = np.uint8(mapSmooth>threshold)
        keypoints = []

        #find the blobs
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #for each blob find the maxima
        for cnt in contours:
            blobMask = np.zeros(mapMask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = mapSmooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

        return keypoints


    # Find valid connections between the different joints of a all persons present
    def getValidPairs(output):
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7
        # loop for every POSE_PAIR
        for k in range(len(mapIdx)):
            # A->B constitute a limb
            pafA = output[0, mapIdx[k][0], :, :]
            pafB = output[0, mapIdx[k][1], :, :]
            pafA = cv2.resize(pafA, (frameWidth, frameHeight))
            pafB = cv2.resize(pafB, (frameWidth, frameHeight))

            # Find the keypoints for the first and second limb
            candA = detected_keypoints[POSE_PAIRS[k][0]]
            candB = detected_keypoints[POSE_PAIRS[k][1]]
            nA = len(candA)
            nB = len(candB)

            # If keypoints for the joint-pair is detected
            # check every joint in candA with every joint in candB
            # Calculate the distance vector between the two joints
            # Find the PAF values at a set of interpolated points between the joints
            # Use the above formula to compute a score to mark the connection valid

            if( nA != 0 and nB != 0):
                valid_pair = np.zeros((0,3))
                for i in range(nA):
                    max_j=-1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        # Find d_ij
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        # Find p(u)
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        # Find L(p(u))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                            pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                        # Find E
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores)/len(paf_scores)

                        # Check if the connection is valid
                        # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                        if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    # Append the connection to the list
                    if found:
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                # Append the detected connections to the global list
                valid_pairs.append(valid_pair)
            else: # If no keypoints are detected
                print("No Connection : k = {}".format(k))
                invalid_pairs.append(k)
                valid_pairs.append([])
        return valid_pairs, invalid_pairs



    # This function creates a list of keypoints belonging to each person
    # For each detected valid pair, it assigns the joint(s) to a person
    def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
        # the last number in each row is the overall score
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(mapIdx)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:,0]
                partBs = valid_pairs[k][:,1]
                indexA, indexB = np.array(POSE_PAIRS[k])

                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        # add the keypoint_scores for the two keypoints and the paf_score
                        row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints


    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]

    t = time.time()
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight/frameHeight)*frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    print("Time Taken in forward pass = {}".format(time.time() - t))

    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)


    frameClone = image1.copy()
    for i in range(nPoints):
        for j in range(len(detected_keypoints[i])):
            cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)
    cv2.circle(frameClone,(465,155), 5,(164,245,255), -1)
    cv2.line(frameClone, (479,97), (469, 212), (0,255,255),2)
    cv2.line(frameClone, (438,108), (478, 213), (0,255,255),2)
    #cv2.imshow("Keypoints",frameClone)

    valid_pairs, invalid_pairs = getValidPairs(output)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)

    for i in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)


    #cv2.imshow("Detected Pose" , frameClone)
    cv2.waitKey(0)
    x=0
    cht=dataloc+"data"+str(im_id)+".csv"
    print(cht)
    BodyData=pd.DataFrame(detected_keypoints)
    BodyData.index=['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
    if len(BodyData.columns)>2:
        BodyData=BodyData.drop(columns=2)
    if len(BodyData.columns)>1:
        BodyData=BodyData.drop(columns=1)
    BodyData.to_csv(cht)

    xsum=0
    ysum=0
    for j in range(8):
        if len(detected_keypoints[j])!=0 :
            print(j)
            xsum=xsum+(detected_keypoints[j][0][0]-(1280/2))
            ysum=ysum+((1112/2)-detected_keypoints[j][0][1])
        else:
            j=j+1
    Xg=xsum/len(detected_keypoints)
    Yg=ysum/len(detected_keypoints)
    Xcoord.append(Xg)
    Ycoord.append(Yg)
    print("Xmoy= ",Xg," Ymoy= ",Yg)
    if len(detected_keypoints[13])!=0:
        Lankle.append(detected_keypoints[13][0][1])
    if len(detected_keypoints[13])!=0:
        Rankle.append(detected_keypoints[10][0][1])
    im_id=im_id+step
    if len(detected_keypoints[8])!=0 and len(detected_keypoints[9])!=0 and len(detected_keypoints[10])!=0 and len(detected_keypoints[11])!=0 and len(detected_keypoints[12])!=0 and len(detected_keypoints[13])!=0:

        V1=getvector(detected_keypoints[8][0][0]-(1280/2),detected_keypoints[9][0][0]-(1280/2),(1112/2)-detected_keypoints[8][0][1],(1112/2)-detected_keypoints[9][0][1])
        V2=getvector(detected_keypoints[9][0][0]-(1280/2),detected_keypoints[10][0][0]-(1280/2),(1112/2)-detected_keypoints[9][0][1],(1112/2)-detected_keypoints[10][0][1])
        V3=getvector(detected_keypoints[11][0][0]-(1280/2),detected_keypoints[12][0][0]-(1280/2),(1112/2)-detected_keypoints[11][0][1],(1112/2)-detected_keypoints[12][0][1])
        V4=getvector(detected_keypoints[12][0][0]-(1280/2),detected_keypoints[13][0][0]-(1280/2),(1112/2)-detected_keypoints[12][0][1],(1112/2)-detected_keypoints[13][0][1])
        angle1.append(angle(V1,V2))
        angle2.append(angle(V3,V4))
time=[]
i=43
df=[Xcoord,Ycoord,angle1,angle2,Lankle,Rankle]

#C:/Users/Bechir/Desktop/data PIE/
chts=ch4+"Finaldata"+".csv"
data=pd.DataFrame(df)
data.index=["Xcoord","Ycoord","angle1","angle2","Lankle","Rankle"]
for i in range(len(data.columns)):
    for j in range(len(data)):
        if m.isnan(data[i][j]):
            if i==0:
                data[i][j]=data.mean(axis=0)
            else:
                data[i][j]=data[i-1][j]
data.to_csv(chts)
while i<236:
    time.append(i+4)
    i=i+4
# to plot Y coordinates
plt.plot(Ycoord)
plt.show()
#to plot knee angles
plt.subplot(211)
plt.plot(Lankle)
plt.subplot(212)
plt.plot(Rankle)
plt.show()