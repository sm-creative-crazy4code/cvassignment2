import cv2
import numpy as np
from matplotlib import pyplot as plt


# loading the images in grayscale
img1=cv2.imread("testphoto1.png",cv2.IMREAD_GRAYSCALE)
img2=cv2.imread("testphoto2.png",cv2.IMREAD_GRAYSCALE)
img3=cv2.imread("testphoto3.png",cv2.IMREAD_GRAYSCALE)

# setting the sift dectector

sift=cv2.SIFT_create()

# setting the keypoints and descripors for the images
kp1,descp1=sift.detectAndCompute(img1,None)
kp2,descp2=sift.detectAndCompute(img2,None)
kp3,descp3=sift.detectAndCompute(img3,None)

# matching between img1 and img2
brute_force12=cv2.BFMatcher()
matches_pair12=brute_force12.match(descp1, descp2)
matches_pair12=sorted(matches_pair12,key= lambda x:x.distance)

# matching between img2 and img3
brute_force23=cv2.BFMatcher()
matches_pair23=brute_force23.match(descp2, descp3)
matches_pair23=sorted(matches_pair23,key= lambda x:x.distance)

# matching between img3 and img1
brute_force31=cv2.BFMatcher()
matches_pair31=brute_force23.match(descp3, descp1)
matches_pair31=sorted(matches_pair31,key= lambda x:x.distance)


# matching results for image1 and image2
matching_results_pair12=cv2.drawMatches(img1,kp1,img2,kp2,matches_pair12[:50],None)
# matching results for image2 and  image3
matching_results_pair23=cv2.drawMatches(img2,kp2,img3,kp3,matches_pair23[:50],None)
# matching results for image3 and  image1
matching_results_pair31=cv2.drawMatches(img3,kp3,img1,kp1,matches_pair31[:50],None)


cv2.imshow("SIFT_Feature_Matching_pair12", matching_results_pair12)
cv2.imshow("SIFT_Feature_Matching_pair23", matching_results_pair23)
cv2.imshow("SIFT_Feature_Matching_pair31", matching_results_pair31)


cv2.imwrite('SIFT_Feature_Matching_pair12.png',matching_results_pair12)
cv2.imwrite('SIFT_Feature_Matching_pair23.png',matching_results_pair23)
cv2.imwrite('SIFT_Feature_Matching_pair31.png',matching_results_pair31) 


cv2.waitKey(0)
cv2.destroyAllWindows()