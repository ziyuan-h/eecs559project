import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import imutils

# Extract images from the directory
def extractImages(work_dir: str, scene: str, numImages: int) -> list:
  """
    Extract two images corresponding to the specified scene index.
  """
  images = []
  for i in range(numImages):
    fileName = scene+'0'+str(i+1)
    if scene != 'snow':
      fileName += '.jpg'
    else:
      fileName += '.png'
    imagePath = os.path.join(work_dir, 'dataset', fileName)
    image = imageio.v3.imread(imagePath)
    images.append(image)
  
  return images


# Display raw images
def showRawImages(images: list) -> None:
  """
    Display raw images
  """
  fig, ax = plt.subplots(nrows=1, ncols=len(images), 
                          constrained_layout=False, figsize=(16,9))
  for i in range(len(images)):
    ax[i].imshow(images[i])
    ax[i].set_xlabel("Image {}".format(i), fontsize=14)

  plt.show()
  return


# Convert images to gray scale
def convert2Gray(images: list) -> list:
  grayImages = []
  for image in images:
    grayImages.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
  
  return grayImages


# Display keypoints detected
def showKeypoints(keypoints: list, images: list) -> None:
  """
    Function that shows the keypoints in two comparing images.
    Input:
      keypoints
  """
  assert len(keypoints) == len(images)
  fig, ax = plt.subplots(nrows=1, ncols=len(images), figsize=(20,8), 
                        constrained_layout=False)
  for i in range(len(images)):
    ax[i].imshow(cv2.drawKeypoints(images[i], keypoints[i], 
                                    None, color=(0,255,0)))
    ax[i].set_xlabel("Image {}".format(i), fontsize=14)

  plt.show()


# Plot matched images
def showMatching(images: list, keypoints: list, matches: dict, num=100):
  """
    Display matched images.
    Input:
      images[list]: List of np.array storing images to be plotted.
      keypoints[list]: Lists of key points corresponding to `images`.
      matches[dict]: Dictionary of matches corresponding to `images`.
      num[int]: Number of matches to be plotted.
  """
  assert len(images) == len(keypoints)
  assert len(images) == len(matches)
  fig, ax = plt.subplots(nrows=len(images), ncols=1, figsize=(8,10),
                         constrained_layout = False)
  for i in range(len(images)):
    ax[i].imshow(cv2.drawMatches(images[i], keypoints[i],
                          images[matches[i][0]], keypoints[matches[i][0]],
                          matches[i][1][:num], None, 
                          flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
    ax[i].set_xlabel("Image {} and {}".format(i, matches[i][0]))
  plt.show()


# Display stitched images.
def showStitched(images: list, homo: dict) -> None:
  """
    Display stitched images using the homography matrices.
    Input:
      images[list] Input a list of images to be stitched.
      homo[dict[imageId, homography matrix]] Input a dictionary 
              of matching images and homographies.
  """
  # Apply panorama correction
  width = np.sum([image.shape[1] for image in images])*2
  height = np.sum([image.shape[0] for image in images])*2

  background = np.zeros((height, width, 3), dtype=np.uint8)

  # Stitch the images
  for i in range(len(images)):
    result = cv2.warpPerspective(images[i], homo[i], (width, height))
    mask = cv2.warpPerspective(np.ones_like(images[i])*255, homo[i], 
                               (width, height))
    background = cv2.bitwise_and(background,
                                 cv2.bitwise_not(mask))
    background = cv2.bitwise_or(background, result)

  # transform the panorama image to grayscale and threshold it 
  gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

  # Finds contours from the binary image
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                          cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  # get the maximum contour area
  c = max(cnts, key=cv2.contourArea)

  # get a bbox from the contour area
  (x, y, w, h) = cv2.boundingRect(c)

  # crop the image to the bbox coordinates
  background = background[y:y + h, x:x + w]

  # show the cropped image
  plt.figure(figsize=(20,10))
  plt.imshow(background)
