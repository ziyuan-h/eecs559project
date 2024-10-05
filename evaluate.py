import numpy as np 
import cv2

def GemanMcClure(x, a):
  """ Geman-McClure function """
  if a == 0:
    return 0
  else:
    return (x**2)/ (1 + x**2 / a**2)

def jointEvaluate(warpIm1: np.array, warpIm2: np.array) -> float:
  """ Evaluate a pair of images using GM metric. """
  warpIm1 = np.int16(warpIm1)
  warpIm2 = np.int16(warpIm2)
  a = np.sqrt(np.mean(np.abs(warpIm1 - warpIm2))*1.4)
  # print(a)
  return np.mean(GemanMcClure(warpIm1 - warpIm2, a))

def evaluate(images: list, matches: dict, globalHomo: dict) -> float:
  # assert len(images) == len(matches)
  assert len(images) == len(globalHomo)

  width = np.sum([image.shape[1] for image in images])*2
  height = np.sum([image.shape[0] for image in images])*2
  scoreList = []
  maskList = []
  warpList = []
  for imageId in matches:
    pairId = matches[imageId][0]
    # Create mask for overlapping regoins
    maskImage = cv2.warpPerspective(np.ones_like(images[imageId])*255, 
                                    globalHomo[imageId], (width, height))
    maskPair = cv2.warpPerspective(np.ones_like(images[pairId])*255, 
                                    globalHomo[pairId], (width, height))
    mask = cv2.bitwise_and(maskImage, maskPair)
    maskList.append(mask)
    
    # Construct the warped images restricted to the overlapping region
    warpImage = cv2.warpPerspective(images[imageId]*255, globalHomo[imageId], 
                                    (width, height))
    warpPair = cv2.warpPerspective(images[pairId]*255, globalHomo[pairId], 
                                    (width, height))
    warpImage = cv2.bitwise_and(warpImage, mask)
    warpPair = cv2.bitwise_and(warpPair, mask)

    # Eliminate non-overlapping points
    mask1d = mask.ravel() > 0
    # print(mask1d.sum())
    warpImage1d = warpImage.ravel()[mask1d]
    warpPair1d = warpPair.ravel()[mask1d]
    warpList.append((warpImage1d, warpPair1d))

    # Compute pairwise evaluation score
    score = jointEvaluate(warpImage1d, warpPair1d)
    scoreList.append(score)

  return scoreList  

def pointTransform(keypoints: dict, matches: dict, 
            globalHomo: dict) -> dict[int, tuple[int, np.array]]:
  """ Transform keypoints using global homography dictionary. 
      Inputs：
        keypoints[dict] Dictionary of arrays of size (2, num-matches, 2)
                        where the first and the last 2's indicate number 
                        matched images and the x-y coordinates resp. Keys 
                        are images being matched / operated (same as matches).
        matches[dict] Dictionary of the form mappedImageId -> 
                        (baseImageId, matchList)
        globalHomo[dict] Global homography mapping each mapped image to its
                        global homography matrix.
      Output: mappedImageId -> (baseImageId, transformed-points (2,m,2)) 
  """
  result = {}
  for imageId in keypoints:
    pairId = matches[imageId][0]
    # Convert to homogeneous coordinate
    points = keypoints[imageId]
    homoCoord = np.ones_like(points[:,:,0])[:,:,np.newaxis]
    points = np.concatenate((points, homoCoord), axis=2)
    newPoints = np.zeros_like(points)
    # Transform using homography matrices
    newPoints[0,:,:] = points[0,:,:] @ globalHomo[imageId].T
    newPoints[1,:,:] = points[1,:,:] @ globalHomo[pairId].T
    newPoints /= newPoints[:,:,[-1]]
    print(newPoints[:,:,:20])  # sanity checks
    result[imageId] = (pairId, newPoints[:,:,:-1])

  return result



