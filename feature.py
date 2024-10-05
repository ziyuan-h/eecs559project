from types import MethodWrapperType
import numpy as np
import cv2

class Detector:
  """ SIFT algorithm that returns features. """
  
  def __init__(self, method: str):
    self.keypointFeature = None
    self.method = method

  def detectAndDescribe(self, images: list) -> None:
    """ Return the list of keypoints associated with 
        each image in `images`. """
    keypointFeature = []
    if self.method == "sift":
      detector = cv2.xfeatures2d.SIFT_create()
    elif self.method == "brisk":
      detector = cv2.BRISK_create()
    elif self.method == "orb":
      detector = cv2.ORB_create()

    for image in images:
      keypointFeature.append(detector.detectAndCompute(image, None))
    
    self.keypointFeature = keypointFeature 

  @property
  def keypoints(self) -> list:
    assert self.keypointFeature is not None
    return [kpsFt[0] for kpsFt in self.keypointFeature]

  @property
  def features(self) -> list:
    assert self.keypointFeature is not None
    return [kpsFt[1] for kpsFt in self.keypointFeature]

  @property
  def methods(self) -> str:
    return self.method


class Matcher:
  """ Image matcher. """

  def __init__(self, detector: Detector) -> None:
    self.features = detector.features
    self.numImages = len(self.features)
    self.method = detector.methods

  def match(self) -> dict:
    """ The returned matches is a list of tuples (targetId, matchList). """
        # Create matcher
    if self.method == "sift":
      print('here')
      matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
      matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bestMatches = {}
    for i in range(self.numImages):
      bestDist = None
      bestMatch = None
      bestPeer = None
      for j in range(self.numImages):
        if j == i:
          continue
        curMatch = matcher.match(self.features[i], self.features[j])
        curDist = np.sum([float(m.distance ) for m in curMatch])
        if bestDist is None or curDist < bestDist:
          bestDist = curDist
          bestMatch = curMatch
          bestPeer = j
      # Sort the features in order of distance.
      # The points with small distance (more similarity) are ordered first in the vector
      bestMatch = sorted(bestMatch, key = lambda x:x.distance)
      bestMatches[i] = (bestPeer, bestMatch)

    return bestMatches

