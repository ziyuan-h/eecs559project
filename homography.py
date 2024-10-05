from os import stat
import cv2
import numpy as np
from util import showStitched

class LocalHomographyInfo:
  """ Structure storing local homographies. """
  
  def __init__(self, dst: int=None, homo: np.array=None, 
          matchPoints: np.array=None, status=None) -> None:
    self.dst = dst  # destination mapping / plane
    self.homo = homo  # actual homography
    self.matchPoints = matchPoints  # actual points matched
    self.status = status  # homograpy status

  def isEmpty(self) -> bool:
    return self.dst is None    
  

class LocalHomography:
  """ Class for local homography generation. """

  def __init__(self, keypoints: list, matches: list) -> None:
    assert len(keypoints) == len(matches)
    self.keypoints = keypoints
    self.matches = matches
    self.localHomoDict = None

  def estimate(self, reprojThresh=5) -> dict:
    localHomoDict = {}
    for i in range(len(self.keypoints)):
      localHomoDict[i] = LocalHomographyInfo()
      if len(self.matches[i][1]) > 4:
        pointsA = np.float32([self.keypoints[i][m.queryIdx].pt 
                              for m in self.matches[i][1]])
        pointsB = np.float32([self.keypoints[self.matches[i][0]][m.trainIdx].pt 
                              for m in self.matches[i][1]])
        # print(pointsA.shape)
        points = np.stack([pointsA, pointsB], axis=0)
        H, status = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, reprojThresh)
        localHomoDict[i] = LocalHomographyInfo(self.matches[i][0], H, 
                                                  points, status)
    
    self.localHomoDict = localHomoDict
    return self.localHomoDict


  def globalHomography(self, images) -> dict:
    assert self.localHomoDict is not None
    assert all([not value.isEmpty() for key, value 
                in self.localHomoDict.items()])
    globalHomo = {key: value.homo.copy() for key, value 
                    in self.localHomoDict.items()}
    # print(globalHomo)
    referenceCount = np.zeros(len(globalHomo))
    for i in self.localHomoDict:
      referenceCount[self.localHomoDict[i].dst] += 1
      
    key = np.argmax(referenceCount)

    width = np.sum([image.shape[1] for image in images])
    height = np.sum([image.shape[0] for image in images])
    globalHomo[key] = np.eye(3)
    globalHomo[key][[0, 1], [2, 2]] = [width/2, height/2]
    # print(globalHomo)
    visited = {key}
    while len(visited) < len(self.localHomoDict):
      curVisit = set()
      for i in self.localHomoDict:
        if self.localHomoDict[i].dst in visited:
          globalHomo[i] = globalHomo[self.localHomoDict[i].dst] \
                                @ globalHomo[i]
          curVisit.add(i)
      visited = visited.union(curVisit)
    # print(globalHomo)
    return globalHomo


  def showLocalStitched(self, images) -> None:
    showStitched(images, self.globalHomography(images))

