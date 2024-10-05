from evaluate import evaluate
import numpy as np
import matplotlib.pyplot as plt
from util import showStitched

class ImageStitching:
  def __init__(self, images, matches, X1=None, X2=None, H12=None):
    self.X1 = X1 
    self.X2 = X2
    self.H12 = H12
    self.images = images
    self.matches = matches

  def parseInput(self, localHomo):
    status = localHomo.status
    X1 = np.float32(localHomo.matchPoints[0])[localHomo.status.squeeze()>0,:]
    X2 = np.float32(localHomo.matchPoints[1])[localHomo.status.squeeze()>0,:]
    X1 = np.hstack((X1, np.ones_like(X1[:,[0]])))
    X2 = np.hstack((X2, np.ones_like(X2[:,[0]])))
    if self.X1 is None:
      self.X1 = X1.T
      self.X2 = X2.T
      self.H12 = localHomo.homo.copy()
    else:
      self.X1 = np.hstack((self.X1, X1.T))
      self.X2 = np.hstack((self.X2, X2.T))

  def trivialTrain(self):
    width = np.sum([image.shape[1] for image in self.images])
    height = np.sum([image.shape[0] for image in self.images])
    self.H2 = np.eye(3)
    self.H2[[0, 1], [2, 2]] = [width/2, height/2]
    self.H1 = self.H2@self.H12
    return

  def evaluateADM(self, M):
    width = np.sum([image.shape[1] for image in self.images])
    height = np.sum([image.shape[0] for image in self.images])
    self.H2 = np.eye(3)
    self.H2[[0, 1], [2, 2]] = [width/2, height/2]
    self.H1 = self.H2@M
    return sum(evaluate(self.images, self.matches, self.globalHomography))

  def evaluateGD(self, H1, H2):
    self.H1 = H1
    self.H2 = H2
    return sum(evaluate(self.images, self.matches, self.globalHomography))

  def ADMTrain(self, lam=0.2, Niter=10000, eps=1e-6):
    M = self.H12
    s = np.ones(self.X1.shape[1])
    obj = lambda M,s: np.linalg.norm(M@self.X1-self.X2*s[np.newaxis,:],'fro')**2 \
                    +lam*np.linalg.norm(M-self.H12,'fro')**2
    n = self.X1.shape[1]
    self.out = np.zeros(Niter+1)
    self.out[0] = self.evaluateADM(M)
    A = np.linalg.inv(lam*np.eye(3) + self.X1@self.X1.T)

    # Window size for running sum
    window_size = 10
    minVal = None
    minM = None
    mins = None
    minItr = None
    for itr in range(Niter):
      M = ((self.X2*s[np.newaxis,:])@self.X1.T + lam*self.H12) @ A
      s = np.diag(self.X1.T@M.T@self.X2).squeeze() / \
          np.linalg.norm(self.X2, ord=2, axis=0)**2
      
      self.out[itr+1] = self.evaluateADM(M)
      if minVal == None or self.out[itr+1] < minVal:
        minVal = self.out[itr+1]
        minM = M.copy()
        mins = s.copy()
        minItr = itr

      print(self.out[itr+1])
      # if np.abs(self.out[itr+1]-self.out[itr])/np.abs(self.out[itr]) < eps:
      #   break
      if itr - minItr > window_size:
        self.evaluateADM(minM)
        break 
    
    self.out = self.out[:itr+2]
    return

  def GDTrain(self, lam=0.2, Niter=4000, eps=1e-10):
    # Preset the value of H2
    width = np.sum([image.shape[1] for image in self.images])
    height = np.sum([image.shape[0] for image in self.images])
    self.H2 = np.eye(3)
    self.H2[[0, 1], [2, 2]] = [width/2, height/2]

    # Initialization
    H1 = self.H2@self.H12
    s = np.ones(self.X1.shape[1])
    
    # Intermediate Routines
    H2inv = np.linalg.inv(self.H2)
    funNew = lambda H1, s: 0.5*np.linalg.norm(H2inv@H1@self.X1 - 
                              self.X2*s[np.newaxis,:], 'fro')**2 + \
                              0.5*lam*np.linalg.norm(H2inv@H1 - self.H12, 
                                                     'fro')**2

    gradH1New = lambda H1, s: H2inv.T @ (H2inv@H1@self.X1 -\
                                         self.X2*s[np.newaxis,:]) @ self.X1.T +\
                              lam*H2inv.T@(H2inv@H1 - self.H12)

    hessH1NewInv = np.eye(3)  # np.linalg.inv(H2inv.T@H2inv@(self.X1@self.X1.T + lam*np.eye(3)))

    def backtrack(H1, fun, eta, dr, alp=0.2, beta=0.5):
      eta = alp*eta 
      while fun(H1-eta*dr) > fun(H1) - (0.5*eta)*np.linalg.norm(dr, 'fro')**2:
          eta = beta*eta
      return eta 
    
    self.out = np.zeros(Niter+1)
    # self.out[0] = self.evaluateGD(H1, self.H2)
    self.out[0] = funNew(H1, s)
    eta = 1
    for itr in range(Niter):
      # H1 -= hessH1inv @ gradH1(H1, s)
      dr = hessH1NewInv @ gradH1New(H1, s)
      eta = backtrack(H1, lambda z: funNew(z, s), eta, dr)
      H1 -= eta*dr
      
      s = np.diag(self.X1.T@H1.T@self.H2@self.X2).squeeze() / \
            np.linalg.norm(self.H2@self.X2, ord=2, axis=0)**2
      self.out[itr+1] = self.evaluateGD(H1, self.H2)

      self.out[itr+1] = funNew(H1, s)
      print(itr, self.out[itr])
      if np.abs(self.out[itr+1]-self.out[itr])/np.abs(self.out[itr]) < eps:
        break
    
    self.H1 = H1
    self.out = self.out[:itr+2]
    return

  def getResult(self):
    return self.H1, self.H2, self.out

  @property
  def globalHomography(self):
    return {0: self.H1, 1: self.H2}

  def plotStitched(self):
    assert self.H1 is not None
    assert self.H2 is not None
    showStitched(self.images, self.globalHomography)
    return

  def plotTrain(self):
    plt.plot(self.out)
