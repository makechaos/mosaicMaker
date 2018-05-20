from PIL import Image
import sys
import os
import glob
import numpy as np

class mosaic():

  def __init__(self, simgFile, imdir, project=None, nImages=70, imgSize=100):
    self.exts = ['jpg']
    self.simgFile = simgFile
    self.imdir = imdir
    self.nImages = nImages
    self.imgSize = imgSize
    if project==None:
      project = 'mosaic_' + datetime.datetime.now().strftime('%Y%m%d_%H%M')
    self.project = project
    self.fil2val = None
    self.srcim = None
    self.img = None
    self.map = None
    self.thumbDir = 'thumb'
  
  def createMosaic(self):
    """
    Master function to make and save the mosaic image
    """
    # load the reference image and scale it down or pixelize it
    self.getScaledSrcImage()
    # load the images and save them as thumbnails, generate representative stats
    self.getImageStats()
    # build the mosaic image using scaled-down reference and above stats
    self.buildMosaic()
    # post processing to achieve good contrast and smooth view
    self.postProcess()
    # store as image
    self.img = Image.fromarray(self.img, mode='RGB')
    self.img.save()
    print('Mosaic image : '+self.project+'.png')
    print('DONE.')
    
  def buildMosaic(self, nrand=3):
    """
    Builds the mosaic image in 2 steps:
    1. map the reference image pixel by pixel with the thumbnail stats. A random selection
       of top "nrand" match is used to improve distribution of images used and also provide 
       variety in mosaic
    2. build the mosaic image from thumbnails
    """
    print('creating image map ...\r'),
    self.map = np.zeros((self.srcim.shape[0],self.srcim.shape[1]),dtype='int')
    N = self.map.shape[0]*self.map.shape[1]
    cnt = 0
    for m in range(self.map.shape[0]):
      for n in range(self.map.shape[1]):
        cnt += 1
        print('creating image map ... %d/%d\r'%(cnt,N)),
        ch = self.srcim[m,n]
        hh = map(lambda(x):[x[0],x[1],np.sum(np.abs(x[2]-ch))], self.fil2val)
        hh.sort(key=lambda(x):x[2])
        self.map[m,n] = int(hh[int(np.random.uniform(0,nrand))][0])
    print('')
    self.hist = np.histogram(self.map, len(self.fil2val))
    hh = self.hist[0]
    print('histogram:')
    print(hh)
    print('stats: max=%d, min=%d, median=%d, zeros=%d'%(np.max(hh),np.min(hh),np.median(hh),np.sum(hh==0)))
    
    print('building the image ...\r'),
    cnt = 0
    self.img = np.zeros((self.srcim.shape[0]*self.imgSize, self.srcim.shape[1]*self.imgSize, 3),dtype='uint8')
    for m in range(self.map.shape[0]):
      for n in range(self.map.shape[1]):
        cnt += 1
        print('building the image ...%d/%d\r'%(cnt,N)),
        mm = m*self.imgSize
        nn = n*self.imgSize
        im = Image.open(self.fil2val[self.map[m,n]][1])
        self.img[mm:mm+self.imgSize, nn:nn+self.imgSize, :] = im.resize((self.imgSize,self.imgSize))
    print('')
    print('done')
    
  def getImageStats(self, createThumbs=True):
    """
    Save the individual images as thumbnails and also get the statistic for pixel matching
    """
    fils = []
    for ext in self.exts:
      fils += glob.glob(os.path.join(self.imdir,'*.'+ext))
    print('found %d images '%len(fils))
    
    if not os.access(self.thumbDir, os.F_OK):
      print('creating thumbnail dir')
      os.makedirs(self.thumbDir)
    
    print('collecting stats....\r'),
    self.fil2val = []
    cnt = 0
    for fil in fils:
      cnt += 1
      print('collecting stats....%d/%d\r'%(cnt,len(fils))),
      tfile = os.path.join(self.thumbDir,os.path.basename(fil))
      if not os.access(tfile, os.F_OK):
        im = Image.open(fil)
        im.thumbnail((self.imgSize,self.imgSize))
        im.save(tfile)
      else:
        im = Image.open(tfile)
      im = np.array(im)
      self.fil2val += [[len(self.fil2val), tfile, np.median(im,axis=(0,1))]]
    print('')
    
  def getScaledSrcImage(self):
    """
    Load the reference image and scaled it down in size
    """
    im = Image.open(self.simgFile)
    print('source image :'+str(im.size))
    im.thumbnail((self.nImages, self.nImages))
    self.srcim = np.array(im)
    print('scaled image :'+str(self.srcim.shape))
  
  def smoothEdges(self):
    """
    Perform smoothing operation at edges of imgaes in the mosaic image
    """
    temp = np.array([1., 3., 7., 3., 1.])/15.0
    img = np.copy(self.img)
    
    mxtemp = np.broadcast_to( np.reshape(temp,(1,-1)), (self.img.shape[0],5))
    for k in [0,1,2]:
      for m in range(self.imgSize, self.img.shape[1], self.imgSize):
        for h in range(-3,4):
          self.img[:,m+h,k] = np.sum(img[:,m+h-3:m+h-3+5,k]*mxtemp, axis=1)
          
    mxtemp = np.broadcast_to( np.reshape(temp,(-1,1)), (5,self.img.shape[1]))
    for k in [0,1,2]:
      for m in range(self.imgSize, self.img.shape[0], self.imgSize):
        for h in range(-3,4):
          self.img[m+h,:,k] = np.sum(img[m+h-3:m+h-3+5,:,k]*mxtemp, axis=0)
  
  def scaleRGB(self):
    """
    scale the RGB values of individual thumbnails to match that of the reference image pixel.
    This offers better contrast or that similar to reference image.
    """
    self.img = np.array(self.img, dtype='float')
    for m in range(self.srcim.shape[0]):
      for n in range(self.srcim.shape[1]):
        mm = m*self.imgSize
        nn = n*self.imgSize
        cc = np.median(self.img[mm:mm+self.imgSize,nn:nn+self.imgSize,:],axis=(0,1))
        sc = self.srcim[m,n,:]/cc
        for k in [0,1,2]:
          self.img[mm:mm+self.imgSize,nn:nn+self.imgSize,k] *= sc[k]
  
  def postProcess(self):
    """
    Composite function with value clipping to yield valid image array data
    """
    self.img = np.array(self.img, dtype='float')
    print('smooth edges ...')
    self.smoothEdges()
    print('scale for contrast ...')
    self.scaleRGB()
    
    self.img[self.img>255] = 255
    self.img[self.img<0] = 0
    self.img = np.array(self.img, dtype='uint8')
    print('done post processing.')
    
if __name__ == '__main__':
  print('Usage: python %s mainImage imageDir <projName> <noOfImages:sizeOfImages>'%sys.argv[0])
  
  proj = None
  setting = [70, 100]
  if len(sys.argv)>3:
    proj = sys.argv[3]
  if len(sys.argv)>4:
    setting = map(int, sys.argv[4].split(':'))
  
  mos = mosaic(sys.argv[1], sys.argv[2], proj, setting[0], setting[1])
  mos.buildMosaic()
