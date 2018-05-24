from PIL import Image
import sys
import os
import glob
import numpy as np


def compareReplacements(hist, fil2val, lthresh=6, uthresh=50):
  N = len(fil2val)
  dmap = np.zeros((N,N))
  for m in range(N):
    for n in range(m,N):
      dd = fil2val[m][2] - fil2val[n][2]
      dmap[m,n] = np.sum(dd*dd)
      dmap[n,m] = dmap[m,n]
  print(dmap.shape)
  
  indx1 = np.where(hist[0]<lthresh)[0]
  indx2 = np.where(hist[0]>uthresh)[0]
  
  d2 = dmap[indx1,:]
  print(d2.shape)
  d2 = d2[:,indx2]
  print(d2.shape)
  
  fd = open('compRepl.html','w')
  fd.write('<html><head></head><body>\n')
  for m in range(len(indx1)):
    mm = indx1[m]
    nn = indx2[np.argmin(d2[m,:])]
    fd.write('<img src="%s"><img> %d, %s -- %d, %s <img src="%s"></img><br><hr>\n'%(fil2val[mm][1],hist[0][mm],str(fil2val[mm][2]),hist[1][nn],str(fil2val[nn][2]),fil2val[nn][1]))
  fd.write('</body></html>')
  fd.close()
  
  return (dmap, indx1, indx2, d2)

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

  def saveData(self):
    dd = {'fil2val':self.fil2val, 'map':self.map, 'srcim':self.srcim, 'hist':self.hist,'imdir':self.imdir,'nImages':self.nImages,'imgSize':self.imgSize}
    pickle.dump(dd,open(self.project+'.pkl','w'))

  def loadData(self):
    dfile = self.project+'.pkl'
    if not os.access(dfile, os.F_OK):
      return False
    dd = pickle.load(open(dfile,'r'))
    self.fil2val = dd['fil2val']
    self.map = dd['map']
    self.srcim = dd['srcim']
    self.hist = dd['hist']
    self.imdir = dd['imdir']
    self.nImages = dd['nImages']
    self.imgSize = dd['imgSize']
    return True
  
  def createStatHtml(self):
    fd = open(self.project+'.html','w')
    fd.write('<html><head><title>%s</title></head><body>\n'%self.project)
    hh = []
    for n in range(len(self.hist[0])):
      hh += [[n,self.hist[0][n]]]
    harr = np.array(self.hist[0])
    hh.sort(key=lambda(x):x[1])
    pn = -1
    for h in hh:
      if pn!=h[1]:
        fd.write('<br><hr><h2>%d (%d)</h2><br>\n'%(h[1],np.sum(harr==h[1])))
      pn = h[1]
      fd.write('<img src="%s" style="margin:2px;"></img>\n'%self.fil2val[h[0]][1])
    fd.write('</body></html>')
    fd.close()
    
  def createMosaic(self, nrand=3, hist=False):
    """
    Master function to make and save the mosaic image
    """
    histOk = False
    if hist:
      histOk = self.loadData()
      
    if not histOk:
      # load the reference image and scale it down or pixelize it
      self.getScaledSrcImage()
      # load the images and save them as thumbnails, generate representative stats
      self.getImageStats()
    
    # build the mosaic image using scaled-down reference and above stats
    self.buildMosaic(nrand)
    
    if not histOk:
      self.saveData()
    
    # post processing to achieve good contrast and smooth view
    self.postProcess()
    # store as image
    self.img = Image.fromarray(self.img, mode='RGB')
    self.img.save(self.project+'.png')
    print('Mosaic image : '+self.project+'.png')
    
    print('DONE.')
    
  def buildMosaic(self, nrand=20):
    """
    Builds the mosaic image in 2 steps:
    1. map the reference image pixel by pixel with the thumbnail stats. A random selection
       of top "nrand" match is used to improve distribution of images used and also provide 
       variety in mosaic. If nrand==-1, then fully randomized picking happens. nrand==1 then 
       only the closest match is used (no variety). Generally nrand~no.of.images/3 or 20 is 
       a good choice.
    2. build the mosaic image from thumbnails
    """
    if nrand>0:
      nrand = max(nrand, len(self.fil2val)
    
    if self.map==None:
      print('creating image map ...\r'),
      self.map = np.zeros((self.srcim.shape[0],self.srcim.shape[1]),dtype='int')
      N = self.map.shape[0]*self.map.shape[1]
      cnt = 0
      for m in range(self.map.shape[0]):
        for n in range(self.map.shape[1]):
          cnt += 1
          print('creating image map ... %d/%d\r'%(cnt,N)),
          if nrand<0:
            self.map[m,n] = int(np.random.uniform(0,len(self.fil2val)))
          else:
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
      
      print('creating stat html ...')
      self.createStatHtml()
    
    print('building the image ...\r'),
    cnt = 0
    N = self.map.shape[0]*self.map.shape[1]
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
  print('Usage: python %s mainImage imageDir <projName> <noOfImages:sizeOfImages:nrand>'%sys.argv[0])
  
  proj = None
  setting = [70, 100]
  if len(sys.argv)>3:
    proj = sys.argv[3]
  if len(sys.argv)>4:
    setting = map(int, sys.argv[4].split(':'))
  
  mos = mosaic(sys.argv[1], sys.argv[2], proj, setting[0], setting[1])
  mos.buildMosaic(nrand=setting[2])
