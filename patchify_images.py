import os
from PIL import Image

def scanSetFolder(folder):
  names = os.listdir(folder)
  names.sort()
  dirs = [os.path.join(folder, name) for name in names]
  return dirs, names

def crop(path, input, height, width, imgName, patchName):
  k = 0
  newDir = os.path.join(directory, patchName)
  if not os.path.exists(newDir):
    os.makedirs(newDir)
  try:
    im = Image.open(input)
  except:
    return
  imgwidth, imgheight = im.size
  for i in range(0,imgheight,height):
    for j in range(0,imgwidth,width):
      box = (j, i, j+width, i+height)
      a = im.crop(box)
      a.save(os.path.join(newDir, imgName[:-4] + "_patch_%s.jpg" % k))
      k +=1

if __name__ == "__main__":
  trainDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"training-data")
  testDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"testing-data")
  validDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"validation-data")

  trainDirs, trainNames = scanSetFolder(trainDir)
  testDirs, testNames = scanSetFolder(testDir)
  validDirs, validNames = scanSetFolder(validDir)

  for directory in trainDirs:
    a = os.path.join(directory,"input")
    b = os.path.join(directory,"GT")
    inputImgs = os.listdir(a)
    gtImgs = os.listdir(b)
    for img in inputImgs:
      path = os.path.join(a,img)
      crop(directory,path,128,128, img, "input_patches")
    for img in gtImgs:
      path = os.path.join(b,img)
      crop(directory,path,128,128, img, "GT_patches")