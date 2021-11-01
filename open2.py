import numpy as np
import sys
import cv2 
import time

class SLIC:
    def __init__(self, img, step, nc):
        self.img = img
        ##cv2.imshow("superpixels", self.img)
        self.height, self.width = img.shape[:2]
       ## print(self.width)
        self._convertToLAB()
        
        self.step = step
        self.nc = nc
        self.ns = step
        self.FLT_MAX = 1000000
        self.ITERATIONS = 10

    def _convertToLAB(self):
        try:
            self.labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
            cv2.imwrite("LAB.jpg", self.labimg)

        except ImportError:
            self.labimg = np.copy(self.img)
            for i in range(self.labimg.shape[0]):
                for j in range(self.labimg.shape[1]):
                    rgb = self.labimg[i, j]
                    self.labimg[i, j] = self._rgb2lab(tuple(reversed(rgb)))

    def _rgb2lab ( self, inputColor ) :

       num = 0
       RGB = [0, 0, 0]

       ###DD
       for value in inputColor :
           ##print(value)
           value = float(value) / 255

           if value > 0.04045 :
               value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
           else :
               value = value / 12.92

           RGB[num] = value * 100
           num = num + 1

       XYZ = [0, 0, 0,]

       X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
       Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
       Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505

       
       XYZ[ 0 ] = round( X, 4 )
       XYZ[ 1 ] = round( Y, 4 )
       XYZ[ 2 ] = round( Z, 4 )



       XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
       XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
       XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883


       num = 0
       for value in XYZ :

           if value > 0.008856 :
               value = value ** ( 0.3333333333333333 )
           else :
               value = ( 7.787 * value ) + ( 16 / 116 )

           XYZ[num] = value
           num = num + 1

       Lab = [0, 0, 0]

       L = ( 116 * XYZ[ 1 ] ) - 16
       a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
       b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )


       Lab [ 0 ] = round( L, 4 )
       Lab [ 1 ] = round( a, 4 )
       Lab [ 2 ] = round( b, 4 )

       return Lab

    def generateSuperPixels(self):
        self._initData()
        # for i in range(5):
        #   for j in range(5):
        #       print(i,j, self.labimg[i,j])



        indnp = np.mgrid[0:self.height,0:self.width].swapaxes(0,2).swapaxes(0,1) ##why??
      ##  print(self.centers.shape[1])
        for i in range(self.ITERATIONS):
            self.distances = self.FLT_MAX * np.ones(self.img.shape[:2])
            # print(self.distances)
            for j in range(self.centers.shape[0]):
                xlow, xhigh = int(self.centers[j][3] - self.step), int(self.centers[j][3] + self.step)
                ylow, yhigh = int(self.centers[j][4] - self.step), int(self.centers[j][4] + self.step)

                if xlow <= 0:
                    xlow = 0
                if xhigh > self.width:
                    xhigh = self.width
                if ylow <=0:
                    ylow = 0
                if yhigh > self.height:
                    yhigh = self.height

                cropimg = self.labimg[ylow : yhigh , xlow : xhigh].astype(np.int64)
   
                colordiff = cropimg - self.labimg[int(self.centers[j][4]), int(self.centers[j][3])]

                # print("crpimg", cropimg)
                # print("self",self.labimg[int(self.centers[j][4]), int(self.centers[j][3])])
               
               ## print("colordiff",np.sum(np.square(colordiff), axis=2))

                colorDist = np.sqrt(np.sum(np.square(colordiff.astype(np.int64)), axis=2)) #Dlab
                ##print("ColorDist", colorDist)

                yy, xx = np.ogrid[ylow : yhigh, xlow : xhigh]
                
                pixdist = ((yy-self.centers[j][4])**2 + (xx-self.centers[j][3])**2)**0.5 #dxy
                dist = ((colorDist/self.nc)**2 + (pixdist/self.ns)**2)**0.5 ##normalized distance measure Ds

                distanceCrop = self.distances[ylow : yhigh, xlow : xhigh]
                # print("distcrp",distanceCrop)
                # print("distt",dist)
            

                idx = dist < distanceCrop
                # print("hi",i, idx)
                distanceCrop[idx] = dist[idx]
                self.distances[ylow : yhigh, xlow : xhigh] = distanceCrop
                self.clusters[ylow : yhigh, xlow : xhigh][idx] = j ##why??
            # t=time.time()

            for k in range(len(self.centers)):
                idx = (self.clusters == k)
                colornp = self.labimg[idx]
                distnp = indnp[idx]
                self.centers[k][0:3] = np.sum(colornp, axis=0)
                sumy, sumx = np.sum(distnp, axis=0)
                self.centers[k][3:] = sumx, sumy
                self.centers[k] /= np.sum(idx)
        # cv2.imwrite("second.jpg", self.clusters)    
        # cv2.imshow("clusters.jpg",slic.clusters)       

    def _initData(self):
        self.clusters = -1 * np.ones(self.img.shape[:2])
        ## grid of -1 with same height and width print(self.clusters)
        self.distances = self.FLT_MAX * np.ones(self.img.shape[:2])
     
        centers = []
        ##print(self.step, self.height - self.step/2, self.step)
        nk=0
        for i in range(self.step, self.width - self.step//2, self.step):
            for j in range(self.step, self.height - self.step//2, self.step):
                # nk=nk+1 
                nc = self._findLocalMinimum(center=(i, j))
                
                color = self.labimg[nc[1], nc[0]]  ###D why reverse
                ##print(color)
                ##print("next")
                center = [color[0], color[1], color[2], nc[0], nc[1]] ##CIELAB
                centers.append(center)
                ##centers= 170*5, 170 array each of size 5;
        self.center_counts = np.zeros(len(centers))
       ## print(centers)
        ##print(len(centers))
        self.centers = np.array(centers)
        
    def createConnectivity(self):
        label = 0
        adjlabel = 0
        lims = self.width * self.height // self.centers.shape[0]
        dx4 = [-1, 0, 1, 0]
        dy4 = [0, -1, 0, 1]
        new_clusters = -1 * np.ones(self.img.shape[:2]).astype(np.int64)
        elements = []
        for i in range(self.width):
            for j in range(self.height):
                if new_clusters[j, i] == -1:
                    elements = []
                    elements.append((j, i))
                    for dx, dy in zip(dx4, dy4):
                        x = elements[0][1] + dx
                        y = elements[0][0] + dy
                        if (x>=0 and x < self.width and 
                            y>=0 and y < self.height and 
                            new_clusters[y, x] >=0):
                            adjlabel = new_clusters[y, x]
                count = 1
                c = 0
                while c < count:
                    for dx, dy in zip(dx4, dy4):
                        x = elements[c][1] + dx
                        y = elements[c][0] + dy

                        if (x>=0 and x<self.width and y>=0 and y<self.height):
                            if new_clusters[y, x] == -1 and self.clusters[j, i] == self.clusters[y, x]:
                                elements.append((y, x))
                                new_clusters[y, x] = label
                                count+=1
                    c+=1
                if (count <= lims >> 2):
                    for c in range(count):
                        new_clusters[elements[c]] = adjlabel
                    label-=1
                label+=1
        self.new_clusters = new_clusters
        cv2.imwrite("connect.jpg",self.new_clusters)

    def displayContours(self, color):
        dx8 = [-1, -1, 0, 1, 1, 1, 0, -1]
        dy8 = [0, -1, -1, -1, 0, 1, 1, 1]

        isTaken = np.zeros(self.img.shape[:2], np.bool_)
        contours = []

        for i in range(self.width):
            for j in range(self.height):
                nr_p = 0
                for dx, dy in zip(dx8, dy8):
                    x = i + dx
                    y = j + dy
                    if x>=0 and x < self.width and y>=0 and y < self.height:
                        if isTaken[y, x] == False and self.clusters[j, i] != self.clusters[y, x]:
                            nr_p += 1

                if nr_p >= 2:
                    isTaken[j, i] = True
                    contours.append([j, i])

        for i in range(len(contours)):
            self.img[contours[i][0], contours[i][1]] = color
        cv2.imwrite("temp3.jpg", self.img)     

    def _findLocalMinimum(self, center):
        ##print(center)
        min_grad = self.FLT_MAX
        loc_min = center

        for i in range(center[0] - 1, center[0] + 2):
            for j in range(center[1] - 1, center[1] + 2): 
                ##print(i,j)
                c1 = self.labimg[j+1, i]
                c2 = self.labimg[j, i+1]
                c3 = self.labimg[j, i]
                if ((c1[0] - c3[0])**2)**0.5 + ((c2[0] - c3[0])**2)**0.5 < min_grad:
                    min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                    loc_min = [i, j]
        return loc_min

def _computeDist(self, ci, pixel, color):
        dc = (sum((self.centers[ci][:3] - color)**2))**0.5
        ds = (sum((self.centers[ci][3:] - pixel)**2))**0.5
        return ((dc/self.nc)**2 + (ds/self.ns)**2)**0.5    

img = cv2.imread('img11.jpeg')
nr_superpixels = int(sys.argv[1]) ##k
nc = int(sys.argv[2]) ##m
num_imgpix= img.shape[0]*img.shape[1] ##n

##print(img.shape[0],img.shape[1],img.shape[2])
# print(nr_superpixels)

step = int((num_imgpix/nr_superpixels)**0.5) ##s



slic = SLIC(img, step, nc)

slic.generateSuperPixels()
slic.createConnectivity()

cv2.imshow("cluster", slic.clusters.astype(np.uint8))
print (slic.clusters)
cv2.imshow("img", slic.img)
slic.displayContours((0, 0, 0))
cv2.imshow("fin", slic.img)

cv2.waitKey(0)
##cv2.imshow("final",slic.img)
cv2.imwrite("SLICimg.jpg", slic.img)