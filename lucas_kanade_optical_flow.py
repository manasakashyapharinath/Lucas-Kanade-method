'''
Created on Oct 13, 2016

@author: Kashyap
'''
  
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.reshape import pivot
from com.pythonAssignment0.AssignTwoProbOne import Imagex
from dask.array.wrap import zeros


def lucasKanade(image,image2,number):
#Please make sure that the image is in the same path as that of your project

    Ione = np.array(Image.open(image).convert('L'))
    Itwo=  np.array(Image.open(image2).convert('L'))
    plt.imshow(Ione)
    plt.show()
    sigma = .5
    mid =  10
    result = np.zeros( 2*mid + 1 )
    Gauss = [(1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)]  


    Gaussx = zeros(len(Gauss))
    for i in range(1,len(Gauss)):
        Gaussx[i] = Gauss[i] - Gauss[i-1]
        
    n, m = Ione.shape
    # Finding the prime spectrum in X and Y direction
    Imagex =np. zeros((n,m))
    Imagey = np.zeros((n,m))
    Imaget=np.zeros((n,m))


    for i in range(n):
        Imagex[i,:] = np.convolve(Ione[i,mid:m - mid] , Gaussx)
 
 
 
    plt.imshow(Imagex)
    plt.show()
        #figure()
        #plt.imshow(Imagex) 
    
    for j in range(m):
        Imagey[:,j] = np.convolve(Ione[mid:n - mid,j] , Gaussx)

        #figure()
        #plt.imshow(Imagey)
        #computation of Intensity difference matrix wrt 't' (time)
    for i in range(n):
        for j in range(m):
            Imaget[i,j]=np.absolute((Ione[i,j])-(Itwo[i,j]))

    #figure()
    #plt.imshow(Imaget)
    #plt.show()
    count=0
    n,m=Ione.shape
    Ix=np.zeros((3,3))
    Iy=np.zeros((3,3))
    It=np.zeros((3,3))
    lucasKanRHS=np.zeros((9,1))
    lucasKanLHS=np.zeros((9,2))
    Ixsq=np.zeros((3,3))
    Iysq=np.zeros((3,3))
    Ixy=np.zeros((3,3))
    Ixt=np.zeros((3,3))
    Iyt=np.zeros((3,3))
    uMat=np.zeros((n,m))
    vMat=np.zeros((n,m))

    plt.gray()
    plt.imshow(Ione)
    eigarr=[]
    eigarr2=[]
    finalMat=np.zeros((n,m))
    for i in range(1,n):
        for j in range(1,m):
            # This is for the 3X3 window in the image    
            Ix=Imagex[i-1:i+2, j-1:j+2]
            Iy=Imagey[i-1:i+2, j-1:j+2]
            It=Imaget[i-1:i+2, j-1:j+2]
            xshape,yshape=Ix.shape
    
        #For every pixel in the 3x3 matrix,  the lucas kanade is computed. A is the summation of all the pixel in the 3X3 window.
        #Then corresponding frames for I(t) (difference btw pixel intensity of Image1 and Image2 is done)
        #Then it is multiplied with the A(T) (transpose) on both sides A*A(t)*(u,v)=-A(t)*t
        #u and v are the velocity vectors of the matrix. using least square method, we get a 2X1 matrix consisiting of u and v
        
            for k in range(xshape):
                for h in range(yshape):
               
                    lucasKanLHS[0:9,0:2]=[[Ix[k,h],Iy[k,h]]]
                    lucasKanRHS[0:9,0:1]=[[It[k,h]]]
                #print(lucasKanLHS.shape)
        #Eigen values are calculated to find the corners. A*A(T) is used to find thee corners       
            eig1,eig2=(np.linalg.eigvals(np.dot(np.transpose(lucasKanLHS),lucasKanLHS)))
            eigarr.append(eig2)
            eigarr2.append(eig1)
                
                
                
            if(eig1 > 400000 and eig1<405000):
                if(eig2>2e-11):
                    #if(eig2>5):
                    
                #Computation of least sqaure method:= A*A(t)*(u,v)=-A(t)*t
                    uMa=np.dot( (np.linalg.pinv(np.dot(np.transpose(lucasKanLHS),lucasKanLHS))),(np.dot(np.transpose(lucasKanLHS),lucasKanRHS)))[0]
                    vMa=np.dot( (np.linalg.pinv(np.dot(np.transpose(lucasKanLHS),lucasKanLHS))),(np.dot(np.transpose(lucasKanLHS),lucasKanRHS)))[1]
                    print("eigen values")
                    print(eig1,eig2)
                # to plot velocity vectors on the image, We use Quiver from matplotlib
                    plt.quiver(i,j,uMa,vMa,color='y')
                #print(uMat,vMat)
                #print(uMat,vMat)
                #plt.plot(i,j,'r->')
                
#x=np.linspace(0,1,n)
#y=np.linspace(0,1,m)
    print(max(eigarr))
    print(max(eigarr2))
    print("Yes done")
    plt.savefig("OutputLucasKanadeProb1b"+str(number)+"jpg")
    plt.show() 
 
 
 #The lucas Kanade method is applied to other images as well.

lucasKanade('basketball1.png', 'basketball2.png',1)
lucasKanade('grove1.png', 'grove2.png',2)
lucasKanade('teddy1.png', 'teddy2.png',3)


        

        
        
        
               
        