import numpy as np
import pandas as pd
from scipy import linalg
import math
#import sklearn
from sklearn.model_selection import train_test_split


def scaledata(datain,minval,maxval):
    (a,b)=np.shape(datain)
    dataout=np.arange(a*b)
    #dataout=np.zeros(a,b)
    dataout=datain-np.amin(datain)
    dataout=(dataout/(np.amax(dataout)-np.amin(dataout))*(maxval-minval))
    dataout=dataout+minval
    return dataout

def divide_data(trainRatio,x,y):
    l=[]

    test_size=1-trainRatio
    #print (np.shape(x))
    #print(np.shape(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    l=[x_train,y_train,x_test,y_test]
    #print (l)
    return l

def tribas(X):

#    for i in np.nditer(X):
#        print (i)
#        if(i>=-1 and i<=1):
#            i=1-i
#        else:
#            i=0
#        print (i==X)
    X=np.clip(1.0-np.fabs(X),0.0,1.0)
    return X
def TanH(X):
    Y=np.tanh(X)
    return Y

def elm_std(Xdata,Ydata,Opts,number_neurons,C,net):

    N1=np.amin(Ydata)
    N2=np.amax(Ydata)
    Xdata=scaledata(Xdata,0,1)
    if(Opts['fixed']==1):
        X=Xdata[0:36,:]
        Xts=Xdata[37:,:]
        Y=Ydata[0:36,:]
        Yts=Ydata[37:,:]
    else:
        l=[]
        x=np.asmatrix(Xdata)
        y=np.asmatrix(Ydata)
    #    print (np.shape(Xdata))
    #    print (np.shape(Ydata))
        l=divide_data(Opts['Tr_ratio'],Xdata,Ydata)
        X=l[0]
        Y=l[1]
        Xts=l[2]
        Yts=l[3]

    a=np.shape(X)
    Nsamples=a[0]
    Nfea=a[1]
    net['X']=X
    net['Y']=Y
    net['Xts']=Xts
    net['Yts']=Yts
    C1=C
    bias=np.random.rand(number_neurons,1)
    ind=np.ones((1,Nsamples))
    biasMatrix=np.ones((number_neurons, Nsamples))
    input_weights=np.random.rand(number_neurons,Nfea)*2-np.ones((number_neurons,Nfea))
    tempH=np.add(np.dot(input_weights,np.transpose(X)),biasMatrix)

    if(Opts['ActivationFunction']=="tribas"):
        H=tribas(tempH)
        b=np.shape(H)
    elif(Opts['ActivationFunction']=="tanh"):
        H=TanH(tempH)
        b=np.shape(H)
    elif(Opts['ActivationFunction']=="sin"):
        H=np.sin(tempH)
        b=np.shape(H)
    elif(Opts['ActivationFunction']=="cos"):
        H=np.cos(tempH)
        b=np.shape(H)
    elif(Opts['ActivationFunction']=="hardlim"):
        H=np.array(tempH>0)
        b=np.shape(H)
    elif(Opts['ActivationFunction']=="sigmoid"):
        H=(1)/(1+np.exp(-tempH))
        b=np.shape(H)


        #C=np.arange(b[1]*b[1])
        #C=np.reshape(C,(b[1],b[1]))
    #    print (b)
    #    print (np.shape(C))
        #C=C1*C

    if(Opts['Regularisation']==1):
        if(number_neurons<Nsamples):
        #    B=np.dot(linalg.lstsq(np.dot(np.add(np.divide(np.eye(b[0],b[0]),C),H),np.transpose(H)),H),Y)
            B=np.dot(np.dot(np.linalg.pinv(np.add(np.dot(H,np.transpose(H)),C*np.eye(b[0],b[0]))),H),Y)
        else:
            B=np.dot(np.transpose(np.dot(np.linalg.pinv(np.add(np.dot(np.transpose(H),(H)),C*np.eye(b[1],b[1]))),np.transpose(H))),Y)

    #        x=np.add((np.eye(b[1],b[1])/C),np.dot(np.transpose(H),H))
    #        y=linalg.solve(x,np.transpose(H))
    #        B=np.dot(np.transpose(linalg.solve(x,np.transpose(H))),Y)
    else:
        B=np.dot(np.linalg.pinv(np.transpose(H)),Y)

    net['OW']=B
    Y_hat=np.dot(np.transpose(H),B)
    Nsamples_test=np.shape(Xts)[0]
    bias=np.random.rand(number_neurons,1)
    ind=np.ones((1,Nsamples))
    biasMatrix_test=np.ones((number_neurons, Nsamples_test))
    tempH_test=np.add(np.dot(input_weights,np.transpose(Xts)), biasMatrix_test)

    if(Opts['ActivationFunction']=="tribas"):
        H_test=tribas(tempH_test)
    elif(Opts['ActivationFunction']=="tanh"):
        H_test=TanH(tempH_test)
    elif(Opts['ActivationFunction']=="sin"):
        H_test=np.sin(tempH_test)
    elif(Opts['ActivationFunction']=="cos"):
        H_test=np.cos(tempH_test)
    elif(Opts['ActivationFunction']=="hardlim"):
        H_test=np.array(tempH_test>0)
    elif(Opts['ActivationFunction']=="sigmoid"):
        H_test=(1)/(1+np.exp(-tempH_test))


    #print (math.sqrt(Y.mean()))
    #print (H_test)

#        print(np.shape(y))
#        print(np.shape(Y))
    Yts_hat=np.dot(np.transpose(H_test),B)
    #print (Yts,Yts_hat)
    TrAccuracy=math.sqrt(np.square(np.subtract(Y,Y_hat)).mean())
    TsAccuracy=math.sqrt(np.square(np.subtract(Yts,Yts_hat)).mean())
    #print (TrAccuracy,TsAccuracy)
    net['Y_hat']=Y_hat
    net['Yts_hat']=Yts_hat
    net['Y']=Y
    net['Yts']=Yts
    net['min']=N1
    net['max']=N2
    net['Opts']=Opts
#    print (Opts['ActivationFunction'],TrAccuracy,TsAccuracy)
    if(TrAccuracy<net['training_accuracy']):
        net['training_accuracy']=TrAccuracy

        net['OptimalCTr']=math.log(C,10)
    #if(TsAccuracy<net['testing_accuracy']):
        net['testing_accuracy']=TsAccuracy
        net['ActivationFunction']=Opts['ActivationFunction']
        net['Optimalneurons']=number_neurons
    #net['testing_accuracy']=TsAccuracy
    net['tempH']=tempH
    a=net['Opts']
    a['ActivationFunction']=Opts['ActivationFunction']

    return net

file='DataSet2.csv'
data = np.loadtxt(file, delimiter=",")
net={'X':None,'Y':None,'Xts':None,'Yts':None,'Yts':None,'Yts_hat':None,'Y':None,'Y_hat':None,'min':None,'max':None,'Opts':None,'OW':None,'training_accuracy':10000000000,'testing_accuracy':100000000,'tempH':None,'OptimalCTr':None,'ActivationFunction':None,'Optimalneurons':None}

Opts={'fixed':None,'Tr_ratio':None,'ActivationFunction':["tribas","sin","cos","hardlim","sigmoid","tanh"],'Regularisation':None}
Xdata=data[:,0:4]
Ydata=data[:,4:5]
#print (np.shape(Ydata))
Opts['fixed']=1
number_neurons=200
Opts['Tr_ratio']=0.8
#Opts['ActivationFunction']=
Opts['Regularisation']=1

#print (np.shape(Xdata))
#print (Ydata)
for i in Opts['ActivationFunction']:
    Opts['ActivationFunction']=i
    for j in range(30,301,5):
        number_neurons=j
        k=-16
        while(k<=16):


            C=pow(10,k)
            k+=0.2

            net=elm_std(Xdata,Ydata,Opts,number_neurons,C,net)

print ("X size: ",np.shape(net['X']))
print ("Y size: ",np.shape(net['Y']))
print ("Xts size: ",np.shape(net['Xts']))
print ("Yts size: ",np.shape(net['Yts']))
print ("OW size: ",np.shape(net['OW']))
print ("Y_hat size: ",np.shape(net['Y_hat']))
print ("Yts_hat size: ",np.shape(net['Yts_hat']))

print ("Optimal C : 10^",net['OptimalCTr'] )
print ("Optimal number_neurons: ",net['Optimalneurons'])
print ("ActivationFunction:", net['ActivationFunction'])
print ("Min: ",net['min'])
print ("Max: ", net['max'])
print ("Training accuracy" ,net['training_accuracy'])
print ("Testing accuracy" ,net['testing_accuracy'])
print ("tempH size: ",np.shape(net['tempH']))
