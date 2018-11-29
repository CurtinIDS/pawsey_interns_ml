import keras
from keras.layers import Conv1D,Dropout,Conv2D,Input,Dense,MaxPool2D,Flatten,BatchNormalization,MaxPool1D
from keras.models import Model
import keras.backend as K
from keras.callbacks import EarlyStopping,TensorBoard
from tensorflow import set_random_seed
import tensorflow as tf


import numpy as np
from scipy.io import wavfile as wave
import scipy.optimize as opt
import glob2
import time
import pandas as pd
import math
import os



import PIL
from PIL import Image
from PIL import Image, ImageDraw
from shutil import copyfile




import plotly.offline as pyp
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact
import plotly.graph_objs as go
import plotly.offline as py

def Splom_plot(data):
    color_vals = [0  if cl==0 else 1 for cl in data['pred']]
    choice=['No diabetes','Pre-diabetic','Diabetic']
    textd=[choice[val] for val in data['pred']]

    traced = go.Splom(dimensions=[dict(label='Family', values=data["Family_history"]),
                                  dict(label='Fasting', values=data["Fasting"]),
                                  dict(label='Random', values=data['Random']),
                                  dict(label='Age', values=data['Age'])],
                      marker=dict(color=data['pred'],
                                  size=5,
                                  colorscale='Portland',#pl_colorscaled,
                                  line=dict(width=0.5,
                                            color='rgb(230,230,230)') ),
                      text=textd,
                      diagonal=dict(visible=False))
    axisd = dict(showline=False,
           zeroline=False,
           gridcolor='#fff',
           ticklen=4,
           titlefont=dict(size=13))
    title = "Scatterplot Matrix"

    layout = go.Layout(title=title,
                       dragmode='select',
                       width=900,
                       height=800,
                       autosize=False,
                       hovermode='closest',
                       plot_bgcolor='rgba(200,200,200, 0.95)',
                       xaxis1=dict(axisd),
                       xaxis2=dict(axisd),
                       xaxis3=dict(axisd),
                       xaxis4=dict(axisd),
                       yaxis1=dict(axisd),
                       yaxis2=dict(axisd),
                       yaxis3=dict(axisd),
                       yaxis4=dict(axisd))


    fig = dict(data=[traced], layout=layout)
    py.iplot(fig, filename='large')



def create_fake_images(number_of_meteors=800,
                       src_dir="meteors/background",
                       dest_dir="meteors",
                       tmp_dir="tmp"):
    '''
    Input: number_of_meteors, src_dir(meteors/background), dest_dir(meteors), tmp_dir(tmp)
    number_of_meters: total number of fake images that would be generated
    src_dir : Directory containing images without meteors
    dest_dir: Directory containing 2 image classes - i) without meteors ii) with meteors
    tmp_dir: Director used to store temporary files
  

    '''
    filter_w=200 #tile width
    filter_h=200 #tile height
    np.random.seed(0)
    def dist(A): #Computes distance between two points in px
        return(((A[0][0]-A[0][2])**2+(A[0][1]-A[0][3])**2)**0.5)

    count=0
    for each in os.listdir(src_dir):
        copyfile(src_dir+'/'+each,tmp_dir+'/'+str(count)+'.jpg')
        count+=1

    image_len=len(os.listdir(src_dir))
    for i in range(number_of_meteors):
        myrand=np.random.randint(0,image_len) #Open files with random background and w/o meteorites
        im=Image.open(tmp_dir+'/'+str(myrand)+'.jpg')
        width,height=im.size
        marker_x=width*2
        marker_y=height*2
        while ((marker_x+filter_w)>width):
            marker_x=int(np.random.rand()*width)
            while ((marker_y+filter_h)>height):
                marker_y=int(np.random.rand()*height)
        im=im.crop((marker_x,marker_y,marker_x+filter_w,marker_y+filter_h))
        draw = ImageDraw.Draw(im)
        A=[(np.random.rand()*im.size[0], np.random.rand()*im.size[1],
            np.random.rand()*im.size[0], np.random.rand()*im.size[1])]
        myrand=str(myrand)
        flag_present=False
        if (np.random.rand()>0.5): #Statistically, only half the samples would have meteorites
            if ((dist(A)>30) and (dist(A)<300)): #The lines shouldn't be too short or too long
                color_rnd=np.random.uniform(0.8,1.0)  #Brightness should be over certain threshold, lower the brightness -> harder to train, more resilient
                width_rand=np.random.choice([1,2])
                draw.line([A[0][0],A[0][1],A[0][2],A[0][3]], fill=int(color_rnd*250),width=width_rand)
                #myrand=myrand+'_y' #Add _y to files that contain  meteors
                flag_present=True
                del draw
        im.save(dest_dir+"/"+str((0,1)[flag_present])+'/'+str(i)+'_'+str(myrand)+'.jpg')



def diabetes_data(per_class=2000):
    #Fields Age, Family History, Fasting, Random, A1C, Diabetic[No,Pre,Diabetic]
    np.random.seed(2)
    unbalanced=True
    vals={}
    data=[]
    pred=[]

    #Three buckets : non-diabetic, pre-diabetic, diabetic
    vals[0]=0
    vals[1]=0
    vals[2]=0

    #Number per class
    total_number=2000

    #Predictors preassignment
    pred=np.zeros([total_number*3,3])
    while unbalanced:

        age=np.random.randint(20,54)
        diab=np.random.choice([0,1])
        is_america=np.random.choice([0,1])
        try:

            Fasting=np.random.randint(3.5,
                                      (5+(1.5*diab))*(age/34))+2*np.random.random()
        except:
            t=0
        Random=np.random.randint(Fasting,12)+np.random.random()
        is_diab=-1
        if Random>Fasting:
            #Non-diabetic
            if Fasting<5.6 and Random<7.8:
                is_diab=0
            #Prediabetic
            elif Fasting>5.6 and Fasting<7 and Random>7.8 and Random<11.0 and age>26:
                is_diab=1
            #Diabetic
            elif Fasting>7.0 and Random>11 and age>30:
                is_diab=2
            if is_diab!=-1:
                if is_diab in vals:
                    if  vals[is_diab]<total_number:
                        data.append([Fasting,Random,diab,age,is_america])
                        pred[len(data)-1,is_diab]=1
                        vals[is_diab]=vals[is_diab]+1

            try:
                if vals[0]>=total_number and vals[1]>=total_number and vals[2]>=total_number:
                    unbalanced=False
            except:
                ignore=1
    data=np.array(data)
    pred=np.array(pred)
    return data,pred




def escalator_data(directory="escalator_faults"):
    types=['0','1']
    file=glob2.glob(directory+"/1/*")
    data=[]
    pred=[]
    for every in types:
        file=glob2.glob(directory+"/"+str(every)+"/*")
        for each in file:
            sampling_fs_fault,fault_data=wave.read(each)
            fault_data=fault_data[0::3]
            tmp=np.array(fault_data).reshape([-1,400])
            data.append(tmp)
            pred.append(np.ones([tmp.shape[0],1])*int(every))
    data=np.vstack(data)
    pred=np.vstack(pred)
    data=data.reshape([data.shape[0],data.shape[1],1])
    data=(data-np.mean(data))/np.std(data)
    return data,pred

def Auto_Conv1D(shape=(400,1),layers=4,filters=2,batch_norm=True,max_pooling=True):
    '''
    Input arguments
    shape: input dimensions (default: (400,1)) 
    layers: Number of convolutional layers (default: 4), 
    filters: Number of filters per layer (default: 2),
    batch_norm: Enable normalization (default: True),
    max_pooling: Enable maxpooling after each conv (default: True)
    '''
    np.random.seed(0)
    inp=Input(shape=shape)
    out=[]
    for i in range(layers):
        out=Conv1D(filters,3,activation='relu')((inp,out)[i!=0])
        if batch_norm:
            out=BatchNormalization()(out)
        if max_pooling:
            out=MaxPool1D(2)(out)
    out=Flatten()(out)
    for i in range(3):
        out=Dense((3,1)[i==2],
                  activation=('relu','sigmoid')[i==2])(out)
    mod=Model(inp,out)
    return mod


def Infer(Num_filters,Num_layers,max_pool=False):
    global X,parameters,max_pooling
    inp=Input(shape=(1024,1024,3))
    out=[]
    N=Num_layers
    #Conv-Maxpool-fully connected deck
    for i in range(N):
        out=Conv2D(Num_filters,(3,3),
                   activation='relu')((out,inp)[i==0])
        if max_pool:
            out=MaxPool2D(2)(out)
    out=Flatten()(out)

    for i in range(5):
        out=Dense((2,12)[i!=4],activation=('sigmoid','relu')[i!=3])(out)

    model=Model(inp,out)
    
    #how many trainable parameters?
    params = int(np.sum([K.count_params(i) for i in set(model.trainable_weights)]))

    #For plotting
    parameters.append(params)
    max_pooling.append(int(max_pool))
    print ("Parameters:",params)
    
def Reset(b):
    global X,parameters,max_pooling
    X=[]
    parameters=[]
    max_pooling=[]

pyp.init_notebook_mode()
global X,parameters,max_pooling
Reset(0)
