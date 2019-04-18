#coding=utf-8
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2;
import numpy as np;
import os;
from math import *

index = {"深":0,"秦":1,"京":2,"海":3,"成":4,"南":5,"杭":6,"苏":7,"松":8,"0": 9, "1": 10, "2":11, "3": 12, "4": 13, "5": 14,
         "6": 15, "7": 16, "8": 17, "9": 18, "A": 19, "B": 20, "C": 21, "D": 22, "E": 23, "F": 24, "G": 25, "H": 26,
         "J": 27, "K": 28, "L": 29, "M": 30, "N": 31, "P": 32, "Q": 33, "R": 34, "S": 35, "T": 36, "U": 37, "V": 38,
         "W": 39, "X": 40, "Y": 41, "Z": 42};

chars = [ "深","秦","京","海","成","南","杭","苏","松", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ];

def AddSmudginess(img, Smu):
    rows = r(Smu.shape[0] - 50)

    cols = r(Smu.shape[1] - 50)
    adder = Smu[rows:rows + 50, cols:cols + 50];
    adder = cv2.resize(adder, (50, 50));
    #   adder = cv2.bitwise_not(adder)
    img = cv2.resize(img,(50,50))
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(adder, img)
    img = cv2.bitwise_not(img)
    return img

def rot(img,angel,shape,max_angel):
    """
添加仿射畸变
    """
    size_o = [shape[1],shape[0]]

    size = (shape[1]+ int(shape[0]*cos((float(max_angel )/180) * 3.14)),shape[0])


    interval = abs( int( sin((float(angel) /180) * 3.14)* shape[0]));

    pts1 = np.float32([[0,0]         ,[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):

        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2);
    dst = cv2.warpPerspective(img,M,size);

    return dst;

def rotRandrom(img, factor, size):
#添加透视畸变
    shape = size;
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [ r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2);
    dst = cv2.warpPerspective(img, M, size);
    return dst;



def tfactor(img):
#增加饱和度光照的噪声
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);

    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2);
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7);
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8);

    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);
    return img

def random_envirment(img,data_set):
#添加自然环境的噪声
    index=r(len(data_set))
    env = cv2.imread(data_set[index])

    env = cv2.resize(env,(img.shape[1],img.shape[0]))

    bak = (img==0);
    bak = bak.astype(np.uint8)*255;
    inv = cv2.bitwise_and(bak,env)
    img = cv2.bitwise_or(inv,img)
    return img

def GenCh(f,val):
#生成中文字符
    img=Image.new("RGB", (45,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 3),val,(0,0,0),font=f)
    img =  img.resize((23,70))
    A = np.array(img)

    return A
def GenCh1(f,val):
#生成英文字符
    img=Image.new("RGB", (23,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 2),val.decode('utf-8'),(0,0,0),font=f)
    A = np.array(img)
    return A
def AddGauss(img, level):
#添加高斯模糊
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1));


def r(val):
    return int(np.random.random() * val)

def AddNoiseSingleChannel(single):
#添加高斯噪声
    diff = 255-single.max();
    noise = np.random.normal(0,1+r(6),single.shape);
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise= diff*noise;
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst

def addNoise(img,sdev = 0.5,avg=10):
    img[:,:,0] =  AddNoiseSingleChannel(img[:,:,0]);
    img[:,:,1] =  AddNoiseSingleChannel(img[:,:,1]);
    img[:,:,2] =  AddNoiseSingleChannel(img[:,:,2]);
    return img;


class GenPlate:


    def __init__(self,fontCh,fontEng,NoPlates):
        self.fontC =  ImageFont.truetype(fontCh,43,0);
        self.fontE =  ImageFont.truetype(fontEng,60,0);
        self.img=np.array(Image.new("RGB", (284,70),(255,255,255)))
        self.bg  = cv2.resize(cv2.imread("./images/template.bmp"),(284,70));
        self.smu = cv2.imread("./images/smu2.jpg");
        self.noplates_path = [];
        for parent,parent_folder,filenames in os.walk(NoPlates):
            for filename in filenames:
                path = parent+"/"+filename;
                self.noplates_path.append(path);


    def draw(self,val):
        offset= 2 ;
        self.img[0:70,offset+8:offset+8+23]= GenCh(self.fontC,val[0]);
        self.img[0:70,offset+8+23+6:offset+8+23+6+23]= GenCh1(self.fontE,val[1]);
        for i in range(7):
            base = offset+8+23+6+23+17 +i*23 + i*6 ;
            self.img[0:70, base:base+23]= GenCh1(self.fontE,val[i+2]);
        return self.img
    def generate(self,text):
        if len(text) == 11:
            fg = self.draw(text.decode(encoding="utf-8"));
            fg = cv2.bitwise_not(fg);
            com = cv2.bitwise_or(fg,self.bg);
            com = rot(com,r(60)-30,com.shape,30);
            com = rotRandrom(com,10,(com.shape[1],com.shape[0]));
            #com = AddSmudginess(com,self.smu)

            com = tfactor(com)
            com = random_envirment(com,self.noplates_path);
            com = AddGauss(com, 1+r(4));
            com = addNoise(com);


            return com
    def genPlateString(self,pos,val):
        plateStr = "";
        box = [0,0,0,0,0,0,0,0,0];
        if(pos!=-1):
            box[pos]=1;
        for unit,cpos in zip(box,xrange(len(box))):
            if unit == 1:
                plateStr += val
            else:
                if cpos == 0:
                    plateStr += chars[r(9)]
                elif cpos == 1:
                    plateStr += chars[19+r(24)]
                else:
                    plateStr += chars[9 + r(34)]

        return plateStr;

    def genBatch(self, batchSize,pos,charRange, outputPath,size):
        if (not os.path.exists(outputPath)):
            os.mkdir(outputPath)
        for i in xrange(batchSize):
                plateStr = G.genPlateString(-1,-1)
                img =  G.generate(plateStr);
                img = cv2.resize(img,size);
                cv2.imwrite(outputPath + "/" + str(i).zfill(2) + ".jpg", img);
                with open(outputPath + "/generate-data-label.txt", 'a+') as obj:
                     line = plateStr + ',  ' + str(i).zfill(2)+'.jpg' + '\r\n'
                     obj.write(line)


G = GenPlate("./font/platech.ttf",'./font/platechar.ttf',"./NoPlates")

G.genBatch(100000,2,range(31,65),"./demo",(356,70))
