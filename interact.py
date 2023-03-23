from tkinter import ttk
import tkinter as tk
import cv2
from PIL import Image,ImageTk
from camera import camera
import param_calib as pc
import numpy as np
import json
import os


class Interact:
    def __init__(self,size):
        self.root= tk.Tk()
        self.root.title('Demo')
        self.root.geometry(size) # 这里的乘号不是 * ，而是小写英文字母 x

        with open('config.json') as fp:
            json_data = json.load(fp)[0]
            self.camera_index = json_data['index']
            self.max_pic_num = json_data['max_pic_num']
            self.casecade_mode = json_data['cascade']
            self.ellipse_threshold=json_data['ellipse_threshold']
            self.purkin_threshold=json_data['purkin_threshold']


        self.panel = tk.Label(self.root,relief=tk.SUNKEN)  # initialize image panel
        self.panel.place(x=10,y=10,height=480,width=640)
        

        self.panel_l = tk.Label(self.root,relief=tk.SUNKEN)  # initialize image panel
        self.panel_l.place(x=685,y=340,height=60,width=80)

        self.panel_r = tk.Label(self.root,relief=tk.SUNKEN)  # initialize image panel
        self.panel_r.place(x=780,y=340,height=60,width=80)

        self.panel_pl = tk.Label(self.root,relief=tk.SUNKEN)  # initialize image panel
        self.panel_pl.place(x=685,y=420,height=60,width=80)

        self.panel_pr = tk.Label(self.root,relief=tk.SUNKEN)  # initialize image panel
        self.panel_pr.place(x=780,y=420,height=60,width=80)

        lb1 = tk.Label(self.root, text='请输入0,1,2...选择相机')
        lb1.place(x=670, y=10, width=150, height=20)
        self.inp1 = tk.Entry(self.root)
        self.inp1.insert(0,self.camera_index)
        self.inp1.place(x=685, y=40, width=120, height=20)

        lb3 = tk.Label(self.root, text='请输入每组的数量')
        lb3.place(x=670, y=70, width=150, height=20)
        self.inp3 = tk.Entry(self.root)
        self.inp3.insert(0,self.max_pic_num)
        self.inp3.place(x=685, y=100, width=120, height=20)


        lb5 = tk.Label(self.root, text='请选择分类器')
        lb5.place(x=670, y=130, width=150, height=20)
        self.combvar = tk.StringVar()
        comb = ttk.Combobox(self.root,textvariable=self.combvar)
        comb['values']=('eye','glass_eye','yolov5_eye')
        comb.current(comb['values'].index(self.casecade_mode))
        comb.place(x=685, y=160, width=120, height=20)
        
        

        btn1 = tk.Button(self.root, text='开始', command=self.start)
        btn1.place(x=830, y=10, width=120, height=20)

        btn2 = tk.Button(self.root, text='停止', command=self.end)
        btn2.place(x=830, y=50, width=120, height=20)

        btn3 = tk.Button(self.root, text='采集', command=self.sample)
        btn3.place(x=830, y=90, width=120, height=20)

        btn4 = tk.Button(self.root, text='清空', command=self.clean)
        btn4.place(x=830, y=130, width=120, height=20)

        btn5 = tk.Button(self.root, text='截图', command=self.cat)
        btn5.place(x=830, y=170, width=120, height=20)
   
        self.ellipse_threshold_var = tk.IntVar()
        self.ellipse_threshold_var.set(self.ellipse_threshold)
        scl = tk.Scale(self.root,orient=tk.HORIZONTAL,length=200,from_=10,to=250,label='调整瞳孔分割的阈值',tickinterval=100,resolution=1,variable=self.ellipse_threshold_var)
        scl.bind('<ButtonRelease-1>',self.get_ellipse_threshold)
        scl.place(x=685, y=190, width=120, height=80)

        self.purkin_threshold_var = tk.IntVar()
        self.purkin_threshold_var.set(self.purkin_threshold)
        scl2 = tk.Scale(self.root,orient=tk.HORIZONTAL,length=200,from_=20,to=250,label='调整光斑分割的阈值',tickinterval=100,resolution=1,variable=self.purkin_threshold_var)
        scl2.bind('<ButtonRelease-1>',self.get_purkin_threshold)
        scl2.place(x=685, y=250, width=120, height=80)

        self.cat_index=0
        self.Isopen = False
        self.Issample = False
        self.isFinishOne = True
        self.point_index =0
        self.pic_count = 0
        self.max_pic_num=20
        self.info = []


    def key_event(self,event):
        s=event.keysym
        print(s)
        if s=='Right':
            self.point_index+=1
            self.isFinishOne = False
            self.Draw_bg()
        elif s=='space':
            self.winNew.attributes("-fullscreen", True)
        elif s=='q':
            self.quit()


    def del_file(self,path):
        for i in os.listdir(path) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
            file_data = path + "/" + i#当前文件夹的下面的所有东西的绝对路径
            if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
                os.remove(file_data)
            else:
                path(file_data)

    def get_ellipse_threshold(self,event):
        self.ellipse_threshold=self.ellipse_threshold_var.get()

    def get_purkin_threshold(self,event):
        self.purkin_threshold=self.purkin_threshold_var.get()

    def cat(self):
        if (self.camera.capture()):
            cv2.imwrite(str(self.cat_index)+'.bmp', self.camera.frame)
            self.cat_index+=1

    def clean(self):
        for i in range(9):
            self.del_file('sources/position'+str(i+1))

    def start(self):
        self.camera_index=int(self.inp1.get())
        self.max_pic_num=int(self.inp3.get())
        self.casecade_mode = self.combvar.get()
        self.camera = camera(self.casecade_mode)
        self.camera.start_camera(self.camera_index)   #摄像头
        self.info.append({
            "index":self.camera_index,
            "max_pic_num":self.max_pic_num,
            "cascade":self.casecade_mode,
            "ellipse_threshold": self.ellipse_threshold,
            "purkin_threshold": self.purkin_threshold
        })
        self.Save_json('config.json')
        self.Isopen= not self.Isopen

    def end(self):
        self.Isopen= not self.Isopen
        self.camera.stop_camera()
        self.camera_index = int(self.inp1.get())
        self.max_pic_num = int(self.inp3.get())
        self.casecade_mode = self.combvar.get()
        self.ellipse_threshold = self.ellipse_threshold_var.get()
        self.purkin_threshold = self.purkin_threshold_var.get()
        self.info.append({
            "index": self.camera_index,
            "max_pic_num": self.max_pic_num,
            "cascade": self.casecade_mode,
            "ellipse_threshold": self.ellipse_threshold,
            "purkin_threshold": self.purkin_threshold
        })
        self.Save_json('config.json')
    
    def quit(self):
        self.Issample=False
        self.point_index =0
        self.pic_count = 0
        self.info=[]
        self.winNew.destroy()
    
    def Draw_bg(self):
        background = np.zeros((self.winNew_height, self.winNew_width, 3), dtype=np.uint8)
        if not self.isFinishOne:
            x = int(self.winNew_width / 4 * (int((self.point_index - 1) % 3) + 1))
            y = int(self.winNew_height / 4 * (int((self.point_index - 1) / 3) + 1))
            cv2.circle(background, (x, y), 10, (0, 255, 0), -1)

        cv2image = cv2.cvtColor(background, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
        current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        self.lb2.imgtk1 = imgtk
        self.lb2.config(image=imgtk)

    def sample(self):
        self.winNew = tk.Toplevel(self.root)
        self.winNew_width =self.winNew.winfo_screenwidth()
        self.winNew_height =self.winNew.winfo_screenheight()
        self.winNew.geometry("%dx%d" %(self.winNew_width, self.winNew_height))

        self.lb2 = tk.Label(self.winNew)
        self.lb2.place(x=0,y=0,height=self.winNew_height,width=self.winNew_width)
        self.lb2.bind('<Key>',self.key_event)
        self.lb2.focus_set() 

        self.point_index =0
        self.isFinishOne = True
        self.Issample=True
        self.Draw_bg()


    def run(self):
        self.video_loop()
        self.root.mainloop()
    
    def Save_json(self,filename='data.json'):
        with open(filename, 'w') as f:
            json.dump(self.info, f)
        self.info=[]

    def show_img(self,img,panel,flag,threshold=0):
        if flag =="eye_pupil":
            ret, th = cv2.threshold(img.copy(),threshold, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            erosion = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            cv2image = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGBA)#转换颜色从GRAY到RGBA
            cv2image=cv2.resize(cv2image,(80,60))
        elif flag =="eye_purkin":
            ret, th = cv2.threshold(img.copy(), threshold, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            erosion = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            cv2image = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGBA)  # 转换颜色从GRAY到RGBA
            cv2image = cv2.resize(cv2image, (80, 60))
        elif flag =="face":
            cv2image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)#转换颜色从BGR到RGBA
            cv2image=cv2.resize(cv2image,(640,480))
        current_image = Image.fromarray(cv2image)#将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk1 = imgtk
        panel.config(image=imgtk)

    def video_loop(self):
        if self.Isopen:
            if self.point_index>9:
                self.quit()

            # 从摄像头读取照片
            if(self.camera.capture()):  
                # 同时检测到两只眼睛
                if(self.camera.detectEye()): 
                    #检测瞳孔
                    lellipse_parameter=pc.ellipse_calib(self.camera.lEyeImg,self.ellipse_threshold)
                    if not isinstance(lellipse_parameter,int):
                        #检测光斑
                        lSpotParameter = pc.SegmentSpotClib(self.camera.lEyeImg,self.purkin_threshold,lellipse_parameter)
                        #画图
                        self.camera.frame,lellipse_parameter,lSpotParameter = pc.Imshow(self.camera.frame,self.camera.lEyeParam,lellipse_parameter,lSpotParameter)
                        #在空间显示
                        self.show_img(self.camera.lEyeImg,self.panel_l,"eye_pupil",self.ellipse_threshold)
                        self.show_img(self.camera.lEyeImg,self.panel_pl,"eye_purkin",self.purkin_threshold)
                        #估计视线
                        

                    rellipse_parameter=pc.ellipse_calib(self.camera.rEyeImg,self.ellipse_threshold)
                    if not isinstance(rellipse_parameter,int):
                        rSpotParameter = pc.SegmentSpotClib(self.camera.rEyeImg,self.purkin_threshold,rellipse_parameter)
                        self.camera.frame,rellipse_parameter,rSpotParameter = pc.Imshow(self.camera.frame,self.camera.rEyeParam,rellipse_parameter,rSpotParameter)
                        self.show_img(self.camera.rEyeImg,self.panel_r,"eye_pupil",self.ellipse_threshold)
                        self.show_img(self.camera.rEyeImg,self.panel_pr,"eye_purkin",self.purkin_threshold)

                    if self.Issample and (not isinstance(lellipse_parameter,int)) and (not isinstance(rellipse_parameter,int)):
                        if not self.isFinishOne:
                            self.info.append({
                                self.pic_count:{
                                    'left_ellipse_parameter':lellipse_parameter,
                                    'left_purkin_parameter':lSpotParameter,
                                    'right_ellipse_parameter':rellipse_parameter,
                                    'right_purkin_parameter':rSpotParameter
                                },
                            })
                            cv2.imwrite('sources/position'+str(self.point_index)+'/'+str(self.point_index)+'_'+str(self.pic_count)+'l'+'.bmp',self.camera.lEyeImg)
                            cv2.imwrite('sources/position'+str(self.point_index)+'/'+str(self.point_index)+'_'+str(self.pic_count)+'r'+'.bmp',self.camera.rEyeImg)
                            self.pic_count+=1
                            if self.pic_count>=self.max_pic_num:
                                self.isFinishOne=True
                                self.pic_count = 0
                                self.Save_json('sources/position'+str(self.point_index)+'/'+'data.json')
                                self.Draw_bg()
                                if self.point_index>=9:
                                    self.quit()

                if self.camera.detectFace():
                    self.show_img(self.camera.frame,self.panel,"face")

        self.root.after(1, self.video_loop)
    

if __name__=="__main__":
    a=Interact(size='1024x500')
    a.run()