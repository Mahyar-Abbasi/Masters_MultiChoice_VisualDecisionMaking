<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch

class Visual_3class_DDM:
    
    def __init__(self,model,f_dict=None,frame_duration=0.1,dt=0.001):

        self.f_dict=f_dict    
        self.snn=copy.deepcopy(model)
        self.dt=dt
        self.frame_duration=frame_duration
        

    def fit_linear_features(self, train_dataset):

        sample_data=[train_dataset[i][0] for i in range(len(train_dataset))]
        labels=np.array([train_dataset[i][1] for i in range(len(train_dataset))])
        images_ordered=[torch.tensor(np.array(sample_data)[labels==0]),torch.tensor(np.array(sample_data)[labels==1]),torch.tensor(np.array(sample_data)[labels==2])]

        output_dict={"0":[], "1":[], "2":[]}
        self.snn.eval()
        for key in output_dict.keys():
            for i in range(len(images_ordered[int(key)])):
                data=images_ordered[int(key)][i]
                output_dict[key].append(self.snn(data)[0].cpu().numpy().reshape(15,-1))
        
            output_dict[key]=np.array(output_dict[key]) 
        
        
        f_dict={}
        
        for key in output_dict.keys():
        
            arr=np.array(output_dict[key])
            f_dict[key]=arr.mean(axis=0)[-1]
            f_dict[key]=f_dict[key]/np.linalg.norm(f_dict[key])

        self.f_dict=f_dict


    def input_current_calculator(self,input_sample):

        frame_duration=self.frame_duration
        dt=self.dt
        f0=self.f_dict["0"]
        f1=self.f_dict["1"]
        f2=self.f_dict["2"]
        
        self.snn.eval()
        out=self.snn(input_sample)[0].cpu().numpy().reshape(15,-1)
        time_steps=out.shape[0]
        
        
        current_values_array=np.zeros((time_steps,3))
    
        for t in range(time_steps):
            current_values_array[t]=np.array([out[t]@f0,out[t]@f1,out[t]@f2])
    
        
        time_array=np.arange(0,time_steps*frame_duration,dt)
    
        current_array=np.empty((len(time_array),3))
        for i,t in enumerate(time_array):
    
            cnt=int(t//frame_duration)
            current_array[i,:]=current_values_array[cnt,:]
    
        return current_array, time_array
    


    def plot_input_currents(self,input_sample,true_label=None,save=False):

        currents, tt= self.input_current_calculator(input_sample)
        true_class="?"
        if true_label is not None:
            true_class=str(true_label)
        
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title("Final output currents of the SNN module, true class = "+true_class)
        plt.plot(tt,currents[:,0],label="0")
        plt.plot(tt,currents[:,1],label="1")
        plt.plot(tt,currents[:,2],label="2")
        plt.legend()
        
        plt.subplot(1,2,2)

        plt.title("Input Filtered Image")
        plt.imshow(input_sample.sum(axis=0).sum(axis=0),cmap="binary")
        
        plt.tight_layout()

        if save:
            plt.savefig("currents_output")

        plt.show()

    #the model plots a equilateral triangle that each of its sides represents a decision boundary of a class(cow, dog and horse)

    def plot_equilateral_triangle(self,r_array,sigma_noise,r=1.0,save=None):
        # Compute side length a from inradius r

        # r=threshold

        a = 2 * np.sqrt(3) * r
    
        # Height of the triangle
        h = np.sqrt(3) * a / 2
    
        # Coordinates of the triangle's vertices centered at the origin
        # Place one vertex at the top, and calculate the other two symmetrically
        p1 = (0, 2*r)  # Top vertex (since 2r = 2h/3)
        p2 = (-a/2, -r)
        p3 = (a/2, -r)
    
        triangle = np.array([p1, p2, p3, p1])  # Close the triangle
    
        plt.figure(figsize=(10, 8))
        plt.plot(triangle[0:2, 0], triangle[0:2, 1], color="darkorange",label="Dog")
        plt.plot(triangle[1:3, 0], triangle[1:3, 1], color="limegreen",label="Horse")
        plt.plot(triangle[2:4, 0], triangle[2:4, 1], color="royalblue",label="Cow")
        
        plt.plot(r_array[:,0],r_array[:,1],alpha=0.7,color="red")    
        plt.gca().set_aspect('equal')
    
        plt.text(0.99,0.97,f"Sigma of the Internal Noise={sigma_noise}",transform=plt.gca().transAxes,
            ha='right', va='top', fontsize=13, color='brown')

     
    
        plt.grid(True)
        plt.title(f"Equilateral Triangular DDM (threshold = {r})")
    
        plt.legend(loc="upper left")   

        if save!=None:
            plt.savefig(str(save)+".jpg",dpi=300, bbox_inches='tight')

         
        plt.show()



    def decide_triangle_DDM(self,sample_input,threshold,internal_noise_level=0.1,tau=0.5,t_max=8,plot=False,save_name=None):

        current_array, time_array= self.input_current_calculator(sample_input)

        
        h_1=np.array([np.sqrt(3)/2,1/2])
        h_2=np.array([-np.sqrt(3)/2,1/2])
        h_3=np.array([0,-1])
    
        sigma=internal_noise_level
    
        t_array=np.arange(0,t_max,self.dt)
    

        currents=np.zeros((len(t_array),3))
        currents[0:len(time_array),:]=current_array
        currents[len(time_array):,0]=current_array[-1,0]
        currents[len(time_array):,1]=current_array[-1,1]
        currents[len(time_array):,2]=current_array[-1,2]
        
        r_array=[np.array([0,0])]
        decision_index=None
        decision_time=None
        confidence=None
        dt=self.dt
        
        for i,t in enumerate(t_array):
                 
            s=currents[i,0]*h_1+currents[i,1]*h_2+currents[i,2]*h_3
            
            r_array.append(-r_array[i]*dt/tau+r_array[i]+dt*s+sigma*np.sqrt(dt)*np.random.normal(size=(2)))
    
            if r_array[i+1][1]>-np.sqrt(3)*r_array[i+1][0]+2*threshold:
                decision_index=0
                decision_time=t
                d1=np.abs(np.sqrt(3)*r_array[i+1][0]-r_array[i+1][1]+2*threshold)/2
                d2=np.abs(r_array[i+1][1]+threshold)
                confidence=min(d1,d2)/(1.5*threshold)
                break
    
            if r_array[i+1][1]>np.sqrt(3)*r_array[i+1][0]+2*threshold:
                decision_index=1
                decision_time=t
                d1=np.abs(-np.sqrt(3)*r_array[i+1][0]-r_array[i+1][1]+2*threshold)/2
                d2=np.abs(r_array[i+1][1]+threshold)
                confidence=min(d1,d2)/(1.5*threshold)
                break
    
            if r_array[i+1][1]<-threshold:
                decision_index=2
                decision_time=t
                d1=np.abs(np.sqrt(3)*r_array[i+1][0]-r_array[i+1][1]+2*threshold)/2
                d2=np.abs(-np.sqrt(3)*r_array[i+1][0]-r_array[i+1][1]+2*threshold)/2
                confidence=min(d1,d2)/(1.5*threshold)
                break
    
    
        r_array=np.array(r_array)
    
        if plot==True:
            self.plot_equilateral_triangle(r_array,sigma,r=threshold,save=save_name)

        if decision_index==None:
            decision_index=np.random.choice([0,1,2])
            decision_time=t_max
            confidence=0
    

=======
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch

class Visual_3class_DDM:
    
    def __init__(self,model,f_dict=None,frame_duration=0.1,dt=0.001):

        self.f_dict=f_dict    
        self.snn=copy.deepcopy(model)
        self.dt=dt
        self.frame_duration=frame_duration
        

    def fit_linear_features(self, train_dataset):

        sample_data=[train_dataset[i][0] for i in range(len(train_dataset))]
        labels=np.array([train_dataset[i][1] for i in range(len(train_dataset))])
        images_ordered=[torch.tensor(np.array(sample_data)[labels==0]),torch.tensor(np.array(sample_data)[labels==1]),torch.tensor(np.array(sample_data)[labels==2])]

        output_dict={"0":[], "1":[], "2":[]}
        self.snn.eval()
        for key in output_dict.keys():
            for i in range(len(images_ordered[int(key)])):
                data=images_ordered[int(key)][i]
                output_dict[key].append(self.snn(data)[0].cpu().numpy().reshape(15,-1))
        
            output_dict[key]=np.array(output_dict[key]) 
        
        
        f_dict={}
        
        for key in output_dict.keys():
        
            arr=np.array(output_dict[key])
            f_dict[key]=arr.mean(axis=0)[-1]
            f_dict[key]=f_dict[key]/np.linalg.norm(f_dict[key])

        self.f_dict=f_dict


    def input_current_calculator(self,input_sample):

        frame_duration=self.frame_duration
        dt=self.dt
        f0=self.f_dict["0"]
        f1=self.f_dict["1"]
        f2=self.f_dict["2"]
        
        self.snn.eval()
        out=self.snn(input_sample)[0].cpu().numpy().reshape(15,-1)
        time_steps=out.shape[0]
        
        
        current_values_array=np.zeros((time_steps,3))
    
        for t in range(time_steps):
            current_values_array[t]=np.array([out[t]@f0,out[t]@f1,out[t]@f2])
    
        
        time_array=np.arange(0,time_steps*frame_duration,dt)
    
        current_array=np.empty((len(time_array),3))
        for i,t in enumerate(time_array):
    
            cnt=int(t//frame_duration)
            current_array[i,:]=current_values_array[cnt,:]
    
        return current_array, time_array
    


    def plot_input_currents(self,input_sample,true_label=None,save=False):

        currents, tt= self.input_current_calculator(input_sample)
        true_class="?"
        if true_label is not None:
            true_class=str(true_label)
        
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title("Final output currents of the SNN module, true class = "+true_class)
        plt.plot(tt,currents[:,0],label="0")
        plt.plot(tt,currents[:,1],label="1")
        plt.plot(tt,currents[:,2],label="2")
        plt.legend()
        
        plt.subplot(1,2,2)

        plt.title("Input Filtered Image")
        plt.imshow(input_sample.sum(axis=0).sum(axis=0),cmap="binary")
        
        plt.tight_layout()

        if save:
            plt.savefig("currents_output")

        plt.show()

    #the model plots a equilateral triangle that each of its sides represents a decision boundary of a class(cow, dog and horse)

    def plot_equilateral_triangle(self,r_array,sigma_noise,r=1.0,save=None):
        # Compute side length a from inradius r

        # r=threshold

        a = 2 * np.sqrt(3) * r
    
        # Height of the triangle
        h = np.sqrt(3) * a / 2
    
        # Coordinates of the triangle's vertices centered at the origin
        # Place one vertex at the top, and calculate the other two symmetrically
        p1 = (0, 2*r)  # Top vertex (since 2r = 2h/3)
        p2 = (-a/2, -r)
        p3 = (a/2, -r)
    
        triangle = np.array([p1, p2, p3, p1])  # Close the triangle
    
        plt.figure(figsize=(10, 8))
        plt.plot(triangle[0:2, 0], triangle[0:2, 1], color="darkorange",label="Dog")
        plt.plot(triangle[1:3, 0], triangle[1:3, 1], color="limegreen",label="Horse")
        plt.plot(triangle[2:4, 0], triangle[2:4, 1], color="royalblue",label="Cow")
        
        plt.plot(r_array[:,0],r_array[:,1],alpha=0.7,color="red")    
        plt.gca().set_aspect('equal')
    
        plt.text(0.99,0.97,f"Sigma of the Internal Noise={sigma_noise}",transform=plt.gca().transAxes,
            ha='right', va='top', fontsize=13, color='brown')

     
    
        plt.grid(True)
        plt.title(f"Equilateral Triangular DDM (threshold = {r})")
    
        plt.legend(loc="upper left")   

        if save!=None:
            plt.savefig(str(save)+".jpg",dpi=300, bbox_inches='tight')

         
        plt.show()



    def decide_triangle_DDM(self,sample_input,threshold,internal_noise_level=0.1,tau=0.5,t_max=8,plot=False,save_name=None):

        current_array, time_array= self.input_current_calculator(sample_input)

        
        h_1=np.array([np.sqrt(3)/2,1/2])
        h_2=np.array([-np.sqrt(3)/2,1/2])
        h_3=np.array([0,-1])
    
        sigma=internal_noise_level
    
        t_array=np.arange(0,t_max,self.dt)
    

        currents=np.zeros((len(t_array),3))
        currents[0:len(time_array),:]=current_array
        currents[len(time_array):,0]=current_array[-1,0]
        currents[len(time_array):,1]=current_array[-1,1]
        currents[len(time_array):,2]=current_array[-1,2]
        
        r_array=[np.array([0,0])]
        decision_index=None
        decision_time=None
        confidence=None
        dt=self.dt
        
        for i,t in enumerate(t_array):
                 
            s=currents[i,0]*h_1+currents[i,1]*h_2+currents[i,2]*h_3
            
            r_array.append(-r_array[i]*dt/tau+r_array[i]+dt*s+sigma*np.sqrt(dt)*np.random.normal(size=(2)))
    
            if r_array[i+1][1]>-np.sqrt(3)*r_array[i+1][0]+2*threshold:
                decision_index=0
                decision_time=t
                d1=np.abs(np.sqrt(3)*r_array[i+1][0]-r_array[i+1][1]+2*threshold)/2
                d2=np.abs(r_array[i+1][1]+threshold)
                confidence=min(d1,d2)/(1.5*threshold)
                break
    
            if r_array[i+1][1]>np.sqrt(3)*r_array[i+1][0]+2*threshold:
                decision_index=1
                decision_time=t
                d1=np.abs(-np.sqrt(3)*r_array[i+1][0]-r_array[i+1][1]+2*threshold)/2
                d2=np.abs(r_array[i+1][1]+threshold)
                confidence=min(d1,d2)/(1.5*threshold)
                break
    
            if r_array[i+1][1]<-threshold:
                decision_index=2
                decision_time=t
                d1=np.abs(np.sqrt(3)*r_array[i+1][0]-r_array[i+1][1]+2*threshold)/2
                d2=np.abs(-np.sqrt(3)*r_array[i+1][0]-r_array[i+1][1]+2*threshold)/2
                confidence=min(d1,d2)/(1.5*threshold)
                break
    
    
        r_array=np.array(r_array)
    
        if plot==True:
            self.plot_equilateral_triangle(r_array,sigma,r=threshold,save=save_name)

        if decision_index==None:
            decision_index=np.random.choice([0,1,2])
            decision_time=t_max
            confidence=0
    

>>>>>>> 065ae40458c78026aed8d427f3a1d0be34f785b2
        return decision_index,np.round(decision_time,4),np.round(confidence,3)   