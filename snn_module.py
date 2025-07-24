<<<<<<< HEAD
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms

use_cuda=False

class Kheradpisheh_SNN(nn.Module):
    def __init__(self):
        super(Kheradpisheh_SNN, self).__init__()

        self.conv1 = snn.Convolution(2, 32, 5, 0.8, 0.05)
        self.conv1_t = 6
        self.k1 = 10
        self.r1 = 2

        self.conv2 = snn.Convolution(32, 150, 5, 0.8, 0.05)
        self.conv2_t = 2
        self.k2 = 100
        self.r2 = 1

        self.conv3 = snn.Convolution(150, 60, 5, 0.8, 0.05)
        self.conv3_t = 1
        self.k3 = 60
        self.r3 = 1
        
        self.stdp1 = snn.STDP(self.conv1, (0.008, -0.001))
        self.stdp2 = snn.STDP(self.conv2, (0.03, -0.001))
        self.stdp3 = snn.STDP(self.conv3, (0.03, -0.001))
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0
        self.spk_cnt3 = 0
    
    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    def forward(self, input, max_layer=None):
        input = sf.pad(input.float(), (2,2,2,2), 0)
        if self.training:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
                self.save_data(input, pot, spk, winners)
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 3, 3, 1), (1,1,1,1))
            spk_in = sf.pointwise_inhibition(spk_in)
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
                self.save_data(spk_in, pot, spk, winners)
                return spk, pot

            spk_in = sf.pad(sf.pooling(spk, 3, 3, 1), (1,1,1,1))
            spk_in = sf.pointwise_inhibition(spk_in)
            pot = self.conv3(spk_in)
            spk, pot = sf.fire(pot, self.conv3_t, True)    
            if max_layer == 3:
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k3, self.r3, spk)
                self.save_data(spk_in, pot, spk, winners)
                return spk, pot    
            
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            spk_1=sf.pooling(spk, 3, 3, 1)
            pot = self.conv2(sf.pad(spk_1, (1,1,1,1)))
            spk, pot = sf.fire(pot, self.conv2_t, True)
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            spk_2 = sf.pooling(spk, 3, 3, 1)
            pot = self.conv3(sf.pad(spk_2, (1,1,1,1)))
            spk, pot = sf.fire(pot, self.conv3_t, True)
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            spk_3 = sf.pooling(spk, 3, 3, 1)

            return spk_3,spk_2,spk_1
    
    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 3:
            self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])    
            
    ##################################################
    
def train_unsupervise(network, data, layer_idx):
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        network(data_in, layer_idx)
        network.stdp(layer_idx)

def test(network, data, target, layer_idx):
    network.eval()
    ans = [None] * len(data)
    t = [None] * len(data)
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        output,_ = network(data_in, layer_idx).max(dim = 0)
        ans[i] = output.reshape(-1).cpu().numpy()
        t[i] = target[i]
    return np.array(ans), np.array(t)
=======
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms

use_cuda=False

class Kheradpisheh_SNN(nn.Module):
    def __init__(self):
        super(Kheradpisheh_SNN, self).__init__()

        self.conv1 = snn.Convolution(2, 32, 5, 0.8, 0.05)
        self.conv1_t = 6
        self.k1 = 10
        self.r1 = 2

        self.conv2 = snn.Convolution(32, 150, 5, 0.8, 0.05)
        self.conv2_t = 2
        self.k2 = 100
        self.r2 = 1

        self.conv3 = snn.Convolution(150, 60, 5, 0.8, 0.05)
        self.conv3_t = 1
        self.k3 = 60
        self.r3 = 1
        
        self.stdp1 = snn.STDP(self.conv1, (0.008, -0.001))
        self.stdp2 = snn.STDP(self.conv2, (0.03, -0.001))
        self.stdp3 = snn.STDP(self.conv3, (0.03, -0.001))
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0
        self.spk_cnt3 = 0
    
    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    def forward(self, input, max_layer=None):
        input = sf.pad(input.float(), (2,2,2,2), 0)
        if self.training:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
                self.save_data(input, pot, spk, winners)
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 3, 3, 1), (1,1,1,1))
            spk_in = sf.pointwise_inhibition(spk_in)
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
                self.save_data(spk_in, pot, spk, winners)
                return spk, pot

            spk_in = sf.pad(sf.pooling(spk, 3, 3, 1), (1,1,1,1))
            spk_in = sf.pointwise_inhibition(spk_in)
            pot = self.conv3(spk_in)
            spk, pot = sf.fire(pot, self.conv3_t, True)    
            if max_layer == 3:
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k3, self.r3, spk)
                self.save_data(spk_in, pot, spk, winners)
                return spk, pot    
            
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            spk_1=sf.pooling(spk, 3, 3, 1)
            pot = self.conv2(sf.pad(spk_1, (1,1,1,1)))
            spk, pot = sf.fire(pot, self.conv2_t, True)
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            spk_2 = sf.pooling(spk, 3, 3, 1)
            pot = self.conv3(sf.pad(spk_2, (1,1,1,1)))
            spk, pot = sf.fire(pot, self.conv3_t, True)
            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            spk_3 = sf.pooling(spk, 3, 3, 1)

            return spk_3,spk_2,spk_1
    
    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 3:
            self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])    
            
    ##################################################
    
def train_unsupervise(network, data, layer_idx):
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        network(data_in, layer_idx)
        network.stdp(layer_idx)

def test(network, data, target, layer_idx):
    network.eval()
    ans = [None] * len(data)
    t = [None] * len(data)
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        output,_ = network(data_in, layer_idx).max(dim = 0)
        ans[i] = output.reshape(-1).cpu().numpy()
        t[i] = target[i]
    return np.array(ans), np.array(t)
>>>>>>> 065ae40458c78026aed8d427f3a1d0be34f785b2
