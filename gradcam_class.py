import cv2
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import PIL

class GradCam:
    def __init__(self, model, target):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.handlers = []
        self.target = target
        self._get_hook()
        self.idx = 0

    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad)

        def _store_grad(grad):  
            self.gradient = self.reshape_transform(grad)

        output_grad.register_hook(_store_grad)
    
    def _get_hook(self):
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_forward_hook(self._get_grads_hook)

    def reshape_transform(self, tensor, height=14, width=14):
        print(tensor.shape)
        result = tensor[:, :, :].reshape(8,
                                          height, 
                                          width, 
                                          tensor.size(2))

        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def __call__(self, inputs):

        self.model.zero_grad()
        output = self.model(inputs) 
        
        index = output.squeeze(0).argmax().item()
        target = output[0][index]
        print("target:", target)
        target.backward()

        print(self.gradient.shape)
        data  = []
        for idx in range(self.gradient.shape[0]):
            if idx != self.gradient.shape[0] -1:
                gradient = self.gradient[idx:idx+1].cpu().data
                print("gradient:", gradient.shape) #[8, 1024, 14, 14])
                weight = torch.mean(gradient, axis=(0, 2, 3))
                print("weight:", weight.shape)
                feature = self.feature[idx:idx+1].cpu().data
                print("feature:", feature.shape)
            else:
                gradient = self.gradient[idx:].cpu().data
                print("gradient:", gradient.shape) #[8, 1024, 14, 14])
                weight = torch.mean(gradient, axis=(0, 2, 3))
                print("weight:", weight.shape)
                feature = self.feature[idx:].cpu().data
                print("feature:", feature.shape)
            # cam = feature * weight[:, np.newaxis, np.newaxis]
            for i in range(feature.shape[1]):
                feature[0, i, :, :] *= weight[i]
            # print("cam:", cam.shape)
            heatmap = torch.mean(feature, dim=1).squeeze()

            # relu on top of the heatmap
            heatmap = F.relu(heatmap)
            
            # normalize the heatmap
            heatmap /= torch.max(heatmap)
            heatmap = np.expand_dims(heatmap.detach().numpy(), axis = 0)
            if idx == 0:
                data = heatmap
            else:
                data = np.concatenate((data,heatmap), axis=0)
            # print("heap_map:", heatmap.shape)
        # print(data.shape) (8,14,14)
            
            # draw the heatmap
            # plt.matshow(heatmap.squeeze())
            # plt.savefig(f"heatmap{idx}.png")
            # print("Done:", idx)
            # plt.imshow(heatmap.detach())
            # plt.show()
        
        return data