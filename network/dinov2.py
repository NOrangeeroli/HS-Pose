import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import time
torch.hub.set_dir('/media/student/Elements/tmp/')


class DinoV2(nn.Module):
    def __init__(self,image_size = (480,640)):
        super(DinoV2, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.eval()
        self.image_size = image_size
        self.transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def prepare_image(self, image,
                  patch_size=14):
        # Crop image to dimensions that are a multiple of the patch size
        image_tensor = image
        chanel, height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
        image_tensor = image_tensor[:,:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // patch_size, cropped_width // patch_size) # h x w (TODO: check)
        return image_tensor, grid_size


    def forward(self, rgb):

        rgb, grid_size = self.prepare_image(rgb)
        #feature = self.model.get_intermediate_layers(rgb)[0]
        feature = self.model.forward_features(rgb)['x_norm_patchtokens']
        feature = feature.reshape(-1,*grid_size, 384)
        
        feature = torch.nn.functional.interpolate(feature.permute(0,3,1,2), size=self.image_size)
        

        return feature

    