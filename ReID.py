import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import models
from torchvision import transforms

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048
        self.cam = False

    def forward(self, x):
        x = self.base(x)
        if self.cam:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ReID_Tracker():
    def __init__(self):
        self.ReID_model=ResNet50(10)
        self.transformation=transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.dist_thresh= 8
        self.ref_emb=torch.tensor([])

    def load_pretrained(self, path):
        checkpoint = torch.load(path)
        pretrain_dict = checkpoint['state_dict']
        ReID_model_dict = self.ReID_model.state_dict()
        #define a dictionary, k=key, v=values, defined using :
        #drop layers that don't match in size
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in ReID_model_dict and ReID_model_dict[k].size() == v.size()}
        ReID_model_dict.update(pretrain_dict)
        self.ReID_model.load_state_dict(ReID_model_dict)

    def embedding_generator(self, tensor_img):
        #tensor_img=tensor_img.unsqueeze(0) #network expect batch of 64 images
        self.ReID_model.eval()
        #feeds it to the model
        with torch.no_grad():
            embedding =self.ReID_model(tensor_img)
        return embedding

    def distance(self, emb1, emb2):
        return torch.cdist(emb1, emb2, p=2)

    def image_preprocessing(self, img):
        #Preprocessing: resize so that it fits with the entry of the neural net, convert to tensor type, normalization
        return self.transformation(img)

    def embedding_comparator(self, detections):
        #aupdate the ref embedding and return the index of correct detection
        idx=None
        #new_ref=None
        best_dist=None
        #ref_emb is a tensor, detections must be a list of tensor image cropped, conf list array
        if self.ref_emb.nelement() == 0:
            if(detections.size(0)==1):
            #may need to squeeze(0)
                ref_emb=self.embedding_generator(detections)
                idx=0
                self.ref_emb=torch.squeeze(ref_emb)
                return idx
            else:
                print("error: trying to initialize with multiple detections")
                return None
                #try to handle the case of multiple detections by using the conf_list as input ??
                #for idx in range(detections.size[0]):
                #ref_emb=embedding_generator(ReID_model, detections)
        else:
            #compute L2 distance for each detection

            emb_list=self.embedding_generator(detections)
            #fill a list of all distances wrt to ref
            #dist_list=[]
            dist_list=self.distance(emb_list, torch.unsqueeze(self.ref_emb,0))# .repeat(emb_list.size(0), 1))
            dist_list=dist_list.squeeze(0)
            best_dist = min(dist_list)
            best_dist=best_dist.squeeze()
            idx=int((dist_list==best_dist).nonzero().squeeze())
            best_dist=float(best_dist)
            #compare to the defined threshold to decide if it's similar enough
            if (best_dist < self.dist_thresh):
                self.ref_emb=emb_list[idx]
                return idx
            else:
                print("under thresh")
                return None