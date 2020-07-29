import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import os

def define_Vgg19(opt):
    use_gpu = len(opt.gpu_ids) > 0
    vgg19net = vgg19(models.vgg19(pretrained=True), opt)

    if use_gpu:
        assert(torch.cuda.is_available())
        vgg19net.cuda(opt.gpu_ids[0])
    return vgg19net

class vgg19(nn.Module):
    def __init__(self, basic_model, opt):
        super(vgg19, self).__init__()
        self.layer_1 = self.make_layers(basic_model,0,2)
        self.layer_2 = self.make_layers(basic_model,2,7)
        self.layer_3 = self.make_layers(basic_model,7,12)
        self.layer_4 = self.make_layers(basic_model,12,21)
        self.layer_5 = self.make_layers(basic_model,21,30)
        self.layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5]

        image_height = image_width = opt.imageSize
        self.Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor
        self.input = self.Tensor(opt.batchSize, opt.input_nc, image_height, image_width)
        self.convergence_threshold = opt.convergence_threshold
        self.old_lr = opt.lr
        self.beta = opt.beta1

    def make_layers(self, basic_model, start_layer, end_layer):
        layer = []
        features = next(basic_model.children())
        original_layer_number = 0
        for module in features.children():
            if original_layer_number >= start_layer and original_layer_number < end_layer:
                layer += [module]
            original_layer_number += 1
        return nn.Sequential(*layer)

    def make_classifier_layer(self, old_classifier, dropout_layers):
        classifier_layer = []
        features = next(old_classifier.children())
        for name, module in old_classifier.named_children():
            if int(name) not in dropout_layers:
                classifier_layer += [module]
        return nn.Sequential(*classifier_layer)

    def set_input(self, input_A, set_new_var = True):
        if set_new_var == True:
            self.input = self.Tensor(input_A.size())
            self.input.copy_(input_A)
        else:
            self.input = input_A

    def forward(self, level = 6, start_level = 0, set_as_var = True):
        assert(level >= start_level)
        if set_as_var == True:
            self.input_sample = Variable(self.input)
        else:
            self.input_sample = self.input

        layer_i_output = layer_i_input = self.input_sample
        for i in range(start_level, level):
            layer_i = self.layers[i]
            layer_i_output = layer_i(layer_i_input)
            layer_i_input = layer_i_output

        return layer_i_output

    def deconve(self, features, original_image_width, src_level, dst_level, print_errors=True):
        dst_feature_size = self.get_layer_size(dst_level, batch_size = features.size(0), width = original_image_width)
        deconvolved_feature = Variable(self.Tensor(dst_feature_size), requires_grad=True)
        deconvolved_feature.data.fill_(0)
        optimizer = torch.optim.Adam([{'params': deconvolved_feature}],
                                            lr=self.old_lr, betas=(self.beta, 0.999))

        src_level_size = self.get_layer_size(src_level, batch_size = features.size(0), width = original_image_width)
        src_layer = Variable(self.Tensor(src_level_size), requires_grad=False)
        src_layer.data.copy_(features)

        error = float('inf')
        criterionPerceptual = PerceptualLoss(tensor=self.Tensor)
        i = 0
        self.reset_last_losses()
        while self.convergence_criterion() > self.convergence_threshold:
            optimizer.zero_grad()
            self.set_input(deconvolved_feature, set_new_var = False)
            deconvolved_feature_forward = self.forward(level=src_level, start_level=dst_level, set_as_var = False)
            loss_perceptual = criterionPerceptual(deconvolved_feature_forward, src_layer)
            loss_perceptual.backward()
            error = loss_perceptual.item()
            self.update_last_losses(error)
            if (i % 3 == 0) and (print_errors == True):
                print("error: ", error)
            optimizer.step()
            i += 1

        return deconvolved_feature

    def reset_last_losses(self):
        self.last_losses = np.array([0,100,200,300,400,500])

    def update_last_losses(self,loss):
        self.last_losses = np.delete(self.last_losses, 0)
        self.last_losses = np.append(self.last_losses, loss)

    def convergence_criterion(self):
        convergence_criterion = np.average(np.abs(np.diff(self.last_losses)))
        return convergence_criterion

    def get_layer_size(self, level, batch_size=1, width = 224):
        channels = [3,64,128,256,512,512]
        if level == 0:
            width_layer = width
        else:
            width_layer = int(width/(2**int(level-1)))
        return torch.Size([batch_size, channels[int(level)], width_layer, width_layer])

class PerceptualLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor, size_average=True):
        super(PerceptualLoss, self).__init__()
        self.Tensor = tensor
        self.loss = nn.MSELoss(size_average=size_average)

    def __call__(self, input, target_tensor):
        return self.loss(input, target_tensor)
