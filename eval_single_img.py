import argparse
import torch
from torchvision.transforms import Compose
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image as pimg

from models.semseg import SemsegModel
from models.semseg import SemsegPyramidModel
from data.transform import *
from data.mux.transform import *
from data.cityscapes.labels import labels


parser = argparse.ArgumentParser(description='Detector train')
parser.add_argument('--pyramid', action='store_true', default=False, help='use SemsegPyramidModel')

if __name__ == '__main__':
    args = parser.parse_args()

    single_scale = not args.pyramid

    downsample = 1
    if single_scale:
        num_levels = 1
        alphas = [1.]
    else:
        alpha = 2.0
        num_levels = 3
        alphas = [alpha ** i for i in range(num_levels)]

    target_size = ts = (1242, 375)
    target_size_feats = (ts[0] // 4, ts[1] // 4)

    scale = 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_classes = 19
    map_to_id = {}
    i = 0
    for label in labels:
        if label.ignoreInEval is False:
            map_to_id[label.id] = i
            i += 1

    transforms = Compose(
        [Open(),
        RemapLabels(map_to_id, num_classes),
        Pyramid(alphas=alphas),
        SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
        Normalize(scale, mean, std),
        Tensor(),
        ]
    )

    use_bn = True
    if single_scale:
        from models.resnet.resnet_single_scale import resnet18
        resnet = resnet18(pretrained=False, efficient=False, use_bn=use_bn)
        model = SemsegModel(resnet, num_classes, use_bn=use_bn)
        # if change number classes, modify here
        model.load_state_dict(torch.load('weights/swiftnet_ss_cs.pt'), strict=True)
    else:
        from models.resnet.resnet_pyramid import resnet18
        resnet = resnet18(pretrained=False, pyramid_levels=num_levels, efficient=False, use_bn=use_bn)
        model = SemsegPyramidModel(resnet, num_classes)
        model.load_state_dict(torch.load('weights/swiftnet_pyr_cs.pt'), strict=True)

    model = model.cuda()

    ret_dict = {
        'image': '000025.png',
    }
    batch = transforms(ret_dict)
    # import ipdb; ipdb.set_trace()
    for i in range(len(batch['pyramid'])):
        batch['pyramid'][i].unsqueeze_(0)

    logits, _ = model.do_forward(batch)
    pred = torch.argmax(logits.data, dim=1).byte().cpu().numpy().astype(np.uint32)

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(2, 1, 1)
    plt.imshow(pred[0])
    fig.add_subplot(2, 1, 2)
    plt.imshow( pimg.open(ret_dict['image']).convert('RGB') )

    plt.show()