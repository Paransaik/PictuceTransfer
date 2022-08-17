##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from torch.autograd import Variable

import utils
from net import Net
from option import Options

def main():
    # figure out the experiments type
    args = Options().parse()
    if args.subcommand is None:
        raise ValueError("ERROR: specify the experiment type")
    if args.cuda and not torch.cuda.is_available():
        raise ValueError("ERROR: cuda is not available, try running on CPU")

    elif args.subcommand == 'eval':
        # Test the pre-trained model
        evaluate(args)

    else:
        raise ValueError('Unknow experiment type')

def evaluate(args):
    content_image = utils.tensor_load_rgbimage(args.content_image, size=args.content_size, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    style = utils.tensor_load_rgbimage(args.style_image, size=args.style_size)
    style = style.unsqueeze(0)
    style = utils.preprocess_batch(style)

    style_model = Net(ngf=args.ngf)
    model_dict = torch.load(args.model)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)

    if args.cuda:
        style_model.cuda()
        content_image = content_image.cuda()
        style = style.cuda()

    style_v = Variable(style)

    content_image = Variable(utils.preprocess_batch(content_image))
    style_model.setTarget(style_v)

    output = style_model(content_image)
    # output = utils.color_match(output, style_v)
    utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)
    print("call by PyTorch-Multi-Style-Transfer")


if __name__ == "__main__":
    main()
