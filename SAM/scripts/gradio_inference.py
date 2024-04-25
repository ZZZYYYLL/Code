import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gradio as gr
import torch
import sys
import torchvision.transforms as transforms
import numpy as np
sys.path.append(".")
sys.path.append("..")
from argparse import Namespace
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im, log_image
from options.test_options import TestOptions
from models.psp import pSp
from PIL import Image



# update test options with options used during training
test_opts = TestOptions().parse()
# update test options with options used during training
ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
opts = ckpt['opts']
opts.update(vars(test_opts))
opts = Namespace(**opts)

net = pSp(opts)
net.eval()
net.cuda()

transforms = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def process_single(input_image, target_age):
    input_image = transforms(input_image)
    with torch.no_grad():
        age_transformers = AgeTransformer(target_age=target_age)
        input_image_age = age_transformers.add_aging_channel(input_image)
        input_image_age = input_image_age.unsqueeze(0).cuda()
        result_image = net(input_image_age, randomize_noise=False, resize=opts.resize_outputs)
        result_image = tensor2im(result_image[0])
    return [result_image]

def process_multi(input_image, target_age1, target_age2, target_age3, target_age4, target_age5, target_age6):
    target_age_list = [target_age1, target_age2, target_age3, target_age4, target_age5, target_age6]
    input_image = transforms(input_image)
    with torch.no_grad():
        result_image_list = []
        for i, target_age in enumerate(target_age_list):
            age_transformers = AgeTransformer(target_age=target_age)
            input_image_age = age_transformers.add_aging_channel(input_image)
            input_image_age = input_image_age.unsqueeze(0).cuda()
            result_image = net(input_image_age, randomize_noise=False, resize=opts.resize_outputs)
            result_image = tensor2im(result_image[0])
            result_image_list.append(result_image)
            # result_image_list.append(np.array(result_image))
        # result_image_total = np.concatenate(result_image_list, axis=1)
        # result_image_total = Image.fromarray(result_image_total)
    return result_image_list


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## SAM Face Aging")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="pil")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                target_age1 = gr.Slider(label="Age1", minimum=0, maximum=100, value=0, step=1)
                target_age2 = gr.Slider(label="Age2", minimum=0, maximum=100, value=20, step=1)
                target_age3 = gr.Slider(label="Age3", minimum=0, maximum=100, value=40, step=1)
                target_age4 = gr.Slider(label="Age4", minimum=0, maximum=100, value=60, step=1)
                target_age5 = gr.Slider(label="Age5", minimum=0, maximum=100, value=80, step=1)
                target_age6 = gr.Slider(label="Age6", minimum=0, maximum=100, value=100, step=1)
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, target_age1, target_age2, target_age3, target_age4, target_age5, target_age6]
    run_button.click(fn=process_multi, inputs=ips, outputs=[result_gallery])
block.launch(server_name='0.0.0.0', share=True)

# if __name__ == "__main__":
#     input_image = Image.open("SAM/notebooks/images/866.jpg")
#     target_age_list = [0,20,40,60,80,100]
#     result_image_total = process_multi(input_image, target_age_list)
#     input_image.save("input_image.jpg")
#     result_image_total[0].save("result_image_total.jpg")
