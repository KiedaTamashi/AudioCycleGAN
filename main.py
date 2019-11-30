import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from logger import Logger
import torch.utils.data
from dataset import AudioDataset
from model import AudioCycleGAN
from hparams import hparams as opt
import time


# MNIST dataset 
dataset = AudioDataset("./data/preprocess")

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=True)
data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
# Fully connected neural network with one hidden layer
model = AudioCycleGAN(opt)

logger = Logger('./logs')

# Loss and optimizer
model.setup(opt)   # regular setup: load and print networks; create schedulers

total_iters = 0                # the total number of training iterations
for epoch in range(opt.n_epochs, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

    for i, data in enumerate(data_iter):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

        # if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
        #     save_result = total_iters % opt.update_html_freq == 0
        #     model.compute_visuals()
        if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            if opt.display_id > 0:
                pass

            # 1. Log scalar values (scalar summary)
            info = {'loss': losses.item()}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, total_iters + 1)

            # # 2. Log values and gradients of the parameters (histogram summary)
            # for tag, value in model.named_parameters():
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
            #     logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)

            # # 3. Log training images (image summary)
            # info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}
            # for tag, images in info.items():
            #     logger.image_summary(tag, images, total_iters + 1)

        if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (
    epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

