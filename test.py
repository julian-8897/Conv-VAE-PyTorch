import argparse
import numpy as np
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torch.nn import functional as F
import torchvision.utils as vutils
from torchvision import transforms
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

#Fixes PosixPath Error
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def latent_traversal(model, samples, n_changes=5, val_range=(-1, 1)):
    """ This function perform latent traversal on a VAE latent space
    model_path: str
        The absolute path of the model to load
    fname: str
        The filename to use for saving the latent traversal
    samples:
        The list of data examples to provide as input of the model
    n_changes: int
        The number of changes to perform on one latent dimension
    val_range: tuple
        The range of values that can be set for one latent dimension
    """
    # TODO: change the next two lines to retrieve the output of your encoder with pytorch
    # m = tf.keras.models.load_model(model_path)
    z_base = model.encode(samples)[-1]
    z_base = z_base.cpu()
    # END TODO
    r, c = n_changes, z_base.shape[1]
    vals = np.linspace(*val_range, r)
    shape = samples[0].shape
    for j, z in enumerate(z_base):
        imgs = np.empty([r * c, *shape])
        for i in range(c):
            z_iter = np.tile(z, [r, 1])
            z_iter[:, i] = vals
            z_iter = torch.from_numpy(z_iter)
            z_iter = z_iter.to(device)
            imgs[r * i:(r * i) + r] = F.sigmoid(model.decode(z_iter)[-1])
        plot_traversal(imgs, r, c, shape[-1] == 1, show=True)
        # save_figure(fname, tight=False)


def plot_traversal(imgs, r, c, greyscale, show=False):
    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111, nrows_ncols=(r, c), axes_pad=0, direction="column")

    for i, (ax, im) in enumerate(zip(grid, imgs)):
        ax.set_axis_off()
        if i % r == 0:
            ax.set_title("z{}".format(i // r), fontdict={'fontsize': 25})
        if greyscale is True:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(im)

    fig.subplots_adjust(wspace=0, hspace=0)
    if show is True:
        plt.show()

    plt.savefig('traversal.png')

def interpolate(autoencoder, x_1, x_2, n=12):
    z_1 = autoencoder.encode(x_1)[2]
    z_2 = autoencoder.encode(x_2)[2]
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = autoencoder.decode(z)
    interpolate_list = interpolate_list.to('cpu').detach()
    print(len(interpolate_list))

    plt.figure(figsize=(64, 64))
    for i in range(len(interpolate_list)):
        ax = plt.subplot(1, len(interpolate_list), i+1)
        plt.imshow(interpolate_list[i].permute(1, 2, 0).numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig('linear_interpolation.png')


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=36,
        shuffle=False,
        validation_split=0.0,
        # training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    # checkpoint = torch.load(config.resume)

    # loading on CPU-only machine
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output, mu, logvar = model(data)

            # computing loss, metrics on test set
            loss = loss_fn(output, data, mu, logvar)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
        #     for i, metric in enumerate(metric_fns):
        #         total_metrics[i] += metric(output, target) * batch_size

        # Reconstructing and generating images for a mini-batch
        test_input, test_label = next(iter(data_loader))
        test_input = test_input.to(device)
        test_label = test_label.to(device)

        recons = model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          os.path.join(
                              "Reconstructions",
                              f"recons_{logger.name}_epoch_{config['trainer']['epochs']}.png"),
                          normalize=True,
                          nrow=6)

        try:
            samples = model.sample(36,
                                   device,
                                   labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(
                "Samples",
                f"{logger.name}.png"),
                normalize=True,
                nrow=6)
        except Warning:
            pass

        # linear interpolation two chosen images
        x_1 = test_input[1].to(device)
        x_1 = torch.unsqueeze(x_1, dim=0)
        x_2 = test_input[2].to(device)
        x_2 = torch.unsqueeze(x_2, dim=0)
        interpolate(model, x_1, x_2, n=5)

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
