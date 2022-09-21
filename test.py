import argparse
import numpy as np
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import torchvision.utils as vutils
from torchvision import transforms
from torch.autograd import Variable
import os


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
