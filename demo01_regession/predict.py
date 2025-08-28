import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('predict')

    # setup data_loader instances
    data_loader = config.init_obj('test_data_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'],weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output = model(data)
          
            # save sample images, or do something with output here


    logger.info()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    config = ConfigParser.from_args(args)
    main(config)
