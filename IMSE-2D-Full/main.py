#!/usr/bin/python3
import argparse
import os
from trainer import Gen_Trainer, Eva_Trainer, Reg_Trainer
import yaml
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def main():
    #torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/Registartion.yaml', help='Path to the config file.')
    parser.add_argument('--train', action='store_true',default=True, help='Train mode')

    opts = parser.parse_args()
    config = get_config(opts.config)
    print(config)
    # Tranin CycleGan or RegGan
    if 'GAN' in config['name']:  
        trainer = Gen_Trainer(config)
        if opts.train:
            trainer.train()
        else:
            trainer.test_generation_fake()
    # Tranin Evaluator   
    elif 'Evaluator' in config['name']:
        trainer = Eva_Trainer(config)
        if opts.train:
            trainer.train()
        else:
            trainer.test_translation()
    # Tranin Registration Network       
    elif 'Registration' in config['name']:
        trainer = Reg_Trainer(config)
        if opts.train:
            trainer.train()
        else:
            trainer.test_mask()
    else:
        raise Exception('Unknown config file.')



###################################
if __name__ == '__main__':
    main()
