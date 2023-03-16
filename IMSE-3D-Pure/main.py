#!/usr/bin/python3

import argparse
import os
from trainer import Eva_Trainer,Reg_Trainer
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/Eva.yaml', help='Path to the config file.')
    parser.add_argument('--train', action='store_true',default=True, help='Train mode')
    
    opts = parser.parse_args()
    config = get_config(opts.config)
    print(config)
    
    
    if 'Evaluator' in config['name']:
        trainer = Eva_Trainer(config)
    else:
        trainer = Reg_Trainer(config)

    if opts.train:
        trainer.train()
    else:
        trainer.test_regisatration()

###################################
if __name__ == '__main__':
    main()
