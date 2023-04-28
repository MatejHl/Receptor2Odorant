import os
import yaml
import json
import datetime
import argparse

from envyaml import EnvYAML

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='config file path')
    parser.add_argument('--cuda_device', type=int,
                        help='Set environment variable CUDA_VISIBLE_DEVICES')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to a directory where output is saved')

    args = parser.parse_args()
    print('Config file: {}'.format(args.config))
    print('---------------')

    # Set visible devices:
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        print('Setting CUDA_VISIBLE_DEVICES to: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    # Read params:
    env = EnvYAML(args.config, flatten = False)
    params = env.yaml_config

    # Read params not using envyaml library (without parsing environment variables):
    # with open(args.config, 'r') as yamlfile:
    #     params = yaml.safe_load(yamlfile)

    # Cast ATOM_FEATURES and BOND_FEATURES to tuple.
    params['ATOM_FEATURES'] = tuple(params['ATOM_FEATURES'])
    params['BOND_FEATURES'] = tuple(params['BOND_FEATURES'])

    # Choose function to run:
    if params['ACTION'] == 'train':
        from Receptor2Odorant.odor_embedding.MPNN.main_train import main_train
        output = main_train(params)
    elif params['ACTION'] == 'eval':
        from Receptor2Odorant.odor_embedding.MPNN.main_eval import main_eval
        output = main_eval(params)
    elif params['ACTION'] == 'predict':
        from Receptor2Odorant.odor_embedding.MPNN.main_predict import main_predict
        output = main_predict(params)
    elif params['ACTION'] == 'predict_single':
        from Receptor2Odorant.odor_embedding.MPNN.main_predict_single import main_predict_single
        output = main_predict_single(params)
    else:
        raise ValueError('Unknown action {}. Available options: {}'.format(params['ACTION'], ['train', 'eval', 'predict']))
    print('Finished...')

    if args.output_dir is not None and output is not None:
        print('Saving results...')
        _datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(args.output_dir, params['ACTION'] + '_output_' + params['MODEL_NAME'] + '_' + _datetime + '.json'), 'w+') as jsonfile:
            json.dump(output, jsonfile)
        print('Done.')