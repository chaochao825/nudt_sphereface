import argparse
import yaml
from easydict import EasyDict
import os
import glob

from utils.sse import sse_input_path_validated, sse_output_path_validated
from utils.yaml_rw import load_yaml, save_yaml
from face_recognizer.main import main as face_main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./input', help='input path')
    parser.add_argument('--output_path', type=str, default='./output', help='output path')
    
    parser.add_argument('--process', type=str, default='attack', help='[adv, attack, defend, train, inference_1_1, inference_1_n, attack_defense_eval, dataset_sampling]')
    parser.add_argument('--model', type=str, default='deepface', help='model name')
    parser.add_argument('--data', type=str, default='lfw', help='data name [vggface2, celeba, webface, lfw, yaleb, megaface]')
    parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')
    parser.add_argument('--sample_count', type=int, default=100, help='number of images to sample')
    parser.add_argument('--threshold', type=float, default=0.55, help='confidence threshold for verification')
    
    parser.add_argument('--attack_method', type=str, default='bim', help='attack method [bim, dim, tim, pgd, cw, deepfool]')
    parser.add_argument('--defend_method', type=str, default='hgd', help='defend method [hgd, tvm, livenessdetection, featurespacepurification, ensembledefense]')
    
    parser.add_argument('--cfg_path', type=str, default='./cfgs', help='cfg path')
    
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--device', type=int, default=0, help='which gpu for cuda')
    parser.add_argument('--workers', type=int, default=0, help='dataloader workers')
    
    parser.add_argument('--epsilon', type=float, default=8/255, help='epsilon for attack method')
    parser.add_argument('--step_size', type=float, default=2/255, help='step size for attack method')
    parser.add_argument('--max_iterations', type=int, default=10, help='max iterations for attack method')
    
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict_environ = {}
    for key, value in args_dict.items():
        args_dict_environ[key] = type_switch(os.getenv(key.upper(), value), value)
    args_easydict = EasyDict(args_dict_environ)
    return args_easydict

def type_switch(environ_value, value):
    if isinstance(value, int):
        return int(environ_value)
    elif isinstance(value, float):
        return float(environ_value)
    elif isinstance(value, bool):
        return bool(environ_value)
    elif isinstance(value, str):
        return environ_value
    
def face_cfg(args):
    os.makedirs(args.cfg_path, exist_ok=True)
    
    cfg = EasyDict()
    cfg.model = args.model
    cfg.data = args.data
    cfg.save_dir = args.output_path
    cfg.project = args.model
    cfg.name = args.process
    cfg.batch = args.batch
    cfg.workers = args.workers
    cfg.device = f'cuda:{args.device}' if args.device >= 0 else 'cpu'
    cfg.num_classes = args.num_classes
    cfg.verbose = True
    cfg.half = False
    
    # Standardize data and model paths
    # We use args.input_path/data as the default data search path
    cfg.data_path = os.path.join(args.input_path, 'data')
    
    if args.process in ['adv', 'attack']:
        cfg.mode = args.process
        cfg.batch = 1
        model_files = glob.glob(os.path.join(args.input_path, 'model', '*'))
        cfg.pretrained = model_files[0] if model_files else None
    elif args.process == 'defend':
        cfg.mode = 'defend'
        cfg.batch = 1
        cfg.device = 'cpu'
    elif args.process == 'train':
        cfg.mode = 'train'
        cfg.epochs = args.epochs
    elif args.process == 'inference_1_1':
        cfg.mode = 'inference_1_1'
    elif args.process == 'inference_1_n':
        cfg.mode = 'inference_1_n'
    elif args.process == 'attack_defense_eval':
        cfg.mode = 'attack_defense_eval'
    elif args.process == 'dataset_sampling':
        cfg.mode = 'dataset_sampling'
        cfg.sample_count = args.sample_count
    
    cfg_dict = dict(cfg)
    args.cfg_yaml = f'{args.cfg_path}/config.yaml'
    save_yaml(cfg_dict, args.cfg_yaml)
    
    return args, cfg

def main(args):
    args, cfg = face_cfg(args)
    face_main(args, cfg)
        
if __name__ == '__main__':
    args = parse_args()
    
    sse_input_path_validated(args)
    sse_output_path_validated(args)
    main(args)
