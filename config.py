import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# Preprocess parameters
parser.add_argument('--n_labels', type=int, default=2,help='number of classes')


# data in/out and dataset
parser.add_argument('--train_image_dir',default = 'C:\\med3d\\traindata\\images',help='fixed trainset root path')
parser.add_argument('--train_label_dir',default = 'C:\\med3d\\traindata\\labels',help='fixed trainset root path')
parser.add_argument('--val_image_dir',default = 'C:\\med3d\\valdata\\images',help='fixed trainset root path')
parser.add_argument('--val_label_dir',default = 'C:\\med3d\\valdata\\labels',help='fixed trainset root path')
parser.add_argument('--test_data_path',default = 'C:\\med3d\\test\\images',help='Testset path')
parser.add_argument('--save',default='./trained_models',help='save path of trained model')
parser.add_argument('--batch_size', type=list, default=1,help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=14, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.0001)')
parser.add_argument('--early-stop', default=30, type=int, help='early stopping (default: 30)')
parser.add_argument('--crop_size', type=int, default=48)
parser.add_argument('--val_crop_max_size', type=int, default=96)

# test
parser.add_argument('--test_cut_size', type=int, default=48, help='size of sliding window')
parser.add_argument('--test_cut_stride', type=int, default=24, help='stride of sliding window')
parser.add_argument('--postprocess', type=bool, default=False, help='post process')


args = parser.parse_args()


