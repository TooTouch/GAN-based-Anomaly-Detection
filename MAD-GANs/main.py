from torch.utils.tensorboard import SummaryWriter
from dataloader import get_dataloader
from model import LSTMDiscriminator, LSTMGenerator
from train import ModelTrain
from evaluate import ModelTest
from utils import version_build, torch_seed, load_model

import torch
import json
import argparse
import os
import pickle


def config_args(parser):
    parser.add_argument('--train', action='store_true', help='training model')
    parser.add_argument('--test', action='store_true', help='anomaly scoring')
    parser.add_argument('--resume', type=int, default=None, help='version number to re-train or test model')
    
    # directory
    parser.add_argument('--datadir', type=str, default='../data', help='data directory')
    parser.add_argument('--logdir',type=str, default='./logs', help='logs directory')

    # data
    parser.add_argument('--dataname', type=str, default='sample_data.p', help='dataset name')
    parser.add_argument('--nb_features', type=int, default=31, help='the number of features')
    parser.add_argument('--window_size', type=int, default=60, help='window size for data loader')
    parser.add_argument('--slide_size', type=float, default=30, help='overlap ratio for data loader')
    parser.add_argument('--scale', type=str, default=None, choices=['minmax', 'minmax_m1p1', 'minmax_square'], help='select scaler')
    
    # train options
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='number of batch')
    parser.add_argument('--gen_lr', type=float, default=0.1, help='learning rate of generator')
    parser.add_argument('--dis_lr', type=float, default=0.1, help='learning rate of discriminator')

    # loss 
    parser.add_argument('--gen_loss', type=str, default='mse', choices=['mse','mae'], help='selec generator loss function')

    # setting
    parser.add_argument('--seed', type=int, default=223, help="set randomness, default 223 is Jaehyuk Heo's birthday")
    parser.add_argument('--gpu', type=int, default=0, help='gpu number')
    
    # anormaly score option
    parser.add_argument('--lam', type=float, default=0.5, help='lambda value')
    parser.add_argument('--optim_iter', type=int, default=50, help='the number of iterations to obtain optimal Z')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # set arguments
    parser = argparse.ArgumentParser(description='MAD-GANs pytorch implementation edited by Tootouch')
    args = config_args(parser)

    # gpu
    GPU_NUM = args.gpu # select gpu number
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check

    # savedir
    savedir = version_build(logdir=args.logdir, is_train=args.train, resume=args.resume)

    # save arguments
    json.dump(vars(args), open(os.path.join(savedir,'arguments.json'),'w'), indent=4)

    # set seed
    torch_seed(args.seed)

    # load data
    trainloader, validloader, testloader = get_dataloader(data_root=args.datadir,
                                                          data_name=args.dataname,
                                                          batch_size=args.batch_size,
                                                          scale=args.scale,
                                                          window_size=args.window_size,
                                                          slide_size=args.slide_size)

    # build models 
    G = LSTMGenerator(nb_features = args.nb_features).to(device)
    D = LSTMDiscriminator(nb_features = args.nb_features).to(device)   

    # load saved model 
    start_epoch = 0
    best_metrics = None
    if args.resume != None:
        print(f'resume version{args.resume}')
        G_weights, D_weights, start_epoch, args.gen_lr, args.dis_lr, best_metrics = load_model(resume=args.resume, logdir=args.logdir)
        G.load_state_dict(G_weights)
        D.load_state_dict(D_weights)

    # optimizer
    gen_optim = torch.optim.Adam(G.parameters(), lr=args.gen_lr)
    dis_optim = torch.optim.SGD(D.parameters(), lr=args.dis_lr) 
    ## scheduler
    gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=gen_optim, mode='min', factor=0.9, patience=10, verbose=True)
    dis_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=dis_optim, mode='min', factor=0.9, patience=10, verbose=True)


    # criterion 
    if args.gen_loss == 'mse':
        gen_criterion = torch.nn.MSELoss()
    elif args.gen_loss == 'mae':
        gen_criterion = torch.nn.L1Loss()

    dis_criterion = torch.nn.BCELoss()

    # Training
    if args.train:    
        # tensorboard
        writer = SummaryWriter(savedir) 

        modeltrain = ModelTrain(models=[G, D],
                                trainloader=trainloader,
                                validloader=validloader,
                                logdir=savedir,
                                start_epoch=start_epoch,
                                epochs=args.epochs,
                                criterions=[gen_criterion, dis_criterion],
                                optimizers=[gen_optim, dis_optim],
                                schedulers=[gen_scheduler, dis_scheduler],
                                last_metrics=best_metrics,
                                tensorboard=writer,
                                device=device)

        # save history
        pickle.dump(modeltrain.history, open(os.path.join(savedir,'history.pkl'),'wb'))

    elif args.test:
        modeltest = ModelTest(models=[G, D], 
                              criterions=[gen_criterion, dis_criterion],
                              savedir=savedir, 
                              device=device)

        modeltest.inference(lam=args.lam, 
                            testloader=testloader, 
                            optim_iter=args.optim_iter, 
                            datadir=args.datadir,
                            dataname=args.dataname,
                            real_time=False)
