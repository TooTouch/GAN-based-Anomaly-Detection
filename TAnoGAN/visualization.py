import numpy as np
import pandas as pd
import pickle as pkl
import os 

import matplotlib.pyplot as plt
import seaborn as sns
import datetime

import torch 

from utils import load_model, torch_seed
from model import LSTMGenerator
from dataloader import get_dataloader

import argparse

def make_plot(scoredir, datadir, dataname, data_start=None):
    # load data
    data = pkl.load(open(os.path.join(datadir, dataname),'rb'))
    score = pd.read_csv(scoredir)

    # start and stop time of abnormal state 
    stop_start = data[data.state=='abnormal'].index[0]
    if 'sample_data.p' in datadir:
        stop_end = datetime.datetime(2019, 4, 10, 2, 41)
    else:
        stop_end = data[data.state=='abnormal'].index[-1]

    # set date columns to index as datetime type
    score['date'] = pd.to_datetime(score['date'])
    score = score.set_index('date')

    # reset start time
    if data_start is not None:
        score = score[score.index >= data_start]

    # plotting
    fig = plt.figure(figsize=(10, 4))
    plt.axvspan(stop_start, stop_end, alpha=0.2, color='red')
    plt.axvspan(stop_start - datetime.timedelta(days=6 / 24), stop_start, alpha=0.2, color='green')
    plt.scatter(score.index, score['anomaly_score'], c='black', s=0.5)

    plt.xlabel('Date')
    plt.ylabel('Anomaly score')
    plt.margins(x=0)

    # save
    if data_start is not None:
        save_path = scoredir.split('.csv')[0] + '_tight.png'
    else:
        save_path = scoredir.split('.csv')[0] + '.png'
    plt.savefig(fname=save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def optim_Z(G, data, criterion, iteration, device='cpu'):
    best_loss = np.inf
    optimal_z = 0
    
    noise = torch.autograd.Variable(torch.nn.init.normal_(torch.zeros(data.size()),mean=0,std=0.1),
                                    requires_grad=True)
    z_optimizer = torch.optim.Adam([noise], lr=0.01)
    
    for i in range(iteration):
        
        gen_outputs = G(noise.to(device))
        gen_loss = criterion(gen_outputs, data)
        gen_loss.backward()
        
        z_optimizer.step()
        
        if gen_loss.item() < best_loss:
            best_loss = gen_loss.item()
            optimal_z = noise
    
    return optimal_z 



def viz_topk_bottomk(
    k, anomaly_score, G, dataloader, window_size, nb_features, criterion, iteration, colnames, device, save_path=False
):
    # slice meaningful anomaly score
    anomaly_score = anomaly_score.iloc[59:,:]

    # top k and bottom k
    bottomk = anomaly_score.sort_values('anomaly_score').head(n=k)
    topk = anomaly_score.sort_values('anomaly_score').tail(n=k)
    
    ################
    # generation
    ################
    G = G.to(device)
    # bottomk
    bottomk_inputs = torch.zeros(k,window_size,nb_features)
    for i, idx in enumerate(bottomk.index):
        bottomk_inputs[i] = dataloader.dataset.data[idx-60+1:idx+1]

    optim_z = optim_Z(G, bottomk_inputs.to(device), criterion, iteration=iteration, device=device)
    bottomk_gen_outputs = G(optim_z.to(device))
    bottomk_gen_outputs = bottomk_gen_outputs.detach().cpu().numpy()
    bottomk_inputs = bottomk_inputs.numpy()

    # topk
    topk_inputs = torch.zeros(k,window_size,nb_features)
    for i, idx in enumerate(topk.index):
        topk_inputs[i] = dataloader.dataset.data[idx-60+1:idx+1]

    optim_z = optim_Z(G, topk_inputs.to(device), criterion, iteration=iteration, device=device)
    topk_gen_outputs = G(optim_z.to(device))
    topk_gen_outputs = topk_gen_outputs.detach().cpu().numpy()
    topk_inputs = topk_inputs.numpy()
    
    ################
    # Visualization
    ################    
    for f_idx in range(nb_features):
        f, ax = plt.subplots(2,k, figsize=(15,7))

        def ymin_ymax(inputs, gen_outputs, feature_idx):
            inputs_min = inputs[...,feature_idx].min()
            inputs_max = inputs[...,feature_idx].max()
            gen_min = gen_outputs[...,feature_idx].min()
            gen_max = gen_outputs[...,feature_idx].max()

            ymin = inputs_min if inputs_min < gen_min else gen_min
            ymax = inputs_max if inputs_max > gen_max else gen_max

            return ymin - 0.1 , ymax + 0.1

        for i in range(k):
            # top k
            ax[0, i].plot(topk_inputs[i, :, f_idx])
            ax[0, i].plot(topk_gen_outputs[i, :, f_idx])
            ax[0, i].legend(['Real','Generation'], loc='upper right')
            ax[0, i].set_title(f'Top {i}\n({topk.date.tolist()[i]})')

            ymin, ymax = ymin_ymax(topk_inputs, topk_gen_outputs, f_idx)
            ax[0, i].set_ylim(ymin, ymax)

            # bottom k 
            ax[1, i].plot(bottomk_inputs[i, :, f_idx])
            ax[1, i].plot(bottomk_gen_outputs[i, :, f_idx])
            ax[1, i].legend(['Real','Generation'], loc='upper right')
            ax[1, i].set_title(f'Bottom {i}\n({bottomk.date.tolist()[i]})')

            ymin, ymax = ymin_ymax(bottomk_inputs, bottomk_gen_outputs, f_idx)
            ax[1, i].set_ylim(ymin, ymax)

        plt.tight_layout()

        # save
        if save_path:
            plt.savefig(fname=os.path.join(save_path, f'top{k}_bottom{k}_{colnames[f_idx]}.png'))
            plt.close()
        else:
            plt.show()



def run_topk_bottomk(
    k, scoredir, logdir, datadir, dataname, version, lam, loss, scale, iteration, save_path, device
):
    # load raw data
    raw = pkl.load(open(os.path.join(datadir, dataname),'rb'))

    # Load anomaly score
    dr_score = pd.read_csv(scoredir)
    _, _, testloader = get_dataloader(data_root=datadir,
                                      data_name=dataname,
                                      scale=scale,
                                      batch_size=1)

    # define window size and the number of features 
    window_size = testloader.dataset.window_size
    nb_features = testloader.dataset.data.size(-1)

    # loss function
    if loss == 'mse':
        gen_criterion = torch.nn.MSELoss()
    elif loss == 'mae':
        gen_criterion = torch.nn.L1Loss()

    # Load saved model
    # logs 안에 version이 있어야함. version은 학습시 log 정보와 함께 생성됨
    G_weights, _, _, _, _, best_metrics = load_model(resume=version, logdir=logdir)
    print('Best Loss: ',best_metrics)
    G = LSTMGenerator(nb_features = nb_features).to(device)
    G.load_state_dict(G_weights)

    # Visualization
    viz_topk_bottomk(k=k, 
                     anomaly_score=dr_score, 
                     G=G, 
                     dataloader=testloader, 
                     window_size=window_size,
                     nb_features=nb_features, 
                     criterion=gen_criterion, 
                     iteration=iteration, 
                     colnames=raw.columns,
                     device=device,
                     save_path=save_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # version 
    parser.add_argument('--version', type=int, default=0, help='experiment version')
    parser.add_argument('--dataname', type=str, default='sample_data.p', help='dataset name')
    parser.add_argument('--lam', type=float, default=0.5, help='lambda for anomaly score')

    # directory
    parser.add_argument('--logdir', type=str, default='./logs_sample', help='log directory')
    parser.add_argument('--datadir', type=str, default='../data', help='data directory')
    
    # ====================
    # visualize state
    # ====================
    parser.add_argument('--abnormal', action='store_true', help='visualize abnormal state')
    
    # date
    parser.add_argument('--data_start', type=str, default='2019-04-07 00:00', help='start time')
    
    # ====================
    # top k and bottom k
    # ====================
    parser.add_argument('--top_bottom', action='store_true', help='visualize features with generation')

    # parameters
    parser.add_argument('--k', type=int, default=5, help='top or bottom k')
    parser.add_argument('--scale', type=str, default='minmax', choices=['minmax','minmax_m1p1','minmax_square'], help='scale methods')
    parser.add_argument('--opt_iter', type=int, default=50, help='the number of iteration for optimal Z')
    parser.add_argument('--gpu', type=int, default=0, choices=[0,1], help='gpu number')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse','mae'], help='loss function')
    parser.add_argument('--seed', type=int, default=223, help='random seed, 223 is my birthday')
    args = parser.parse_args()

    # seed
    torch_seed(args.seed) # my birthday

    # visualize total period with state
    score_data_path = os.path.join(args.logdir, f'version{args.version}', f'dr_score_lambda{args.lam}.csv')
    
    if args.abnormal: 
        make_plot(scoredir=score_data_path, 
                  datadir=args.datadir,
                  dataname=args.dataname)

        make_plot(scoredir=score_data_path,
                  datadir=args.datadir,
                  dataname=args.dataname,
                  data_start=datetime.datetime.strptime(args.data_start, '%Y-%m-%d %H:%M'))

    if args.top_bottom:
        # gpu
        GPU_NUM = args.gpu # select gpu number
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device) # change allocation of current GPU
        print ('Current cuda device ', torch.cuda.current_device()) # check

        # check save directory
        save_path = os.path.join(args.logdir, f'version{args.version}', f'top{args.k}_bottom{args.k}_lambda{args.lam}')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # run
        run_topk_bottomk(k=args.k, 
                         scoredir=score_data_path,
                         logdir=args.logdir, 
                         datadir=args.datadir, 
                         dataname=args.dataname, 
                         version=args.version, 
                         lam=args.lam, 
                         loss=args.loss,
                         scale=args.scale, 
                         iteration=args.opt_iter, 
                         save_path=save_path,
                         device=device)