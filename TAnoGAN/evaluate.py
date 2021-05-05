import torch 
import numpy as np
import pandas as pd
import pickle as pkl
import os

from utils import progress_bar

class ModelTest:
    '''
    Model Test for Anomaly Detection 
    '''
    def __init__(self, models, savedir, device):
        """
        Arguments
        ---------
        - models: training model
        - testloader: test dataset
        - criterions: model critetion
        - savedir: directory to save
        - device: device to use [cuda:i, cpu], i=0, ...,n
        """
        self.G, self.D = models

        self.savedir = savedir
        self.device = device

    def inference(self, testloader, lam, optim_iter, datadir, dataname, real_time=False):
        """
        Arguments
        ---------
        - lam: lambda for DR score
        - optim_iter: iteration to obtain optimal Z
        - datadir: data directory
        """
        self.G.train()
        self.D.train()

        batch_size = testloader.batch_size
        window_size = testloader.dataset.window_size
        dr_loss_arr = np.zeros(len(testloader.dataset.data))
    
        for batch_idx, inputs in enumerate(testloader):
            inputs = inputs.to(self.device)

            # get optimal Z
            optimal_z = self.optim_z(inputs, lam=lam, iterations=optim_iter)
    
            # DR Score
            dr_loss = self.dr_score(inputs=inputs, Z=optimal_z, lam=lam, reduction='sum').cpu().detach().numpy()

            # save
            if real_time:
                dr_loss_arr[batch_idx * batch_size:(batch_idx + 1) * batch_size] = dr_loss[:,-1]
            else:
                if batch_idx == 0:
                    dr_loss_arr[:window_size] = dr_loss[0]
                    dr_loss_arr[window_size:window_size + (batch_size - 1)] = dr_loss[1:,-1]
                else:
                    start_idx = (window_size - 1) + (batch_idx * batch_size)
                    if batch_idx == len(testloader)-1:
                        end_idx = (window_size - 1) + (batch_idx * batch_size) + inputs.size(0)
                    else:
                        end_idx = (window_size - 1) + ((batch_idx + 1) * batch_size)

                    dr_loss_arr[start_idx:end_idx] = dr_loss[:,-1]
            
            progress_bar(current=batch_idx, 
                        total=len(testloader), 
                        name='TEST')

        # save DR score
        raw_data = pkl.load(open(os.path.join(datadir,dataname), 'rb'))
        score_df = pd.DataFrame({'date':raw_data.index, 'anomaly_score':dr_loss_arr})
        score_df.to_csv(os.path.join(self.savedir,f'dr_score_lambda{lam}.csv'), index=False)
    
    def dr_score(self, inputs, Z, lam, reduction='none'):
        """
        Arguments
        ---------
        - inputs: input data (batch size x sequence length X the number of features)
        - lam: lambda for DR score
        """
        # Generator 
        fake, _ = self.G(Z.to(self.device))
        
        # Discriminator
        dis_out_real, dis_feature_real = self.D(inputs)
        dis_out_fake, dis_feature_fake = self.D(fake)

        # loss
        if reduction=='none':
            gen_loss = torch.sum(torch.abs(inputs - fake))
            dis_loss = torch.sum(torch.abs(dis_feature_real - dis_feature_fake))
        elif reduction=='sum':
            gen_loss = torch.sum(torch.abs(inputs - fake), axis=2)
            dis_loss = torch.sum(torch.abs(dis_feature_real - dis_feature_fake), axis=2)

        dr_loss = (1 - lam) * gen_loss + lam * dis_loss

        return dr_loss

    def optim_z(self, inputs, lam, iterations): 
        """
        Arguments
        ---------
        - inputs: input data (batch size x sequence length X the number of features)
        - lam: lambda for DR score
        - iterations: iteration to obtain optimal Z
        """
        best_loss = np.inf
        optimal_z = 0
        
        noise = torch.autograd.Variable(torch.nn.init.normal_(torch.zeros(inputs.size()),mean=0,std=0.1),
                                        requires_grad=True)
        z_optimizer = torch.optim.Adam([noise], lr=0.01)
        
        for i in range(iterations):
            score = self.dr_score(inputs=inputs, Z=noise, lam=lam) 
            score.backward()
            
            z_optimizer.step()

            # TODO best score 왜 안하지             
            # if score.item() < best_loss:
            #     best_loss = score.item()
            #     optimal_z = noise

        return noise

    
