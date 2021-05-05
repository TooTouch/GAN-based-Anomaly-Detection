import torch
import time 
import datetime
import numpy as np 
from utils import progress_bar, CheckPoint
from evaluate import ModelTest


class ModelTrain:
    '''
    Model Training and Validation
    '''
    def __init__(
        self, 
        models: list, 
        trainloader, 
        validloader,
        logdir: str, 
        start_epoch: int, 
        epochs: int, 
        criterions: list, 
        optimizers: list, 
        schedulers: list, 
        tensorboard = None, 
        last_metrics: float = None,
        lam: float = 0.5,
        optim_iter: int = 50,
        device: str = 'cpu', 
    ):
        '''
        Arguments
        ---------
        - models: training model
        - trainloader: train dataset
        - validloader: test set to evaluate in train [default: None]
        - logdir: directory to save
        - start_epoch: start epoch
        - epochs: epochs
        - criterions: model critetion
        - optimizers: model optimizer
        - schedulers: model scheduler
        - tensorboard: TensorBoard
        - last_metrics: latest best model metrics for checkpoint 
        - lam: lambda for DR score
        - optim_iter: iteration to obtain optimal Z
        - device: device to use [cuda:i, cpu], i=0, ...,n
        '''

        self.G, self.D = models
        self.trainloader = trainloader
        self.validloader = validloader
        self.gen_criterion, self.dis_criterion = criterions
        self.gen_optim, self.dis_optim = optimizers
        self.gen_scheduler, self.dis_scheduler = schedulers
        self.lam = lam
        self.optim_iter = optim_iter
        self.device = device
        self.history = {}
        self.writer = tensorboard

        # set checkpoint
        ckp = CheckPoint(logdir=logdir,
                         last_metrics=last_metrics)

        # Training time check
        total_start = time.time()

        # Initialize history list
        train_loss_lst, train_gen_loss_lst, train_dis_loss_lst, train_dis_real_loss_lst, train_dis_fake_loss_lst = [], [], [], [], []
        val_loss_lst, val_gen_loss_lst, val_dis_loss_lst, val_dis_real_loss_lst, val_dis_fake_loss_lst = [], [], [], [], []
        epoch_time_lst = []

        end_epoch = start_epoch + epochs
        for i in range(start_epoch, end_epoch):
            print('Epoch: ',i+1)

            epoch_start = time.time()
            train_dis_loss, train_dis_real_loss, train_dis_fake_loss, train_gen_loss, train_loss = self.train()
            val_dis_loss, val_dis_real_loss, val_dis_fake_loss, val_gen_loss, val_loss = self.validation()        
            val_dr_score = self.validation_dr_score()

            # scheduler
            if self.gen_scheduler!=None:
                self.gen_scheduler.step(train_gen_loss)
                G_last_lr = self.gen_scheduler.state_dict()['_last_lr'][0]
            else:
                G_last_lr = self.gen_optim.state_dict()['param_groups'][0]['lr']

            if self.dis_scheduler!=None:
                self.dis_scheduler.step(train_dis_loss)
                D_last_lr = self.dis_scheduler.state_dict()['_last_lr'][0]
            else:
                D_last_lr = self.dis_optim.state_dict()['param_groups'][0]['lr']

            # check 
            ckp.check(epoch=i+1, G=self.G, D=self.D, score=val_dr_score, G_lr=G_last_lr, D_lr=D_last_lr)

            # Save history
            train_loss_lst.append(train_loss)
            train_gen_loss_lst.append(train_gen_loss)
            train_dis_loss_lst.append(train_dis_loss)
            train_dis_real_loss_lst.append(train_dis_real_loss)
            train_dis_fake_loss_lst.append(train_dis_fake_loss)
            self.writer.add_scalar('Loss/Train', train_loss, i)
            self.writer.add_scalar('Loss/Train(G)', train_gen_loss, i)
            self.writer.add_scalar('Loss/Train(D)', train_dis_loss, i)
            self.writer.add_scalar('Loss/Train(D-Real)', train_dis_real_loss, i)
            self.writer.add_scalar('Loss/Train(D-Fake)', train_dis_fake_loss, i)
        
            val_loss_lst.append(val_loss)
            val_gen_loss_lst.append(val_gen_loss)
            val_dis_loss_lst.append(val_dis_loss)
            val_dis_real_loss_lst.append(val_dis_real_loss)
            val_dis_fake_loss_lst.append(val_dis_fake_loss)
            self.writer.add_scalar('Loss/Validation', val_loss, i)
            self.writer.add_scalar('Loss/Validation(G)', val_gen_loss, i)
            self.writer.add_scalar('Loss/Validation(D)', val_dis_loss, i)
            self.writer.add_scalar('Loss/Validation(D-Real)', val_dis_real_loss, i)
            self.writer.add_scalar('Loss/Validation(D-Fake)', val_dis_fake_loss, i)
            
            end = time.time()
            epoch_time = datetime.timedelta(seconds=end - epoch_start)
            epoch_time_lst.append(str(epoch_time))

            print()

        end = time.time() - total_start
        total_time = datetime.timedelta(seconds=end)
        print('\nFinish Train: Training Time: {}\n'.format(total_time))

        # Make history 
        self.history = {}
        self.history['train'] = []
        self.history['train'].append({
            'loss':train_loss_lst,
            'gen_loss':train_gen_loss_lst,
            'dis_loss':train_dis_loss_lst,
            'dis_real_loss':train_dis_real_loss_lst,
            'dis_fake_loss':train_dis_fake_loss_lst
        })
        
        self.history['validation'] = []
        self.history['validation'].append({
            'loss':val_loss_lst,
            'gen_loss':val_gen_loss_lst,
            'dis_loss':val_dis_loss_lst,
            'dis_real_loss':val_dis_real_loss_lst,
            'dis_fake_loss':val_dis_fake_loss_lst
        })

        self.history['time'] = []
        self.history['time'].append({
            'epoch':epoch_time_lst,
            'total':str(total_time)
        })

    def train(self):
        self.D.train()
        self.G.train()
        train_loss = []
        train_dis_loss = []
        train_dis_real_loss = []
        train_dis_fake_loss = []
        train_gen_loss = [] 
    
        # TODO gen, dis scheduler
        gen_max = 3 
        gen_cnt = 0 
        
        for batch_idx, inputs in enumerate(self.trainloader):
            inputs = inputs.to(self.device)
            noise = torch.autograd.Variable(torch.randn(size=inputs.size())).to(self.device)

            # TAnoGAN training Discriminator first

            # Discriminator
            # set dis optimizer init
            self.dis_optim.zero_grad()

            # generate fake inputs
            inputs_fake, _ = self.G(noise)

            # discriminate
            dis_real_outputs, _ = self.D(inputs)
            dis_fake_outputs, _ = self.D(inputs_fake)

            # dis loss
            dis_real_loss = self.dis_criterion(dis_real_outputs, torch.full(dis_real_outputs.size(), 1, dtype=torch.float, device=self.device))
            dis_fake_loss = self.dis_criterion(dis_fake_outputs, torch.full(dis_fake_outputs.size(), 0, dtype=torch.float, device=self.device))
            dis_loss = dis_real_loss + dis_fake_loss 

            # dis update
            dis_loss.backward()
            self.dis_optim.step()
        

            # Generator
            # set gen optimizer init
            self.gen_optim.zero_grad()

            # generate
            fakes, _ = self.G(noise)

            # gen loss
            gen_loss = self.gen_criterion(fakes, inputs)

            # gen update
            gen_loss.backward()
            self.gen_optim.step()

            # Loss
            train_dis_loss.append(dis_loss.item())
            train_dis_real_loss.append(dis_real_loss.item())
            train_dis_fake_loss.append(dis_fake_loss.item())
            train_gen_loss.append(gen_loss.item())
            loss = dis_loss + gen_loss
            train_loss.append(loss.item())

            progress_bar(current=batch_idx, 
                         total=len(self.trainloader), 
                         name='TRAIN',
                         msg='Total Loss: %.3f | Gen Loss: %.3f | Dis Loss: %.3f (Real %.3f / Fake %.3f)' % (np.mean(train_loss), 
                                                                                                             np.mean(train_gen_loss), 
                                                                                                             np.mean(train_dis_loss),
                                                                                                             np.mean(train_dis_real_loss),
                                                                                                             np.mean(train_dis_fake_loss)))

        train_dis_loss = np.mean(train_dis_loss)
        train_dis_real_loss = np.mean(train_dis_real_loss)
        train_dis_fake_loss = np.mean(train_dis_fake_loss)
        train_gen_loss = np.mean(train_gen_loss)
        train_loss = np.mean(train_loss)

        return train_dis_loss, train_dis_real_loss, train_dis_fake_loss, train_gen_loss, train_loss

    def validation(self):
        self.D.eval()
        self.G.eval()
        val_loss = []
        val_dis_loss = []
        val_dis_real_loss = []
        val_dis_fake_loss = []
        val_gen_loss = [] 

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.validloader):
                inputs = inputs.to(self.device)
                noise = torch.autograd.Variable(torch.randn(size=inputs.size())).to(self.device)

                # Generator 
                gen_outputs, _ = self.G(noise)
                gen_outputs = gen_outputs.detach() 

                # gen loss
                gen_loss = self.gen_criterion(gen_outputs, inputs)
                
                # Discriminator
                dis_real_outputs, _ = self.D(inputs)
                dis_fake_outputs, _ = self.D(gen_outputs)
                dis_real_outputs = dis_real_outputs.detach()
                dis_fake_outputs = dis_fake_outputs.detach()

                # dis loss
                dis_real_loss = self.dis_criterion(dis_real_outputs, torch.full(dis_real_outputs.size(), 1, dtype=torch.float, device=self.device))
                dis_fake_loss = self.dis_criterion(dis_fake_outputs, torch.full(dis_fake_outputs.size(), 0, dtype=torch.float, device=self.device))
                dis_loss = dis_real_loss + dis_fake_loss 
                
                # loss
                val_dis_loss.append(dis_loss.item())
                val_dis_real_loss.append(dis_real_loss.item())
                val_dis_fake_loss.append(dis_fake_loss.item())
                val_gen_loss.append(gen_loss.item())
                loss = gen_loss + dis_loss 
                val_loss.append(loss.item())

                progress_bar(current=batch_idx, 
                             total=len(self.validloader), 
                             name='VALID',
                             msg='Total Loss: %.3f | Gen Loss: %.3f | Dis Loss: %.3f (Real %.3f / Fake %.3f)' % (np.mean(val_loss), 
                                                                                                                 np.mean(val_gen_loss), 
                                                                                                                 np.mean(val_dis_loss),
                                                                                                                 np.mean(val_dis_real_loss),
                                                                                                                 np.mean(val_dis_fake_loss)))

        val_dis_loss = np.mean(val_dis_loss)
        val_dis_real_loss = np.mean(val_dis_real_loss)
        val_dis_fake_loss = np.mean(val_dis_fake_loss)
        val_gen_loss = np.mean(val_gen_loss)
        val_loss = np.mean(val_loss)

        return val_dis_loss, val_dis_real_loss, val_dis_fake_loss, val_gen_loss, val_loss


    def validation_dr_score(self):
        self.G.train()
        self.D.train()

        model_test = ModelTest(models=[self.G, self.D],
                                savedir=None,
                                device=self.device)

        val_dr_score = []
        
        for batch_idx, inputs in enumerate(self.validloader):
            inputs = inputs.to(self.device)
            noise = torch.autograd.Variable(torch.randn(size=inputs.size())).to(self.device)

            optimal_z = model_test.optim_z(inputs=inputs, lam=self.lam, iterations=self.optim_iter)
            dr_score = model_test.dr_score(inputs=inputs, Z=optimal_z, lam=self.lam)

            val_dr_score.append(dr_score)

            progress_bar(current=batch_idx, 
                            total=len(self.validloader), 
                            name='VALID DR Score',
                            msg='DR Score: %.3f' % (dr_score))

        val_dr_score = np.sum(val_dr_score)

        return val_dr_score