import torch 
import numpy as np
import os 
import random
import sys
import time

def torch_seed(random_seed: int = 223):
    """
    set deterministic seed

    Argument
    --------
    - random_seed : random seed number, default 223 is my birth day. haha

    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def load_model(resume, logdir):
    """
    load saved model
    
    Arguments
    ---------
    - resume : version to re-train or test
    - logdir : version directory to load model

    Return
    ------
    - G_weights    : saved generator weights
    - D_weights    : saved discriminator weights
    - start_epoch  : last epoch of saved experiment version
    - last_lr      : saved learning rate
    - best_metrics : best metrics of saved experiment version

    """
    savedir = os.path.join(logdir, f'version{resume}')
    for name in os.listdir(savedir):
        if '.pth' in name:
            modelname = name

    modelpath = os.path.join(savedir, modelname)
    print('modelpath: ',modelpath)

    loadfile = torch.load(modelpath)
    G_weights = loadfile['G']
    D_weights = loadfile['D']
    start_epoch = loadfile['best_epoch'] 
    last_gen_lr = loadfile['best_gen_lr']
    last_dis_lr = loadfile['best_dis_lr']
    best_metrics = loadfile['best_loss']
    
    return G_weights, D_weights, start_epoch, last_gen_lr, last_dis_lr, best_metrics


def version_build(logdir: str, is_train: bool, resume: int):
    """
    make n th version folder

    Arguments
    ---------
    - logdir   : log directory
    - is_train : train or not
    - resume   : version to re-train or test

    Return
    ------
    - logdir : version directory to log history

    """
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    if is_train:
        version = len(os.listdir(logdir))
    else:
        version = resume

    logdir = os.path.join(logdir, f'version{version}')

    if is_train:
        os.makedirs(logdir)

    return logdir

last_time = time.time()
begin_time = last_time

def progress_bar(current, total, name, msg=None, width=None):
    """
    progress bar to show history
    
    Arguments
    ---------
    - current : current batch index
    - total   : total data length
    - name    : description of progress bar
    - msg     : history of training model
    """

    if width==None:
        _, term_width = os.popen('stty size', 'r').read().split()
        term_width = int(term_width)
    else:
        term_width = width
    
    TOTAL_BAR_LENGTH = 65.

    global last_time, begin_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(f'{name} [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



class CheckPoint:
    def __init__(self, logdir: str, last_metrics: float = None):
        '''
        Arguments
        ---------
        - logdir       : log directory
        - last_metrics : lastest model metrics
        '''

        self.best_score = np.inf if last_metrics==None else last_metrics
        self.best_epoch = 0 
        self.logdir = logdir

    def check(self, epoch: int, G, D, score: float, G_lr: float, D_lr: float):
        '''
        Arguments
        ---------
        - epoch : current epoch
        - G     : Generator
        - D     : Discriminator
        - score : current score
        - G_lr  : current learning rate of generator
        - D_lr  : current learning rate or discriminator
        '''

        if score < self.best_score:
            self.model_save(epoch, G, D, score, G_lr, D_lr)
        
        
    def model_save(self, epoch: int, G, D, score: float, G_lr: float, D_lr: float):
        '''
        Arguments
        ---------
        - epoch : current epoch
        - G     : Generator
        - D     : Discriminator
        - score : current score
        - G_lr  : current learning rate of generator
        - D_lr  : current learning rate or discriminator
        '''

        print('Save complete, epoch: {0:}: Best loss has changed from {1:.5f} to {2:.5f}'.format(epoch, self.best_score, score))

        state = {
            'G':G.state_dict(),
            'D':D.state_dict(),
            'best_loss':score,
            'best_epoch':epoch,
            'best_gen_lr':G_lr,
            'best_dis_lr':D_lr
        }

        save_lst =  os.listdir(self.logdir)
        for f in save_lst:
            if '.pth' in f:
                os.remove(os.path.join(self.logdir, f))   

        f_name = f'epoch={epoch}.pth'
        torch.save(state, os.path.join(self.logdir, f_name))

        self.best_score = score
        self.best_epoch = epoch

