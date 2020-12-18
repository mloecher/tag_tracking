from timeit import default_timer as timer
from .loss_utils import multi_loss, weighted_mse_loss, two_loss
from .utils import get_ref_plots
from .datagen import Dataset, do_all_batch_aug

import torch
import torch.optim as optim
import torch.nn as nn

from collections import defaultdict 
import copy
import pickle

import numpy as np

def test_training(model, optimizer, device, train_set, valid_set, load_train_params, load_valid_params, name='', 
                  history = defaultdict(list), n_epochs = 60, print_network = True, batch_stride = 1000, weighted_loss = False, 
                  writer = None, epoch_scheduler = None, batch_scheduler = None, loss_func = None, save_val = False):
    
    
    if not 'best_loss_validate' in history:
        history['best_loss_validate'] = 999999999.0

    train_gen = torch.utils.data.DataLoader(train_set, **load_train_params)
    valid_gen = torch.utils.data.DataLoader(valid_set, **load_valid_params)
    
    if print_network:
        print(model)
    else:
        print('Starting')
    
    train_iter = iter(train_gen)
    
    if weighted_loss:
        criterion = weighted_mse_loss
    elif loss_func == 'smoothl1':
        print('Using Smooth L1 Loss')
        criterion = nn.SmoothL1Loss()
    elif loss_func == 'l1':
        print('Using L1 Loss')
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
        # criterion = nn.SmoothL1Loss()

    history['name'] = name
    
    print('Total Batches:', len(train_iter), '  Batch Stride:', batch_stride)

    for epoch in range(n_epochs):

        print('----- Epoch {:02d} -----'.format(epoch))
        
        train_set.shuffle()
        train_iter = iter(train_gen)

        train(train_iter, model, criterion, optimizer, epoch, device, history, batch_stride=batch_stride, writer = writer, scheduler = batch_scheduler)
        
        validate(valid_gen, model, criterion, optimizer, epoch, device, history, writer = writer)

        if history['loss_val'][-1] < history['best_loss_validate']:
            # torch.save(model.state_dict(), './states/best_val_{}.pt'.format(name))
            history['best_state_validate'] = copy.deepcopy(model.state_dict())
            history['best_loss_validate'] = history['loss_val'][-1]
            print('New Best Validate:  {:.2e}'.format(history['best_loss_validate']))
            
            if save_val:
                save_name = './%s_bestval.pt' % name
                torch.save(model.state_dict(), save_name)

            writer.flush()

        if epoch_scheduler is not None:
            epoch_scheduler.step()


def train(loader_iter, model, criterion, optimizer, epoch, device, history, batch_stride = None, scheduler=None, writer = None, verbose = 1):
    
    if batch_stride is None:
        batch_stride = len(loader_iter)
        
    start_time = timer()
    model.train()
    
    running_loss = 0.0
    running_mse = 0.0
    running_mae = 0.0
    
    for i in range(batch_stride):
        # x, y_true, iter_id = next(loader_iter)
        x, y_true = next(loader_iter)
        x, y_true = x.to(device), y_true.to(device)

        # if i < 2:
        #     print(epoch, i, iter_id)
        
        optimizer.zero_grad()
        
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step(epoch + i/batch_stride)
        
        running_loss += loss.item()

        t_mse, t_mae = two_loss(y_pred, y_true)
        running_mse += t_mse.item()
        running_mae += t_mae.item()
        
        # This is for recording individual loss (smoothed), and learning rate
        # This is basiclaly replaced by tensorboard, but is maybe still useful
        lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
        smoothing = 0.5
        if len(history['batch_loss']) == 0:
            smooth_loss = loss.item()
        else:
            smooth_loss = smoothing * history['batch_loss'][-1] + (1.0 - smoothing) * loss.item()
        history['batch_loss'].append(smooth_loss)
        history['batch_lr'].append(lr_step)
        
    
    # Take mean of error functions
    final_loss = running_loss/i
    final_mse = running_mse/i
    final_mae = running_mae/i

    history['loss_train'].append(final_loss)
    history['mse_train'].append(final_mse)
    history['mae_train'].append(final_mae)

    if writer is not None:
        writer.add_scalar('Loss/train', final_loss, epoch)
        writer.add_scalar('MSE/train', final_mse, epoch)
        writer.add_scalar('MAE/train', final_mae, epoch)

    total_time = timer() - start_time
    
    if verbose:
        print('           Loss: {:.2e}   MSE:{:.2e}    MAE:{:.2e}    {:d}  [{:.1f} sec]  LR = {:.2e}'.format(final_loss, final_mse, final_mae, i, total_time, lr_step))
    
    
def validate(loader, model, criterion, optimizer, epoch, device, history, writer = None):
    
    start_time = timer()
    model.eval()
    
    running_loss = 0.0
    running_mse = 0.0
    running_mae = 0.0
    
    with torch.no_grad():
        for i, (x, y_true) in enumerate(loader):
            x, y_true = x.to(device), y_true.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y_true)

            running_loss += loss.item()
            
            t_mse, t_mae = two_loss(y_pred, y_true)
            running_mse += t_mse.item()
            running_mae += t_mae.item()
    
    final_loss = running_loss/i
    final_mse = running_mse/i
    final_mae = running_mae/i

    history['loss_val'].append(final_loss)
    history['mse_val'].append(final_mse)
    history['mae_val'].append(final_mae)

    if writer is not None:
        writer.add_scalar('Loss/validate', final_loss, epoch)
        writer.add_scalar('MSE/validate', final_mse, epoch)
        writer.add_scalar('MAE/validate', final_mae, epoch)

    total_time = timer() - start_time
    
    print('Validation Loss: {:.2e}   MSE:{:.2e}    MAE:{:.2e}    {:d}  [{:.1f} sec]'.format(final_loss, final_mse, final_mae, i, total_time))
    
    