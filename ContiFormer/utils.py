import time
import math
import numpy as np
import iisignature
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import torch
from main import *

"""
This is from Pytorch tutorial (https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
"""
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def leadlag(X):
    '''
    Returns lead-lag-transformed stream of X

    Arguments:
        X: list, whose elements are tuples of the form
        (time, value).

    Returns:
        list of points on the plane, the lead-lag
        transformed stream of X
    '''

    l=[]

    for j in range(2*(len(X))-1):
        i1=j//2
        i2=j//2
        if j%2!=0:
            i1+=1
        l.append(np.concatenate([X.loc[i1].values[:], X.loc[i2].values[:]]))

    return np.stack(l)

def save_checkpoint(state, epoch , loss_is_best, filename, save_all_epochs):
    if(save_all_epochs):
        torch.save(state, filename+'/'+str(epoch)+'_v_loss.pth.tar')
    if(loss_is_best):
        torch.save(state, filename+'/best_v_loss.pth.tar')



def plot_predictions(device, model, dataloader, outdir, num_predictions=10, indices_keep=None, all_indices=None):
    model.eval()  # Set the model to evaluation mode
    fig, axs = plt.subplots(num_predictions, 1, figsize=(10, 20))
    with torch.no_grad():  # No need to track gradients

        batch = next(iter(dataloader))

        for i in range(num_predictions):
            if i >= num_predictions:
                break

            print(args.dataset)

            # Get the inputs and labels
            if (args.dataset == 'sMNIST'): 
                inputs, labels = batch
                inputs = inputs.reshape(inputs.size(0), 1, 784)
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.permute(0, 2, 1)
            elif (args.dataset == 'psMNIST'): 
                inputs = inputs.reshape(inputs.size(0), 1, 784)
                inputs = inputs.permute(0, 2, 1)        
                inputs = inputs[:,random_permutation , :]
                inputs, labels = inputs.to(device), labels.to(device)
            elif (args.dataset == 'sinusoidal') or (args.dataset == 'll'):
                inputs, labels = batch['input'].to(device), batch['label'].to(device) 
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

            x = np.linspace(0, inputs.shape[1], inputs.shape[1])
            if args.random:
                indices_keep = sorted(random.sample(all_indices, 500))
                x = x[indices_keep]
                inputs = inputs[:,indices_keep,:]

            # Forward pass
            outputs = model(inputs)

            # Plot the original signal and the model's prediction
            axs[i].plot(inputs[i].squeeze().cpu().numpy())

            #axs[i].set_title(f'Label: {labels[i].item()} ({(labels[i].item() + 1) * 5} Hz), Prediction: {torch.argmax(outputs[i]).item()}') #The Hz computation should be change according to the dataset
            axs[i].set_title(f'Label: {labels[i].item()} , Prediction: {torch.argmax(outputs[i]).item()}') #The Hz computation should be change according to the dataset


    plt.tight_layout()
    plt.savefig(f'{outdir}/predictions_not_using_signatures.png')
    plt.savefig(f'{outdir}/predictions_not_using_signatures.pdf')


def plot_predictions_signatures(device, args, model, dataloader, outdir, num_predictions=10, random_permutation=None, indices_keep=None, all_indices=None):
    model.eval()  # Set the model to evaluation mode
    fig, axs = plt.subplots(num_predictions, 2, figsize=(10, 20))
    with torch.no_grad():  # No need to track gradients

        batch = next(iter(dataloader))

        for i in range(num_predictions):
            if i >= num_predictions:
                break

            

            # Get the inputs and labels
            if (args.dataset == 'sMNIST'): 
                inputs, labels = batch
                inputs = inputs.reshape(inputs.size(0), 1, 784)
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.permute(0, 2, 1)
            elif (args.dataset == 'psMNIST'): 
                inputs = inputs.reshape(inputs.size(0), 1, 784)
                inputs = inputs.permute(0, 2, 1)        
                inputs = inputs[:,random_permutation , :]
                inputs, labels = inputs.to(device), labels.to(device)
            elif (args.dataset == 'sinusoidal') or (args.dataset == 'll'):
                inputs, labels = batch['input'].to(device), batch['label'].to(device) 
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)


            x = np.linspace(0, inputs.shape[1], inputs.shape[1])
            if args.random:
                indices_keep = sorted(random.sample(all_indices, 1000))
                x = x[indices_keep]
                inputs = inputs[:,indices_keep,:]
            

            inputs_original = inputs.clone()

            if (args.use_signatures):
                if not args.stack:
                    if args.overlapping_sigs and args.univariate:
                        inputs=Signature_overlapping_univariate(inputs,args.sig_level, args.sig_win_len, device)
                    elif args.overlapping_sigs and not args.univariate:
                        if args.irreg:
                            inputs = Signature_overlapping_irreg(inputs,args.sig_level, args.num_windows, x, device)
                        else:
                            inputs = Signature_overlapping(inputs, args.sig_level, args.sig_win_len, device)
                    elif not args.overlapping_sigs and args.univariate:
                        inputs = Signature_nonoverlapping_univariate(inputs,args.sig_level, args.sig_win_len, device)
                    elif not args.overlapping_sigs and not args.univariate:
                        if args.irreg:
                            inputs = Signature_nonoverlapping_irreg(inputs,args.sig_level, args.num_windows, x, device)
                        else:
                            inputs = Signature_nonoverlapping(inputs,args.sig_level, args.sig_win_len, device)
                else:
                    if not args.univariate:
                        if args.irreg:
                            inputs1 = Signature_nonoverlapping_irreg(inputs,args.sig_level, args.num_windows, x, device)
                            inputs2 = Signature_overlapping_irreg(inputs,args.sig_level, args.num_windows, x, device)
                        else:
                            inputs1 = Signature_nonoverlapping(inputs,args.sig_level, args.sig_win_len, device)
                            inputs2 = Signature_overlapping(inputs,args.sig_level, args.sig_win_len, device)
                        inputs = torch.cat((inputs1, inputs2), dim=2)
                        
                    else:
                        inputs1 = Signature_nonoverlapping_univariate(inputs,args.sig_level, args.sig_win_len, device)
                        inputs2 = Signature_overlapping_univariate(inputs,args.sig_level, args.sig_win_len, device)
                        inputs = torch.cat((inputs1, inputs2), dim=2)
                
          


            # Forward pass
            outputs = model(inputs)

            # Plot the original signal and the model's prediction
            axs[i, 1].plot(inputs[i].squeeze().cpu().numpy())
            #axs[i, 1].set_title(f'Label: {labels[i].item()} ({(labels[i].item() + 1) * 5} Hz), Prediction: {torch.argmax(outputs[i]).item()}') #The Hz computation should be change according to the dataset

            axs[i, 0].plot(inputs_original[i].squeeze().cpu().numpy())
            #axs[i, 0].set_title(f'Label: {labels[i].item()} ({(labels[i].item() + 1) * 5} Hz), Prediction: {torch.argmax(outputs[i]).item()}')
            
            axs[i, 0].set_title(f'Label: {labels[i].item()} , Prediction: {torch.argmax(outputs[i]).item()}')


            # Add a title to the row
            axs[i, 1].set_ylabel('Transformed Input')
            axs[i, 0].set_ylabel('Original Input')

    plt.tight_layout()
    plt.savefig(f'{outdir}/predictions_using_signatures.png')
    plt.savefig(f'{outdir}/predictions_using_signatures.pdf')




def plot_functions(train_loss_list, val_accuracy_list, outdir, use_signatures, test_accuracy_list=None):
    # Plotting the training loss
    plt.close('all')
    plt.plot(train_loss_list)
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    if use_signatures:
        plt.savefig(f'{outdir}/training_loss_using_signatures.png')
        plt.savefig(f'{outdir}/training_loss_using_signatures.pdf')
    else:
        plt.savefig(f'{outdir}/training_loss_not_using_signatures.png')
        plt.savefig(f'{outdir}/training_loss_not_using_signatures.pdf')

    # Plotting the validation accuracy
    plt.close('all')
    plt.plot(val_accuracy_list)
    pd.DataFrame(val_accuracy_list).to_csv(f'{outdir}/validation_accuracy_using_signatures.csv')
    if test_accuracy_list is not None:
        pd.DataFrame(val_accuracy_list).to_csv(f'{outdir}/test_accuracy_using_signatures.csv')

    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if use_signatures:
        plt.savefig(f'{outdir}/validation_accuracy_using_signatures.png')
        plt.savefig(f'{outdir}/validation_accuracy_using_signatures.pdf')
    else:
        plt.savefig(f'{outdir}/validation_accuracy_not_using_signatures.png')
        plt.savefig(f'{outdir}/validation_accuracy_not_using_signatures.pdf')
    


def plot_error_distribution(error_distribution, outdir, use_signatures):
    import matplotlib.pyplot as plt

    plt.bar(error_distribution.keys(), error_distribution.values())
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    #plt.savefig('test_error_distribution.png')
    if use_signatures:
        plt.savefig(f'{outdir}/test_error_distribution.png')
        plt.savefig(f'{outdir}/test_error_distribution.pdf')
    else:
        plt.savefig(f'{outdir}/test_error_distribution.png')
        plt.savefig(f'{outdir}/test_error_distribution.pdf')