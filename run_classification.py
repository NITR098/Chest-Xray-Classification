import argparse
import logging
import os
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import medmnist
from medmnist import INFO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid

import matplotlib as mpl
mpl.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt

from dataloader import ChestXray14, JSRT
from models import build_model
from utils import seed_it_all, train_one_epoch, validation, save_checkpoint, my_transform
from utils import plot_performance, test_model, test_classification, metric_AUROC

from torchinfo import summary
from sklearn.metrics import accuracy_score



def run_experiments(args, train_loader, val_loader, test_loader, model_path, output_path):

    torch.cuda.empty_cache()
    model = build_model(args)
    model = model.to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()
    accuracy = []
    mean_auc = []

    for idx in range(args.num_trial):
        print (f"Run: {idx+1}")
        experiment = args.exp_name + "_run_" + str(idx+1)
        save_model_path = model_path.joinpath(experiment)
        args.plot_path = model_path / (experiment+ ".pdf")
        # print(str(args.plot_path))
        
        log_file = output_path.joinpath(f"run_{str(idx+1)}.log")
        # print(log_file)
        logging.basicConfig(filename=log_file, level=logging.INFO, filemode='a',
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        
        loss_train_hist = []
        loss_valid_hist = []
        acc_train_hist = []
        acc_valid_hist = []
        best_loss_valid = torch.inf
        epoch_counter = 0
        for epoch in range(epoch_counter, args.max_epochs):
            model, loss_train, acc_train = train_one_epoch(args, 
                                                            model, 
                                                            train_loader,
                                                            loss_fn,
                                                            optimizer)
            logging.info(f"Epoch:{epoch+1}, TrainLoss:{loss_train:0.4f}, TrainAcc:{acc_train:0.4f}")
            
            print("start validation.....")
            loss_valid, acc_valid = validation(args, model, val_loader, loss_fn)
            logging.info(f"Epoch:{epoch+1}, ValidLoss:{loss_valid:0.4f}, ValidAcc:{acc_valid:0.4f}")
            # print(f"Epoch:{epoch+1}, ValidLoss = {loss_valid:0.4f}, ValidAcc = {acc_valid:0.4f}")
            
            loss_train_hist.append(loss_train)
            loss_valid_hist.append(loss_valid)
            acc_train_hist.append(acc_train)
            acc_valid_hist.append(acc_valid)
            
            if loss_valid < best_loss_valid:
                save_checkpoint({
                'epoch': epoch + 1,
                'lossMIN': best_loss_valid,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scheduler': lr_scheduler.state_dict()
                }, filename=str(save_model_path))
                
                # torch.save(model, f'model.pt')
                best_loss_valid = loss_valid
                print('Model Saved!')
                
            epoch_counter += 1
            
        plot_performance(args, loss_train_hist, loss_valid_hist,
                            acc_train_hist, acc_valid_hist, epoch_counter)
        
        print ("start testing.....")
        output_file = os.path.join(output_path, args.exp_name + "_results.txt")
        saved_model = model_path.joinpath(f"{experiment}.pth.tar")
        
        # acc, confusion_matrix, auroc = test_model(args, model, str(saved_model), train_loader)
    #     print(f">>{experiment}: ACC = {acc:0.4f}")
    #     accuracy.append(acc)
    #     print(f">>{experiment}: AUC = {auroc:0.4f}")
    #     mean_auc.append(auroc)

    # mean_auc = np.array(mean_auc)
    # print(f">> All trials: mAUC  = {np.array2string(mean_auc, precision=4, separator=',')}")
    # logging.info("All trials: mAUC  = {}\n".format(np.array2string(mean_auc, precision=4, separator='\t')))
    # print(f">> Mean AUC over All trials: = {np.mean(mean_auc):0.4f}")
    # logging.info("Mean AUC over All trials = {:.4f}\n".format(np.mean(mean_auc)))
    # print(f">> STD over All trials:  = {np.std(mean_auc):0.4f}")
    # logging.info("STD over All trials:  = {:.4f}\n".format(np.std(mean_auc)))

        y_test, p_test = test_classification(args, str(saved_model), test_loader)
        
        if args.dataset_name == "RSNAPneumonia":
            acc = accuracy_score(np.argmax(y_test.cpu().numpy(),axis=1),np.argmax(p_test.cpu().numpy(),axis=1))
            print(">>{}: ACCURACY = {}".format(experiment,acc))
            logging.info("{}: ACCURACY = {}\n".format(experiment, np.array2string(np.array(acc), precision=4, separator='\t')))
            accuracy.append(acc)
        individual_results = metric_AUROC(y_test, p_test, args.num_classes)
        
        print(">>{}: AUC = {}".format(experiment, np.array2string(np.array(individual_results), precision=4, separator=',')))
        logging.info("{}: AUC = {}\n".format(experiment, np.array2string(np.array(individual_results), precision=4, separator='\t')))
        
        mean_over_all_classes = np.array(individual_results).mean()
        print(">>{}: AUC = {:.4f}".format(experiment, mean_over_all_classes))
        logging.info("{}: AUC = {:.4f}\n".format(experiment, mean_over_all_classes))

        mean_auc.append(mean_over_all_classes)

    mean_auc = np.array(mean_auc)
    print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=4, separator=',')))
    logging.info("All trials: mAUC  = {}\n".format(np.array2string(mean_auc, precision=4, separator='\t')))
    print(">> Mean AUC over All trials: = {:0.4f}".format(np.mean(mean_auc)))
    logging.info("Mean AUC over All trials = {:0.4f}\n".format(np.mean(mean_auc)))
    print(">> STD over All trials:  = {:0.4f}".format(np.std(mean_auc)))
    logging.info("STD over All trials:  = {:0.4f}\n".format(np.std(mean_auc)))