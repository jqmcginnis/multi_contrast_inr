import argparse
import time
import os
import yaml
import pathlib
import pdb
import time
import asyncio
import cProfile
import pstats

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import nibabel as nib
import numpy as np
import lpips

from model import MLPv1, MLPv2, Siren, WireReal
from dataset import MultiModalDataset, InferDataset
from visualization_utils import show_slices_gt
from sklearn.preprocessing import MinMaxScaler
from utils import input_mapping, compute_metrics, dict2obj, get_string, compute_mi_hist, compute_mi
from loss_functions import MILossGaussian, NMI, NCC


def parse_args():
    parser = argparse.ArgumentParser(description='Train Neural Implicit Function for a single scan.')
    parser.add_argument('--config', default='config.yaml', help='config file (.yaml) containing the hyper-parameters for training.')
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0], help="GPU ID following PCI order.")

    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)

    # patient
    parser.add_argument('--subject_id', type=str, default=None)
    parser.add_argument('--experiment_no', type=int, default=None)
    return parser.parse_args()


def main(args):

    # Init arguments 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    # Load the config 
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config_dict)

    # we bypass lr, epoch and batch_size if we provide them via arparse

    if args.lr != None:
        config.TRAINING.LR = args.lr
        config_dict["TRAINING"]["LR"] = args.lr
    
    if args.batch_size != None:
        config.TRAINING.BATCH_SIZE = args.batch_size
        config_dict["TRAINING"]["BATCH_SIZE"] = args.batch_size
    
    if args.epochs != None:
        config.TRAINING.EPOCHS = args.epochs
        config_dict["TRAINING"]["EPOCHS"] = args.epochs

    # dataset specific
    if args.subject_id != None:
        config.DATASET.SUBJECT_ID = args.subject_id
        config_dict["DATASET"]["SUBJECT_ID"] = args.subject_id

    # experiment type
    if args.experiment_no == 1:
        # t1 / FLAIR
        config.DATASET.LR_CONTRAST1= 't1_LR' 
        config.DATASET.LR_CONTRAST2= 'flair_LR'
        config_dict["DATASET"]["LR_CONTRAST1"] = config.DATASET.LR_CONTRAST1
        config_dict["DATASET"]["LR_CONTRAST2"] = config.DATASET.LR_CONTRAST2

    elif args.experiment_no == 2:
        # DIR / FLAIR
        config.DATASET.LR_CONTRAST1= 'dir_LR' 
        config.DATASET.LR_CONTRAST2= 'flair_LR'
        config_dict["DATASET"]["LR_CONTRAST1"] = config.DATASET.LR_CONTRAST1
        config_dict["DATASET"]["LR_CONTRAST2"] = config.DATASET.LR_CONTRAST2

    elif args.experiment_no == 3:
        # T1w / T2w
        config.DATASET.LR_CONTRAST1= 't1_LR' 
        config.DATASET.LR_CONTRAST2= 't2_LR'
        config_dict["DATASET"]["LR_CONTRAST1"] = config.DATASET.LR_CONTRAST1
        config_dict["DATASET"]["LR_CONTRAST2"] = config.DATASET.LR_CONTRAST2
    
    else:
        # use the settings in config.yaml instead
        if args.experiment_no != None:
            raise ValueError("Experiment not defined.")


    # logging run
    if args.logging:
        wandb.login()
        wandb.init(config=config_dict, project=config.SETTINGS.PROJECT_NAME)

    # make directory for models
    weight_dir = f'runs/{config.SETTINGS.PROJECT_NAME}_weights'
    image_dir = f'runs/{config.SETTINGS.PROJECT_NAME}_images'

    pathlib.Path(weight_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)

    # seeding
    torch.manual_seed(config.TRAINING.SEED)
    np.random.seed(config.TRAINING.SEED)
    
    device = f'cuda:{config.SETTINGS.GPU_DEVICE}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # load dataset
    dataset = MultiModalDataset(
                    image_dir = config.SETTINGS.DIRECTORY,
                    name = config.SETTINGS.PROJECT_NAME,
                    subject_id=config.DATASET.SUBJECT_ID,
                    contrast1_LR_str=config.DATASET.LR_CONTRAST1,
                    contrast2_LR_str=config.DATASET.LR_CONTRAST2, 
                    )


    # Model Selection
    model_name = (
                f'{config.SETTINGS.PROJECT_NAME}_subid-{config.DATASET.SUBJECT_ID}_'
                f'ct1LR-{config.DATASET.LR_CONTRAST1}_ct2LR-{config.DATASET.LR_CONTRAST2}_'
                f's_{config.TRAINING.SEED}_shuf_{config.TRAINING.SHUFFELING}_'
    )


    # output_size
    if config.TRAINING.CONTRAST1_ONLY or config.TRAINING.CONTRAST2_ONLY:
        output_size = 1
        if config.TRAINING.CONTRAST1_ONLY:
            model_name = f'{model_name}_CT1_ONLY_'
        else:
            model_name = f'{model_name}_CT2_ONLY_'

    else:
        output_size = 2

    # Embeddings
    if config.MODEL.USE_FF:
        mapping_size = config.FOURIER.MAPPING_SIZE  # of FF
        input_size = 2* mapping_size
        B_gauss = torch.tensor(np.random.normal(scale=config.FOURIER.FF_SCALE, size=(config.FOURIER.MAPPING_SIZE, 3)), dtype=torch.float32).to(device)
        input_mapper = input_mapping(B=B_gauss, factor=config.FOURIER.FF_FACTOR).to(device)
        model_name = f'{model_name}_FF_{get_string(config_dict["FOURIER"])}_'

    else:
        input_size = 3

    # Model Selection
    if config.MODEL.USE_SIREN:
        model = Siren(in_features=input_size, out_features=output_size, hidden_features=config.MODEL.HIDDEN_CHANNELS,
                    hidden_layers=config.MODEL.NUM_LAYERS, first_omega_0=config.SIREN.FIRST_OMEGA_0, hidden_omega_0=config.SIREN.HIDDEN_OMEGA_0)   # no dropout implemented
        model_name = f'{model_name}_SIREN_{get_string(config_dict["SIREN"])}_'
    elif config.MODEL.USE_WIRE_REAL:
        model = WireReal(in_features=input_size, out_features=output_size, hidden_features=config.MODEL.HIDDEN_CHANNELS,
                    hidden_layers=config.MODEL.NUM_LAYERS, 
                    first_omega_0=config.WIRE.WIRE_REAL_FIRST_OMEGA_0, hidden_omega_0=config.WIRE.WIRE_REAL_HIDDEN_OMEGA_0,
                    first_s_0=config.WIRE.WIRE_REAL_FIRST_S_0, hidden_s_0=config.WIRE.WIRE_REAL_HIDDEN_S_0
                    )
        model_name = f'{model_name}_WIRE_{get_string(config_dict["WIRE"])}_'   
    
    else:
        if config.MODEL.USE_TWO_HEADS:
            if (config.TRAINING.CONTRAST1_ONLY or config.TRAINING.CONTRAST2_ONLY) == True:
                raise ValueError('Do not use MLPv2 for single contrast.')

            model = MLPv2(input_size=input_size, output_size=output_size, hidden_size=config.MODEL.HIDDEN_CHANNELS,
                        num_layers=config.MODEL.NUM_LAYERS, dropout=config.MODEL.DROPOUT)
            model_name = f'{model_name}_MLP2_'
        else:
            model = MLPv1(input_size=input_size, output_size=output_size, hidden_size=config.MODEL.HIDDEN_CHANNELS,
                        num_layers=config.MODEL.NUM_LAYERS, dropout=config.MODEL.DROPOUT)
            model_name = f'{model_name}_MLP2_'

    model.to(device)

    print(f'Number of MLP parameters {sum(p.numel() for p in model.parameters())}')

    # model for lpips metric
    lpips_loss = lpips.LPIPS(net='alex').to(device)

    model_name = f'{model_name}_NUML_{config.MODEL.NUM_LAYERS}_N_{config.MODEL.HIDDEN_CHANNELS}_D_{config.MODEL.DROPOUT}_'     
    # Loss

    if config.TRAINING.LOSS == 'L1Loss':
        criterion = nn.L1Loss()
    elif config.TRAINING.LOSS == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError('Loss function not defined!')

    model_name = f'{model_name}_{config.TRAINING.LOSS}__{config.TRAINING.LOSS_MSE_C1}__{config.TRAINING.LOSS_MSE_C2}_'     

    # custom losses in addition to normal loss
    if config.TRAINING.USE_MI:
        mi_criterion = MILossGaussian(num_bins=config.MI_CC.MI_NUM_BINS, sample_ratio=config.MI_CC.MI_SAMPLE_RATIO, gt_val=config.MI_CC.GT_VAL)
        model_name = f'{model_name}_{get_string(config_dict["MI_CC"])}_'     
    
    if config.TRAINING.USE_CC:
        cc_criterion = NCC()
        model_name = f'{model_name}_{get_string(config_dict["MI_CC"])}_'    
        
    if config.TRAINING.USE_NMI:
        mi_criterion = NMI(intensity_range=(0,1), nbins=config.MI_CC.MI_NUM_BINS, sigma=config.MI_CC.NMI_SIGMA)
        model_name = f'{model_name}_{get_string(config_dict["MI_CC"])}_'    

    # optimizer
    if config.TRAINING.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING.LR)#, weight_decay=5e-5)
        model_name = f'{model_name}_{config.TRAINING.OPTIM}_{config.TRAINING.LR}_'    
    else:
        raise ValueError('Optim not defined!')
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= config.TRAINING.EPOCHS)
    
    mi_buffer = np.zeros((4,1))
    mi_mean = -1.0

    # Load Training Data
    train_dataloader = DataLoader(dataset, batch_size=config.TRAINING.BATCH_SIZE, 
                                 shuffle=config.TRAINING.SHUFFELING, 
                                 num_workers=config.SETTINGS.NUM_WORKERS)

    for epoch in range(config.TRAINING.EPOCHS):
        # set model to train
        model.train()
        wandb_epoch_dict = {}

        model_name_epoch = f'{model_name}_e{int(epoch)}_model.pt'  
        model_path = os.path.join(weight_dir, model_name_epoch)

        print(model_path)

        loss_epoch = 0.0
        start = time.time()

        for batch_idx, (data, labels, segm) in enumerate(train_dataloader):
            loss_batch = 0
            wandb_batch_dict = {}

            if not config.TRAINING.CONTRAST1_ONLY and not config.TRAINING.CONTRAST2_ONLY:               
                contrast1_mask = (labels[:,0] != -1.0)
                contrast1_labels = labels[contrast1_mask,0]
                contrast1_labels = contrast1_labels.reshape(-1,1).to(device=device)
                contrast1_segm = segm[contrast1_mask,:]
                contrast1_data = data[contrast1_mask,:]
                
                contrast2_mask = (labels[:,1] != -1.0)
                contrast2_labels = labels[contrast2_mask,1]
                contrast2_labels = contrast2_labels.reshape(-1,1).to(device=device)
                contrast2_segm = segm[contrast2_mask,:]
                contrast2_data = data[contrast2_mask,:]

                data = torch.cat((contrast1_data,contrast2_data), dim=0)
                # labels = torch.cat((contrast1_labels,contrast2_labels), dim=0)

            else:
                # we need to filter data and labels
                if config.TRAINING.CONTRAST2_ONLY:
                    mask = (labels[:,0] != -1.0)
                    labels = labels[mask,0]
                    labels = labels.reshape(-1,1).to(device=device)
                    data = data[mask,:]
                else:
                    mask = (labels[:,1] != -1.0)
                    labels = labels[mask,1]
                    labels = labels.reshape(-1,1).to(device=device)
                    data = data[mask,:]      

            if config.TRAINING.CONTRAST2_ONLY or config.TRAINING.CONTRAST1_ONLY:
                if torch.cuda.is_available():
                    data, labels  = data.to(device=device), labels.to(device=device)

            else:
                if torch.cuda.is_available():
                    data, contrast1_labels, contrast2_labels  = data.to(device=device), contrast1_labels.to(device=device), contrast2_labels.to(device=device)

            if config.MODEL.USE_FF:
                data = input_mapper(data)
            elif config.MODEL.USE_SIREN:
                data = data*np.pi

            # pass to model
            target = model(data)

            if config.MODEL.USE_SIREN or config.MODEL.USE_WIRE_REAL: # TODO check syntax compatibility
                target, _ = target

            if not config.TRAINING.CONTRAST1_ONLY and not config.TRAINING.CONTRAST2_ONLY:   
                # compute the loss on both modalities!
                # target = torch.where(intensity_index, target[:, 1], target[:, 0])
                mse_target1 = target[:len(contrast1_data),0:1]  # contrast1 output for contrast1 coordinate
                mse_target2 = target[len(contrast1_data):,1:2]  # contrast2 output for contrast2 coordinate
                # target_mse = torch.cat((mi_target1, mi_target2), dim=0)

                if config.MI_CC.MI_USE_PRED:
                    mi_target1 = target[:,0:1]
                    mi_target2 = target[:,1:2]
                    
                elif config.TRAINING.USE_MI or config.TRAINING.USE_NMI or config.TRAINING.USE_CC:
                    mi_target1 = target[:len(contrast1_data),1][contrast1_segm.squeeze()]  # contrast2 output for contrast1 coordinate
                    mi_target2 = target[len(contrast1_data):,0][contrast2_segm.squeeze()]   # contrast1 output for contrast2 coordinate

            else:
                # for contrast 1 or contrast 2 only 
                target_mse = target
                   
            if not config.TRAINING.CONTRAST1_ONLY and not config.TRAINING.CONTRAST2_ONLY:  
                
                loss = config.TRAINING.LOSS_MSE_C1*criterion(mse_target1, contrast1_labels)+config.TRAINING.LOSS_MSE_C2*criterion(mse_target2, contrast2_labels)
            
            else:
                loss = criterion(target_mse, labels)
            
            if args.logging:
                wandb_batch_dict.update({'mse_loss': loss.item()})
            
            # mutual information loss
            if config.TRAINING.USE_MI or config.TRAINING.USE_NMI:
                if config.MI_CC.MI_USE_PRED:
                    mi_loss = mi_criterion(mi_target1.unsqueeze(0).unsqueeze(0), mi_target2.unsqueeze(0).unsqueeze(0))
                    loss += config.MI_CC.LOSS_MI*(mi_loss)
                    if args.logging:
                        wandb_batch_dict.update({'mi_loss': (mi_loss).item()})
                else:
                    mi_loss1 = mi_criterion(mi_target1.unsqueeze(0).unsqueeze(0), contrast1_labels[contrast1_segm].unsqueeze(0).unsqueeze(0))
                    mi_loss2 = mi_criterion(mi_target2.unsqueeze(0).unsqueeze(0), contrast2_labels[contrast2_segm].unsqueeze(0).unsqueeze(0))
                    loss += config.MI_CC.LOSS_MI*(mi_loss1+mi_loss2)
                    if args.logging:
                        wandb_batch_dict.update({'mi_loss': (mi_loss1+mi_loss2).item()})
                        
            if config.TRAINING.USE_CC:
                cc_loss1 = cc_criterion(mi_target1.unsqueeze(0).unsqueeze(0), contrast1_labels[contrast1_segm].unsqueeze(0).unsqueeze(0))
                cc_loss2 = cc_criterion(mi_target2.unsqueeze(0).unsqueeze(0), contrast2_labels[contrast2_segm].unsqueeze(0).unsqueeze(0))
                loss += config.MI_CC.LOSS_CC*(cc_loss1+cc_loss2)
                if args.logging:
                    wandb_batch_dict.update({'cc_loss': -(cc_loss1+cc_loss2).item()})
                    
                
            # zero gradients
            optimizer.zero_grad()
            # backprop
            loss.backward()
            optimizer.step()
            # epoch loss
            loss_batch += loss.item()
            loss_epoch += loss_batch

            if args.logging:
                wandb_batch_dict.update({'batch_loss': loss_batch})
                wandb.log(wandb_batch_dict)  # update logs per batch

        # collect epoch stats
        epoch_time = time.time() - start

        lr = optimizer. param_groups[0]["lr"]
        if args.logging:
            wandb_epoch_dict.update({'epoch_no': epoch})
            wandb_epoch_dict.update({'epoch_time': epoch_time})
            wandb_epoch_dict.update({'epoch_loss': loss_epoch})
            wandb_epoch_dict.update({'lr': lr})

        if epoch == (config.TRAINING.EPOCHS -1):
            torch.save(model.state_dict(), model_path)


        scheduler.step()
        ################ INFERENCE #######################

        model_inference = model
        model_inference.eval()

        # start inference
        start = time.time()

        # coordinate grid, affine and inference dataset are static
        # only process once!

        if epoch == 0:

            # assumes GT contrasts share common grid and affine
            mgrid = dataset.get_coordinates()
            mgrid_affine = dataset.get_affine()
            x_dim, y_dim, z_dim = dataset.get_dim()

            infer_data = InferDataset(mgrid)
            infer_loader = torch.utils.data.DataLoader(infer_data,
                                                       batch_size=5000,
                                                       shuffle=False,
                                                       num_workers=config.SETTINGS.NUM_WORKERS)

        out = np.zeros((int(x_dim*y_dim*z_dim), 2))
        model_inference.to(device)
        for batch_idx, (data) in enumerate(infer_loader):

            if torch.cuda.is_available():
                data = data.to(device)
                
            if config.MODEL.USE_FF:
                data = input_mapper(data)
            elif config.MODEL.USE_SIREN:
                data = data*np.pi
            else:
                data = data
                
            output = model_inference(data)

            if config.MODEL.USE_SIREN or config.MODEL.USE_WIRE_REAL:
                output, _ = output

            out[batch_idx*5000:(batch_idx*5000 + len(output)),:] = output.cpu().detach().numpy() 

        model_intensities=out

        ################ EVALUATION #######################

        if not config.TRAINING.CONTRAST1_ONLY and not config.TRAINING.CONTRAST2_ONLY:

            model_intensities_contrast1 = model_intensities[:,1] # contrast1
            model_intensities_contrast2 = model_intensities[:,0] # contrast2

            scaler = MinMaxScaler()
            label_arr = np.array(model_intensities_contrast1, dtype=np.float32)
            model_intensities_contrast1= scaler.fit_transform(label_arr.reshape(-1, 1))

            label_arr = np.array(model_intensities_contrast2, dtype=np.float32)
            model_intensities_contrast2= scaler.fit_transform(label_arr.reshape(-1, 1))

            inference_time = time.time() - start
            if args.logging:
                wandb_epoch_dict.update({'inference_time': inference_time})

            print("Generating NIFTIs.")
            img_contrast1 = model_intensities_contrast1.reshape((x_dim, y_dim, z_dim))#.cpu().numpy()
            img_contrast2 = model_intensities_contrast2.reshape((x_dim, y_dim, z_dim))#.cpu().numpy()

            gt_contrast1 = dataset.get_contrast1_gt().reshape((x_dim, y_dim, z_dim)).cpu().numpy()
            gt_contrast2 = dataset.get_contrast2_gt().reshape((x_dim, y_dim, z_dim)).cpu().numpy()

            label_arr = np.array(gt_contrast1, dtype=np.float32)
            gt_contrast1= scaler.fit_transform(gt_contrast1.reshape(-1, 1)).reshape((x_dim, y_dim, z_dim))

            label_arr = np.array(gt_contrast2, dtype=np.float32)
            gt_contrast2= scaler.fit_transform(gt_contrast2.reshape(-1, 1)).reshape((x_dim, y_dim, z_dim))

            pred_contrast1 = img_contrast1
            pred_contrast2 = img_contrast2
            affine = np.array(mgrid_affine)

            # to debug if we are comparing the right contrasts
            # nib.save(nib.Nifti1Image(pred_contrast1, affine), "pred_contrast1.nii.gz")
            # nib.save(nib.Nifti1Image(pred_contrast2, affine), "pred_contrast2.nii.gz")
            # nib.save(nib.Nifti1Image(gt_contrast1, affine), "gt_contrast1.nii.gz")
            # nib.save(nib.Nifti1Image(gt_contrast2, affine), "gt_contrast2.nii.gz")

            metrics_contrast1 = compute_metrics(gt=gt_contrast1.copy(), pred=pred_contrast1.copy(), mask=dataset.get_contrast1_gt_mask(), lpips_loss=lpips_loss, device=device)
            metrics_contrast2 = compute_metrics(gt=gt_contrast2.copy(), pred=pred_contrast2.copy(), mask=dataset.get_contrast2_gt_mask(), lpips_loss=lpips_loss, device=device)
            
            metrics_mi_true = compute_mi_hist(gt_contrast1.copy(), gt_contrast2.copy(), dataset.get_contrast2_gt_mask(), bins=32)
            metrics_mi_1 = compute_mi_hist(pred_contrast1.copy(), gt_contrast2.copy(), dataset.get_contrast2_gt_mask(), bins=32)
            metrics_mi_2 = compute_mi_hist(pred_contrast2.copy(), gt_contrast1.copy(), dataset.get_contrast2_gt_mask(), bins=32)
            metrics_mi_pred = compute_mi_hist(pred_contrast1.copy(), pred_contrast2.copy(), dataset.get_contrast2_gt_mask(), bins=32)
                   
            metrics_mi_approx = compute_mi(pred_contrast1.copy(), pred_contrast2.copy(), dataset.get_contrast2_gt_mask(),device)
            
            if args.logging:
                wandb_epoch_dict.update({f'contrast1_ssim': metrics_contrast1["ssim"]})
                wandb_epoch_dict.update({f'contrast1_psnr': metrics_contrast1["psnr"]})
                wandb_epoch_dict.update({f'contrast1_lpips': metrics_contrast1["lpips"]})
                wandb_epoch_dict.update({f'contrast2_ssim': metrics_contrast2["ssim"]})
                wandb_epoch_dict.update({f'contrast2_psnr': metrics_contrast2["psnr"]})
                wandb_epoch_dict.update({f'contrast2_lpips': metrics_contrast2["lpips"]})
                wandb_epoch_dict.update({f'mutual_information': metrics_mi_pred["mi"]})
                wandb_epoch_dict.update({f'mutual_information_appox': metrics_mi_approx["mi"]})
                wandb_epoch_dict.update({f'MI_error_contrast1': np.abs(metrics_mi_1["mi"]-metrics_mi_true["mi"])})
                wandb_epoch_dict.update({f'MI_error_contrast2': np.abs(metrics_mi_2["mi"]-metrics_mi_true["mi"])})
                wandb_epoch_dict.update({f'MI_error_pred': np.abs(metrics_mi_pred["mi"]-metrics_mi_true["mi"])})


            img = nib.Nifti1Image(img_contrast1, affine)

            if epoch == (config.TRAINING.EPOCHS -1):
                nib.save(img, os.path.join(image_dir, model_name_epoch.replace("model.pt", f"_ct1.nii.gz")))

            slice_0 = img_contrast1[int(x_dim/2), :, :]
            slice_1 = img_contrast1[:, int(y_dim/2), :]
            slice_2 = img_contrast1[:, :, int(z_dim/2)]

            img = nib.Nifti1Image(img_contrast2, affine)
            if epoch == (config.TRAINING.EPOCHS -1):
                nib.save(img, os.path.join(image_dir, model_name_epoch.replace("model.pt", f"_ct2.nii.gz")))

            bslice_0 = gt_contrast1[int(x_dim/2), :, :]
            bslice_1 = gt_contrast1[:, int(y_dim/2), :]
            bslice_2 = gt_contrast1[:, :, int(z_dim/2)]

            im = show_slices_gt([slice_0, slice_1, slice_2],[bslice_0, bslice_1, bslice_2], epoch)
            if args.logging:
                image = wandb.Image(im, caption=f"{config.DATASET.LR_CONTRAST1} prediction vs gt.")
                wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST1}": image})

            slice_0 = img_contrast2[int(x_dim/2), :, :]
            slice_1 = img_contrast2[:, int(y_dim/2), :]
            slice_2 = img_contrast2[:, :, int(z_dim/2)]

            bslice_0 = gt_contrast2[int(x_dim/2), :, :]
            bslice_1 = gt_contrast2[:, int(y_dim/2), :]
            bslice_2 = gt_contrast2[:, :, int(z_dim/2)]

            im = show_slices_gt([slice_0, slice_1, slice_2],[bslice_0, bslice_1, bslice_2], epoch)
            if args.logging:
                image = wandb.Image(im, caption=f"{config.DATASET.LR_CONTRAST2} prediction vs gt.")
                wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2}": image})
                
            wandb.log(wandb_epoch_dict)  # update logs per epoch
            
            mi_buffer[1:] = mi_buffer[:-1]  # shifting buffer
            mi_buffer[0] = metrics_mi_pred["mi"]  # update buffer
            curr_mean = np.mean(np.abs(mi_buffer[:-1]-mi_buffer[1:]))  # srtore diff of abs change
            print("Current buffer: ", mi_buffer, "mean:", curr_mean)

            if np.abs(curr_mean-mi_mean)<0.0001:
                if args.early_stopping:
                    print("Early stopping training", mi_buffer)
                    break

            else:
                mi_mean = curr_mean

        else:

            model_intensities = model_intensities[:,0] 

            scaler = MinMaxScaler()
            label_arr = np.array(model_intensities, dtype=np.float32)
            model_intensities= scaler.fit_transform(label_arr.reshape(-1, 1))
            inference_time = time.time() - start
            if args.logging:
                wandb_epoch_dict.update({'inference_time': inference_time})

            print("Generating NIFTIs.")
            gt_contrast1 = dataset.get_contrast1_gt().reshape((x_dim, y_dim, z_dim)).cpu().numpy()
            gt_contrast2 = dataset.get_contrast2_gt().reshape((x_dim, y_dim, z_dim)).cpu().numpy()
            if config.TRAINING.CONTRAST2_ONLY:
                gt = gt_contrast2
                gt_other = gt_contrast1
            else:
                gt = gt_contrast1
                gt_other = gt_contrast2

            label_arr = np.array(gt, dtype=np.float32)
            gt = scaler.fit_transform(gt.reshape(-1, 1)).reshape((x_dim, y_dim, z_dim))

            img = model_intensities.reshape((x_dim, y_dim, z_dim))#.cpu().numpy()
            pred = img
            metrics = compute_metrics(gt=gt.copy(), pred=pred.copy(), mask=dataset.get_contrast1_gt_mask(), lpips_loss=lpips_loss, device=device)
            metrics_mi_true = compute_mi_hist(gt.copy(), gt_other.copy(), dataset.get_contrast2_gt_mask(), bins=32)
            metrics_mi = compute_mi_hist(pred.copy(), gt_other.copy(), dataset.get_contrast2_gt_mask(), bins=32)

            if args.logging:
                if config.TRAINING.CONTRAST2_ONLY:
                    wandb_epoch_dict.update({f'contrast2_ssim': metrics["ssim"]})
                    wandb_epoch_dict.update({f'contrast2_psnr': metrics["psnr"]})
                    wandb_epoch_dict.update({f'contrast2_lpips': metrics["lpips"]})
                    wandb_epoch_dict.update({f'MI_error_contrast2': np.abs(metrics_mi["mi"]-metrics_mi_true["mi"])})

                else:
                    wandb_epoch_dict.update({f'contrast1_ssim': metrics["ssim"]})
                    wandb_epoch_dict.update({f'contrast1_psnr': metrics["psnr"]})
                    wandb_epoch_dict.update({f'contrast1_lpips': metrics["lpips"]})
                    wandb_epoch_dict.update({f'MI_error_contrast1': np.abs(metrics_mi["mi"]-metrics_mi_true["mi"])})

            if config.TRAINING.CONTRAST2_ONLY:
                nifti_name = model_name_epoch.replace("model.pt", f"_ct2.nii.gz")
            else:
                nifti_name = model_name_epoch.replace("model.pt", f"_ct1.nii.gz")

            slice_0 = img[int(x_dim/2), :, :]
            slice_1 = img[:, int(y_dim/2), :]
            slice_2 = img[:, :, int(z_dim/2)]

            bslice_0 = gt[int(x_dim/2), :, :]
            bslice_1 = gt[:, int(y_dim/2), :]
            bslice_2 = gt[:, :, int(z_dim/2)]

            im = show_slices_gt([slice_0, slice_1, slice_2],[bslice_0, bslice_1, bslice_2], epoch)
            if args.logging:
                if config.TRAINING.CONTRAST2_ONLY:
                    image = wandb.Image(im, caption=f"{config.DATASET.LR_CONTRAST2} prediction vs gt.")
                    wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST2}": image})
                else:
                    image = wandb.Image(im, caption=f"{config.DATASET.LR_CONTRAST1} prediction vs gt.")
                    wandb_epoch_dict.update({f"{config.DATASET.LR_CONTRAST1}": image})

            affine = np.array(mgrid_affine)
            img = nib.Nifti1Image(img, affine)
            if epoch == (config.TRAINING.EPOCHS -1):
                nib.save(img, os.path.join(image_dir, nifti_name))
            
            if args.logging:
                wandb.log(wandb_epoch_dict)  # update logs per epoch


if __name__ == '__main__':
    args = parse_args()
    pr = cProfile.Profile()

    # python 3.8
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    # stats.dump_stats(filename='code_profiling.prof')


    # python 3.6
    # pr.enable()
    main(args)
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('time', 'cumulative')
    # ps.dump_stats(filename='code_profiling_improved.prof')
