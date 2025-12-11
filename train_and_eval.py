import torch
from utils.print_utils import *
import os
from utils.lr_scheduler import get_lr_scheduler
from utils.metric_utils import *
import gc
from utils.utils import *
from utils.build_dataloader import build_data_loader
from utils.build_model import build_model
from utils.build_optimizer import build_optimizer, update_optimizer, read_lr_from_optimzier
from utils.build_criterion import build_criterion
from utils.build_backbone import BaseFeatureExtractor
import numpy as np
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print('scikit-survival not installed. Exiting...')
    raise
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from copy import deepcopy
import torch
import csv
from utils.criterions.survival_loss import *


def find_duplicates_with_indices(lst):
    count_dict = {}
    duplicates = {}
    for index, item in enumerate(lst):
        if item in count_dict:
            count_dict[item] += 1
            duplicates[item].append(index)
        else:
            count_dict[item] = 1
            duplicates[item] = [index]
    return {item: indices for item, indices in duplicates.items() if count_dict[item]>1}

def compute_metric(output, target, is_show=False):
    with torch.no_grad():
        y_pred = output.detach().cpu().numpy()
        y_true = target.detach().cpu().numpy()
        ind = np.where(y_true != -1)
        y_pred = y_pred[ind]
        y_true = y_true[ind]

        if is_show:
            y_pred_true = []
            for i in range(y_pred.shape[0]):
                y_pred_true.append((y_pred[i], y_true[i]))
            y_pred_true = sorted(y_pred_true, key=lambda x: x[0])
            for yp, yt in y_pred_true:
                print(yp, yt)
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return -1.0


class Trainer(object):
    '''This class implemetns the training and validation functionality for training ML model for medical imaging'''

    def __init__(self, opts, printer):
        super().__init__()
        self.opts = opts
        self.best_c_index = 0
        self.start_epoch = 1
        self.printer = printer
        self.global_setter()  
 
    def global_setter(self):
        self.setup_device()
        self.setup_dataloader()
        self.setup_model_optimizer_lossfn()
        self.setup_lr_scheduler()

    def setup_device(self):
        num_gpus = torch.cuda.device_count()
        self.num_gpus = num_gpus
        if num_gpus > 0:
            print_log_message('Using {} GPUs'.format(num_gpus), self.printer)
        else:
            print_log_message('Using CPU', self.printer)

        self.device = torch.cuda.current_device()
        self.use_multi_gpu = True if num_gpus > 1 else False
        self.opts.device = self.device
        print_log_message(f'Using {self.device}', self.printer)


    def setup_lr_scheduler(self):
        self.lr_scheduler = get_lr_scheduler(self.opts, printer=self.printer)

    def setup_dataloader(self):
        train_loader, test_loader, diag_classes, class_weights = build_data_loader(opts=self.opts, printer=self.printer)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.diag_classes = diag_classes
        self.class_weights = torch.from_numpy(class_weights)

    def setup_model_optimizer_lossfn(self):
        # Build Model
        mi_model = build_model(opts=self.opts, printer=self.printer)

        mi_model = mi_model.to(device=self.device)
        if self.use_multi_gpu:
            mi_model = torch.nn.DataParallel(mi_model)
        self.mi_model = mi_model
        if self.opts.seed == 1:
            print_info_message(f'Model Archetecture\n{self.mi_model}', printer=self.printer)

        # Build Loss function
        criterion = build_criterion(opts=self.opts, class_weights=self.class_weights.float(), printer=self.printer)
        self.criterion = criterion.to(device=self.device)

        # Build optimizer
        self.optimizer = build_optimizer(model=self.mi_model, opts=self.opts, printer=self.printer)

    def training(self, epoch, epochs, lr, *args, **kwargs):
        train_stats = Statistics(printer=self.printer)
        self.mi_model.train()
        self.optimizer.zero_grad()
        num_batches = len(self.train_loader) # number of batches
        epoch_start_time = time.time()
        risk_score_lst, true_diag_labels_lst, surv_times_lst = [], [], []
        surv_loss_lst, align_loss_lst, loss_lst, c_index_lst = [], [], [], []

        for batch_id, batch in enumerate(self.train_loader):
            for key in batch:
                if not isinstance(batch[key][0], str):
                    batch[key] = batch[key].float().to(device=self.device)
                    
            results = self.mi_model(batch, opts=self.opts)
        
            results['risk_score'] = results['risk_score'].squeeze()
            results['rfs'] = batch['rfs']
            results['label'] = batch['label']
            pids = batch['id']
            
            # if (torch.isnan(results['risk_score']).any().item()): 
            #     print_log_message(f'Training Epoch {epoch} batch {batch_id} output Nan risk score, discard', printer=self.printer)
            #     continue

            if (np.asarray(results['label'].cpu()) == 0).all(): 
                print_log_message(f'Training Epoch {epoch} batch {batch_id} without event sample!',printer=self.printer)
                continue
            # else: 
                # max if one patient has multiple risk_scores
                # if len(pids) != len(set(pids)):
                #     discard_idx = []
                #     for _, idx in find_duplicates_with_indices(pids).items():
                #         results['risk_score'][idx[0]] = max([results['risk_score'][i] for i in idx])
                #         discard_idx.extend(idx[1:])
                #     mask = torch.ones_like(results['risk_score'], dtype=torch.bool)
                #     mask[discard_idx] = False
                #     results['risk_score'] = results['risk_score'][mask]
                #     results['label'] = results['label'][mask]
                #     results['rfs'] = results['rfs'][mask]
                    
                # c_index = concordance_index(results['rfs'].cpu(), results['risk_score'].cpu().detach().numpy(), results['label'].cpu())
                # c_index_lst.append(c_index)
                
            surv_loss, align_loss, total_loss = self.criterion(results)
            
            if total_loss is not None:
                (total_loss / self.opts.log_interval).backward() # accumulate gradients
                torch.nn.utils.clip_grad_norm_(self.mi_model.parameters(), max_norm=20, norm_type=2)
                loss_lst.append(total_loss.item())
                surv_loss_lst.append(surv_loss.item())
                align_loss_lst.append(align_loss.item())

            if (batch_id+1) % self.opts.log_interval == 0 or (batch_id+1) == num_batches: 
                self.optimizer.step() # perform gradient descent every log_interval batches
                self.optimizer.zero_grad()

            # if (torch.isnan(results['risk_score']).any().item()): 
            #     print_log_message(f'Training Epoch {epoch} batch {batch_id} output Nan risk score, discard', printer=self.printer)
            
            risk_score_lst.extend(results['risk_score'].cpu().detach().numpy())
            surv_times_lst.extend(results['rfs'].cpu())
            true_diag_labels_lst.extend(results['label'].cpu())

        avg_surv_loss = np.mean(surv_loss_lst)
        avg_align_loss = np.mean(align_loss_lst)
        avg_c_index = concordance_index_censored(event_indicator=np.array(true_diag_labels_lst).astype(bool),event_time=surv_times_lst,estimate=risk_score_lst)[0]
        train_stats.update(loss=np.mean(loss_lst), c_index=avg_c_index) # np.mean(c_index_lst)
        if epoch % 100 == 0 or epoch == self.opts.epochs:
            train_stats.output(epoch=epoch, batch=batch_id+1, n_batches=num_batches, start=epoch_start_time, lr=lr)
       

        return train_stats.avg_c_idnex(), train_stats.avg_loss(), avg_surv_loss, avg_align_loss

    def test(self, epoch, lr, *args, **kwargs):
        self.mi_model.eval()
        risk_scores, surv_times, labels, pids = [], [], [], []
        surv_loss_lst, align_loss_lst, loss_lst, Pinfo_attenS = [], [], [], []
        info_lst = {'slide_wsi_id':[], 'slide_us_id':[]}

        with torch.no_grad():
            for batch_id, batch in enumerate(self.test_loader): 
                for key in batch:
                    # print(key)
                    if key in info_lst.keys():
                        info_lst[key].extend(batch[key])
                    elif not isinstance(batch[key][0], str): # for tensor,float and int. Excluding patient_id, wsi_id and us_id
                        batch[key] = batch[key].float().to(device=self.device)

                results = self.mi_model(batch, opts=self.opts)
                _pids=batch['id']
                results['label'] = batch['label']
                results['rfs'] = batch['rfs']
                results['risk_score'] = results['risk_score'].squeeze()
                # print(f'batch{batch_id} label:{results['label'].shape}  times: {results['rfs'].shape}  risks: {results['risk_score'].shape}')
                # if batch_id < 5:
                #     print_log_message(f'Test Epoch {epoch} batch {batch_id} pid: {_pids[:5]}',printer=self.printer)

                # if (torch.isnan(results['risk_score']).any().item()): 
                #     print_log_message(f'Test Epoch {epoch} batch {batch_id} output Nan risk score, discard', printer=self.printer)
                #     print_log_message(f'{batch['slide_wsi_id']}\n{results['risk_score']}', printer=self.printer)
                #     continue

                if (np.asarray(batch['label'].cpu()) == 0).all(): 
                    print_log_message(f'Test Epoch {epoch} batch {batch_id} without event sample, discard', printer=self.printer)
                    continue
                # else:
                    # if len(_pids) != len(set(_pids)):
                    #     discard_idx = []
                    #     for _, idx in find_duplicates_with_indices(_pids).items():
                    #         results['risk_score'][idx[0]] = max([results['risk_score'][i] for i in idx])
                    #         discard_idx.extend(idx[1:])
                    #     mask = torch.ones_like(results['risk_score'], dtype=torch.bool)
                    #     mask[discard_idx] = False
                    #     results['risk_score'] = results['risk_score'][mask]
                    #     results['rfs'] = results['rfs'][mask]
                    #     results['label'] = results['label'][mask]
                    #     _pids = _pids[mask]

                    # c_index = concordance_index(results['rfs'].cpu(), results['risk_score'].cpu(), results['label'].cpu())
                    # c_index_lst.append(c_index)

                surv_loss, align_loss, total_loss = self.criterion(results)
                risk_scores.extend(results['risk_score'].squeeze().cpu())
                surv_times.extend(results['rfs'].cpu())
                labels.extend(results['label'].cpu())
                pids.extend(_pids)
                Pinfo_attenS.extend(self.mi_model.info_dict["clinical_weight"])

                loss_lst.append(total_loss.item())
                surv_loss_lst.append(surv_loss.item())
                align_loss_lst.append(align_loss.item())

                torch.cuda.empty_cache()
                
        avg_loss = np.mean(loss_lst)
        avg_surv_loss = np.mean(surv_loss_lst)
        avg_align_loss = np.mean(align_loss_lst)
        avg_c_index = concordance_index_censored(event_indicator=np.array(labels).astype(bool),event_time=surv_times,estimate=risk_scores)[0]
        
        # save patient information attention scores (inference modal)
        if epoch == -1 and self.opts.external_val:
            pred_save_dir = os.path.join(self.opts.save_dir, str(self.opts.seed)+"_Pinfo_attenS")
            os.makedirs(pred_save_dir, exist_ok=True)
            csv_path = os.path.join(pred_save_dir, f"{epoch:03d}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["id", "label","HER2","Molecularsubtype","Tumorsize","LNstatus","Grade","Age","ER","PR","Ki67"]
                writer.writerow(header)

                for i in range(len(labels)):
                    row = [pids[i], labels[i]] + [float(Pinfo_attenS[i][j]) for j in range(9)]
                    writer.writerow(row)

        # save risk scores
        if epoch == -1 or epoch % 100 == 0 or epoch == self.opts.epochs:
            # print_log_message(f'pids: {len(pids)},slide_wsi_id: {len(info_lst["slide_wsi_id"])},slide_us_id: {len(info_lst["slide_us_id"])}, \
            #     surv_times: {len(surv_times)}, labels: {len(labels)}, risk_scores: {len(risk_scores)}', printer=self.printer)
            pred_save_dir = os.path.join(self.opts.save_dir, str(self.opts.seed)+"_pred")
            os.makedirs(pred_save_dir, exist_ok=True)
            f = open(os.path.join(os.path.join(pred_save_dir, f'{epoch:03d}.csv')), 'w')
            f.write("id,slide_wsi_id,slide_us_id,RFS_day,label,risk_score\n") # 
            for i in range(len(labels)):
                f.write(f'{pids[i]},{info_lst["slide_wsi_id"][i]},{info_lst["slide_us_id"][i]},{surv_times[i]},{labels[i]},{risk_scores[i]}\n') # 
            f.close()

            print_log_message('* Test Loss: {:.4f}, C_index(RFS): {:.4f}'.format(
                            avg_loss, avg_c_index), printer=self.printer)

        return avg_c_index, avg_loss, avg_surv_loss, avg_align_loss

    def run(self, *args, **kwargs):
        kwargs['need_attn'] = False
        
        res_dict = {
            "TrainingLoss": [],
            "TrainingCIndex": [],
            # "TrainingAlignLoss": [],
            # "TrainingCoxLoss": [],
            "TestLoss": [],
            "TestCIndex": [],
            # "TestAlignLoss": [],
            # "TestCoxLoss": [],
        }

        test_c_index, test_loss, test_surv_loss, test_align_loss = self.test(epoch=-1, lr=self.opts.lr, args=args, kwargs=kwargs)
        if self.opts.external_val: 
            return
        
        for epoch in range(self.start_epoch, self.opts.epochs+1):
            epoch_lr = self.lr_scheduler.step(epoch)
            self.optimizer = update_optimizer(optimizer=self.optimizer, lr_value=epoch_lr)
            train_c_index, train_loss, surv_loss, align_loss = self.training(epoch=epoch, lr=epoch_lr, epochs=self.opts.epochs, args=args, kwargs=kwargs)
            test_c_index, test_loss, test_surv_loss, test_align_loss = self.test(epoch=epoch, lr=epoch_lr, args=args, kwargs=kwargs)
            gc.collect()

            # save checkpoint and best model per epoch
            is_best = test_c_index > self.best_c_index
            self.best_c_index = max(test_c_index, self.best_c_index)
            if is_best or (epoch == self.opts.epochs):
                model_state = self.mi_model.module.state_dict() if isinstance(self.mi_model, torch.nn.DataParallel) else self.mi_model.state_dict()
                optimizer_state = self.optimizer.state_dict()
                model_fname = save_checkpoint(epoch=epoch,
                                model_state=model_state,
                                optimizer_state=optimizer_state,
                                save_dir=self.opts.save_dir,
                                is_best=is_best,
                                is_last=(epoch == self.opts.epochs),
                                printer=self.printer,
                                )
            
            # plot result
            res_dict["TrainingLoss"].append(train_loss)
            res_dict["TrainingCIndex"].append(train_c_index)
            # res_dict["TrainingCoxLoss"].append(surv_loss)
            # res_dict["TrainingAlignLoss"].append(align_loss)

            res_dict["TestLoss"].append(test_loss)
            # res_dict["TestCoxLoss"].append(test_surv_loss)
            # res_dict["TestAlignLoss"].append(test_align_loss)
            res_dict["TestCIndex"].append(test_c_index)
        plot_results(res_dict, os.path.join(self.opts.save_dir, "result"))

        # subres = {}
        # subres["TrainingLoss"] = res_dict["TrainingLoss"][-50:]
        # subres["TestLoss"] = res_dict["TestLoss"][-50:]
        # # subres["TrainingCoxLoss"] = res_dict["TrainingCoxLoss"][-50:]
        # # subres["TrainingAlignLoss"] = res_dict["TrainingAlignLoss"][-50:]
        # # subres["TestCoxLoss"] = res_dict["TestCoxLoss"][-50:]
        # # subres["TestAlignLoss"] = res_dict["TestAlignLoss"][-50:]
        # plot_results(subres, os.path.join(self.opts.save_dir, "subres"), prekey=self.opts.epochs-50)

        # print_log_message('Additional Test', printer=self.printer) # checked Right
        # self.mi_model.load_state_dict(torch.load(model_fname))
        # test_c_index, test_loss, test_surv_loss, test_align_loss = self.test(epoch=-1, lr=self.opts.lr, args=args, kwargs=kwargs)
        # print_log_message(f'Model state\n{self.mi_model.state_dict()}', printer=self.printer)
        

