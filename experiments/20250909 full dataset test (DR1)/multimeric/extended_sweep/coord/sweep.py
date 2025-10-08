import os, sys, time, random, torch, wandb
import numpy as np
from torch import nn
import torch_geometric as pyg
import matplotlib.pyplot as plt

root_dir = os.path.normpath(os.getcwd()+'/..' * 5)
package_dir = os.path.join(root_dir, 'src')
exp_root_dir = os.path.join(os.getcwd()+'/..' * 3)
sys.path.append(package_dir)

from modules.training.model_arch import SimplifiedMultiGCN, model_version
from modules.training.trainer import Trainer
from modules.training.metrics import (
    comp_tp_fp_tn_fn,
    comp_metrics_avg,
)
from modules.data.datasets import Dataset #DeepSTABp_Dataset
# from ml_modules.data.transforms import norm_0to1
import modules.training.visualization as vis


# fix random generator for reproducibility
random_seed = 69
torch.manual_seed(random_seed)
rand_gen = torch.Generator().manual_seed(random_seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




# def backbone_merge(data):
#     if data['residue', 'backbone-complementary', 'residue'].edge_index.numel():
#         data['residue', 'merged', 'residue'].edge_index = torch.cat((
#             data['residue', 'merged', 'residue'].edge_index,
#             data['residue', 'backbone-complementary', 'residue'].edge_index
#         ), dim=1)
#     return data

########################################################################
# SETUP
########################################################################

### REPRESENTATION PARAMETERS

merge_edge_types = False

use_pi = False


### DATASET PARAMETERS

set_types = ['train', 'valid', 'test']

identifier_files = {
    set_type: f'{exp_root_dir}/{set_type}_identifiers.txt'
    for set_type in set_types
}
annotation_files = {
    set_type: f'{exp_root_dir}/{set_type}_annotations.csv'
    for set_type in set_types
}

# fold_file = f'{exp_root_dir}/the fold.json'
dataset_version = 'v0'
transform = None

entries_should_be_ready = False


### TRAINING SETUP (SWEEP DEFINITION)

n_epochs = 40

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'best valid loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {'values': [8, 16, 32, 64, 128, 256]}, # {'min': 8, 'max': 128},
        'learning_rate': {'values': [0.1, 0.01, 0.001, 0.0001]}, # {'min': 0.0001, 'max': 0.1},

        'edge_types_to_use': {'values': [['coord']]},
        'edge_policy': {'values': ['1CONT']},
        'use_monomers': {'values': [False]},

        'node_feat_name': {'values': ['x']}, # 'res1hot'
        # 'node_feat_size': {'min': , 'max': },
        'gnn_type': {'values': ['gcn']}, # 'gat', 'gin'
        'gat_atten_heads': {'values': [None]},
        'n_gnn_layers': {'values': [1, 3, 5]}, # {'min': 1, 'max': 6},
        'dim_nodes': {'values': [8, 16, 32, 64, 128, 256, 512]}, # {'min': 8, 'max':512},
        'conv_norm': {'values': [True]}, # False
        'norm_graph_input': {'values': [False]}, # True
        'norm_graph_output': {'values': [False]}, # not working
        'graph_global_pool': {'values': ['mean', 'max']},
        'graph_dropout_rate': {'values': [0]}, # {'min': 0.0, 'max': 0.3},
        'dropfeat_rate': {'values': [0]}, # {'min': 0.0, 'max': 0.3},
        'dropedge_rate': {'values': [0]}, # {'min': 0.0, 'max': 0.3},
        'dropnode_rate': {'values': [0]}, # not working
        'jk_concat': {'values': [False, True]},

        'pi_dropout_rate': {'values': [None]},
        'dim_pi_embedding': {'values': [None]},

        'fc_hidden_ls': {'values': [None]},
        'n_fc_hidden_layers': {'values': [2]}, # {'min': 1, 'max': 5},
        'fc_norm': {'values': [True]}, # False
        'norm_fc_input': {'values': [False]}, # True
        'fc_dropout_rate': {'values': [0]}, # {'min': 0.0, 'max': 0.3},
    }
}


### OUTPUT EVALUATION SETUP

# thresholds for binary classification
n_thres_evals = 501
thres_list = np.linspace(0, 1, n_thres_evals)


# ### FILES
# history_file = 'training history (df_thres).csv'
# best_loss_file = 'training history - best by loss (df_thres).csv'
# best_f1_max_file = 'training history - best by f1_max (f1_max).csv'

### MACHINE SPECIFIC PARAMETERS

num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    # INITIALIZE RUN USING SWEEP CONFIG
    run = wandb.init(project='Multimeric PFP')
    config = wandb.config

    config.model_version = model_version

    print(f'Using {num_workers} workers for dataloading')
    print(f'Training on {device} device\n')

    ####################################################################
    # INSTANTIATE DATASET AND DATALOADER
    ####################################################################

    # with open(fold_file, 'r') as f:
    #     fold_acc = json.load(f)
    # assert len(fold_acc['train']) == len(fold_acc['valid'])
    # n_folds = len(fold_acc['train'])

    assembly_ids = {
        set_type: np.loadtxt(
            identifier_files[set_type],
            dtype=np.str_
        ) for set_type in set_types
    }
    annotations = {
        set_type: np.loadtxt(
            annotation_files[set_type],
            delimiter=',',
            dtype=np.int32
        ) for set_type in set_types
    }

    n_GO_terms = annotations['train'].shape[1]
    assert annotations['valid'].shape[1] == n_GO_terms
    assert annotations['test'].shape[1] == n_GO_terms

    datasets = {
        set_type: Dataset(
            pdb_assembly_ids=assembly_ids[set_type],
            annotations=annotations[set_type],
            version='v0',
            sequence_embedding='ProtTrans',
            enm_type='anm',
            use_monomers=config.use_monomers,
            thresholds={
                'contact' : '12',
                'codir'   : config.edge_policy,
                'coord'   : config.edge_policy,
                'deform'  : config.edge_policy,
            },
            merge_edge_types=merge_edge_types,
            time_limit=60,
            transform=transform,
            device=device,
            entries_should_be_ready=False,
            rebuild=False
        ) for set_type in set_types
    }
    print()

    dataloaders = {
        set_type: pyg.loader.DataLoader(
            datasets[set_type],
            batch_size=config.batch_size,
            shuffle=(set_type=='train'), #flag
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=rand_gen,
        ) for set_type in set_types
    }

    ################################################################
    # INSTANTIATE MODEL, OPTIMIZER, AND LOSS FUNCTION
    ################################################################

    model = SimplifiedMultiGCN(
        dim_model_output=n_GO_terms,

        # FEATURE SELECTION
        use_pi=use_pi,

        # GRAPH CONVOLUTION SETUP
        node_feat_name=config.node_feat_name,
        node_feat_size=1024 if config.node_feat_name=='x' else 20,
        gnn_type=config.gnn_type,
        gat_atten_heads=config.gat_atten_heads,
        dim_node_hidden_dict={
            et: [config.dim_nodes]*config.n_gnn_layers
            for et in config.edge_types_to_use
        },
        conv_norm=config.conv_norm,
        norm_graph_input=config.norm_graph_input,
        norm_graph_output=config.norm_graph_output,
        graph_global_pool=config.graph_global_pool,
        graph_dropout_rate=config.graph_dropout_rate,
        dropfeat_rate=config.dropfeat_rate,
        dropedge_rate=config.dropedge_rate,
        dropnode_rate=config.dropnode_rate,
        jk_concat=config.jk_concat,

        # PERSISTENCE IMAGES SETUP
        pi_dropout_rate=config.pi_dropout_rate,
        dim_pi_embedding=config.dim_pi_embedding,

        # FC SETUP
        fc_hidden_ls=config.fc_hidden_ls,
        n_fc_hidden_layers=config.n_fc_hidden_layers,
        fc_norm=config.fc_norm,
        norm_fc_input=config.norm_fc_input,
        fc_dropout_rate=config.fc_dropout_rate,

        # OTHERS
        debug=False,
    )
    model.save_args('.')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, amsgrad=False
    )
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=0.0001, max_lr=0.01, mode='triangular',
    #     cycle_momentum=False
    # )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=datasets['train'].pos_weight)

    run.save('model-summary.txt')

    ################################################################
    # FILE & PLACEHOLDER SETUP
    ################################################################

    # ### FILE TO KEEP TRACK OF TRAINING PERFORMANCE
    # # training history
    # header = '# epoch,t_loss,v_loss,t_acc,v_acc,t_f1,v_f1,t_recall,v_recall,t_spec,v_spec,t_prec,v_prec'
    # with open(history_file, 'w+') as f:
    #     f.write(header + '\n')
    # # epoch where performance improves
    # with open(best_loss_file, 'w+') as f:
    #     f.write(header + '\n')
    # with open(best_f1_max_file, 'w+') as f:
    #     f.write(header + ',threshold_value\n')

    # # prediction on validation set
    # header = '# epoch,' + ','.join(valid_accessions_ordered)
    # line_1 = '# true_labels,' + ','.join([f'{Tm:.8f}' for Tm in valid_Tm])
    # with open(prediction_file_valid, 'w+') as f:
    #     f.write(header +'\n')
    #     f.write(line_1 +'\n')
    # # best prediction on validation set
    # with open(prediction_file_valid_best, 'w+') as f:
    #     f.write(header +'\n')
    #     f.write(line_1 +'\n')
    # # prediction on training set
    # header = '# epoch,' + ','.join(train_accessions_ordered)
    # line_1 = '# true_labels,' + ','.join([f'{Tm:.8f}' for Tm in train_Tm])
    # with open(prediction_file_train, 'w+') as f:
    #     f.write(header +'\n')
    #     f.write(line_1 +'\n')
    # # best prediction on training set
    # with open(prediction_file_train_best, 'w+') as f:
    #     f.write(header +'\n')
    #     f.write(line_1 +'\n')

    prec   = np.zeros((n_thres_evals,))
    recall = np.zeros((n_thres_evals,))
    spec   = np.zeros((n_thres_evals,))
    f1     = np.zeros((n_thres_evals,))
    acc    = np.zeros((n_thres_evals,))

    v_tp_fp_tn_fn_all = torch.zeros(
        (n_thres_evals, n_GO_terms, 4),
        dtype=torch.int32,
        device=device
    )

    ################################################################
    # train / valid loop
    ################################################################

    ### INSTANTIATE THE MODEL-TRAINING CONTROLLER
    trainer = Trainer(
        n_GO_terms=n_GO_terms,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_workers=num_workers,
        device=device,
    )

    ### TRAIN FOR N_EPOCHS
    best_v_loss = 1e8
    best_f1_max = 0
    for i in range(n_epochs):
        epoch = i + 1
        print(f'Epoch {epoch}')

        # # get learning rate of this epoch (NOTE: is this working properly?)
        # for param_group in optimizer.param_groups:
        #     current_lr = param_group['lr']

        # # time it
        # start = time.time()

        ### ONE PASS OVER TRAINING SET
        t_loss, t_outputs, t_labels = trainer.train_one_epoch(
            dataloaders['train']
        )

        ### ONE PASS OVER VALID SET
        v_loss, v_outputs, v_labels = trainer.evaluate(
            dataloaders['valid']
        )

        ### COMPUTE VARIOUS METRICS
        # validation metrics for all thresholds
        for idx, thres in enumerate(thres_list):
            v_tp_fp_tn_fn_all[idx] = comp_tp_fp_tn_fn(
                v_outputs, v_labels, thres=thres
            )
            prec[idx], recall[idx], spec[idx], f1[idx], acc[idx] = (
                comp_metrics_avg(v_tp_fp_tn_fn_all[idx])
            )

        # training metrics at thres=0.5
        t_tp_fp_tn_fn = comp_tp_fp_tn_fn(
            t_outputs, t_labels, thres=0.5
        )
        t_prec, t_recall, t_spec, t_f1, t_acc = comp_metrics_avg(
            t_tp_fp_tn_fn
        )

        # validation metrics at thres=0.5
        df_thres_idx = n_thres_evals//2+1
        v_prec, v_recall, v_spec, v_f1, v_acc = (
            prec[df_thres_idx], recall[df_thres_idx],
            spec[df_thres_idx], f1[df_thres_idx], acc[df_thres_idx]
        )

        # training metrics at threshold of f1 max on validation
        # CAUTION: f1_max is computed from the macro recall and prec
        #          instead of macro f1
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2*recall*prec / (recall+prec)
        f1_max_idx = np.nanargmax(f1)
        f1_max = f1[f1_max_idx]

        t_tp_fp_tn_fn_f1_max = comp_tp_fp_tn_fn(
            t_outputs, t_labels, thres=thres_list[f1_max_idx]
        )
        t_prec_f1_m, t_recall_f1_m, t_spec_f1_m, t_f1_f1_m, t_acc_f1_m = (
            comp_metrics_avg(t_tp_fp_tn_fn_f1_max)
        )

        # ### SAVE MODEL PERFORMANCE
        # line = f'{epoch},' + ','.join([f'{v:.8f}' for v in [
        #     t_loss,   v_loss,   t_acc,    v_acc,
        #     t_f1,     v_f1,     t_recall, v_recall,
        #     t_spec,   v_spec,   t_prec,   v_prec
        # ]])
        # with open(history_file, 'a+') as f:
        #     f.write(line + '\n')

        ### SOME ADDITIONAL ACTIONS IF PERFORMANCE IMPROVES
        if v_loss < best_v_loss:
            best_v_loss = v_loss
            # best_epoch = epoch
            # with open(best_loss_file, 'a+') as f:
            #     f.write(line + '\n')
            torch.save(
                model.state_dict(), 'model-best_loss.pt'
            )

        if f1_max > best_f1_max:
            # line_f1_m = f'{epoch},' + ','.join([f'{v:.8f}' for v in [
            #     t_loss,         v_loss,
            #     t_acc_f1_m,     acc[f1_max_idx],
            #     t_f1_f1_m,      f1[f1_max_idx],
            #     t_recall_f1_m,  recall[f1_max_idx],
            #     t_spec_f1_m,    spec[f1_max_idx],
            #     t_prec_f1_m,    prec[f1_max_idx],
            #     thres_list[f1_max_idx]
            # ]])

            best_f1_max = f1_max
            # best_f1_max_epoch = epoch
            # with open(best_f1_max_file, 'a+') as f:
            #     f.write(line_f1_m + '\n')
            torch.save(
                model.state_dict(), 'model-best_f1_max.pt'
            )

        ### PRINT METRICS
        print(f'      <LOSS>         '
                f'train: {t_loss:.4f}, valid: {v_loss:.4f}')
        print(f'    Threshold: 0.5')
        print(f'      <Accuracy>     '
                f'train: {t_acc:.4f}, valid: {v_acc:.4f}')
        print(f'      <Recall (BA)>  '
                f'train: {t_recall:.4f}, valid: {v_recall:.4f}')
        print(f'      <Macro F1>     '
                f'train: {t_f1:.4f}, valid: {v_f1:.4f}')

        print(f'    Threshold @ F1_max on PR Curve')
        print(f'      <Accuracy>     '
                f'train: {t_acc_f1_m:.4f}, '
                f'valid: {acc[f1_max_idx]:.4f}')
        print(f'      <Recall (BA)>  '
                f'train: {t_recall_f1_m:.4f}, '
                f'valid: {recall[f1_max_idx]:.4f}')
        print(f'      <F1>           '
                f'train: {t_f1_f1_m:.4f}, '
                f'valid: {f1_max:.4f}')
        print()

        run.log({
            'train/loss': t_loss,
            'valid/loss': v_loss,

            'train/accuracy@0.5': t_acc,
            'valid/accuracy@0.5': v_acc,
            'train/precision@0.5': t_prec,
            'valid/precision@0.5': v_prec,
            'train/recall@0.5': t_recall,
            'valid/recall@0.5': v_recall,
            'train/specificity@0.5': t_spec,
            'valid/specificity@0.5': v_spec,
            'train/macro_f1@0.5': t_f1,
            'valid/macro_f1@0.5': v_f1,

            'train/accuracy@f1_max': t_acc_f1_m,
            'valid/accuracy@f1_max': acc[f1_max_idx],
            'train/recall@f1_max': t_recall_f1_m,
            'valid/recall@f1_max': recall[f1_max_idx],
            'train/f1@f1_max': t_f1_f1_m,
            'valid/f1@f1_max': f1_max,
        }, step=epoch)

    ################################################################
    # PR CURVES AND MORE LOGGING
    ################################################################

    for index_name, (model_file, best_epoch) in {
        'loss': ('model-best_loss.pt', best_v_loss),
        'f1 max': ('model-best_f1_max.pt', best_f1_max)
    }.items():

        trainer.load_model_state_dict(
            torch.load(
                model_file,
                map_location=device
            )
        )

        tp_fp_tn_fn_all = torch.zeros(
            (n_thres_evals, n_GO_terms, 4),
            dtype=torch.int32,
            device=device
        )

        ### ONE PASS OVER TRAIN SET
        bt_loss, bt_outputs, bt_labels = trainer.evaluate(dataloaders['train'])
        for idx, thres in enumerate(thres_list):
            tp_fp_tn_fn_all[idx] = comp_tp_fp_tn_fn(
                bt_outputs, bt_labels, thres=thres
            )
            prec[idx], recall[idx], spec[idx], f1[idx], acc[idx] = (
                comp_metrics_avg(tp_fp_tn_fn_all[idx])
            )
        vis.plot_pr(
            None, prec, recall, thres_list,
            filename_suffix=f'train (best {index_name})',
            wandb_run=run
        )

        ### ONE PASS OVER VALID SET
        bv_loss, bv_outputs, bv_labels = trainer.evaluate(dataloaders['valid'])
        for idx, thres in enumerate(thres_list):
            tp_fp_tn_fn_all[idx] = comp_tp_fp_tn_fn(
                bv_outputs, bv_labels, thres=thres
            )
            prec[idx], recall[idx], spec[idx], f1[idx], acc[idx] = (
                comp_metrics_avg(tp_fp_tn_fn_all[idx])
            )
        vis.plot_pr(
            None, prec, recall, thres_list,
            filename_suffix=f'valid (best {index_name})',
            wandb_run=run
        )

        ### ONE PASS OVER TEST SET
        bte_loss, bte_outputs, bte_labels = trainer.evaluate(dataloaders['test'])
        for idx, thres in enumerate(thres_list):
            tp_fp_tn_fn_all[idx] = comp_tp_fp_tn_fn(
                bte_outputs, bte_labels, thres=thres
            )
            prec[idx], recall[idx], spec[idx], f1[idx], acc[idx] = (
                comp_metrics_avg(tp_fp_tn_fn_all[idx])
            )
        vis.plot_pr(
            None, prec, recall, thres_list,
            filename_suffix=f'test (best {index_name})',
            wandb_run=run
        )

        run.log_model(
            model_file, f'best_{index_name.replace(" ", "_")}'
        )

        if index_name == 'loss':
            run.summary['best valid loss'] = best_v_loss
            run.summary['test loss'] = bte_loss

    print(f'best valid loss: {best_v_loss:.4f} at epoch {best_epoch}')
    print(f'bv_loss: {bv_loss:.4f}\n')

    # ################################################################
    # # plot learning curve
    # ################################################################
    # history = np.loadtxt(history_file, delimiter=',')

    # # loss
    # plt.plot(history[:,0], history[:,1], label='training')
    # plt.plot(history[:,0], history[:,2], label='validation')
    # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel(f'loss (BCE)')
    # # plt.ylim(0, 1)
    # plt.savefig(
    #     'learning_curve-loss.png',
    #     dpi=300
    # )
    # plt.close()

if __name__ == '__main__': # REQUIRED TO ENABLE MULTIPROCESSING

    sweep_id = wandb.sweep(sweep_config, project='Multimeric PFP')
    wandb.agent(sweep_id, function=main, count=100)
