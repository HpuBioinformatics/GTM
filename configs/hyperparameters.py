import torch


def get_hyperparameters(exp_name=None):
    hp = {
        'device': 'cuda:2' if torch.cuda.is_available() else 'cpu',
        'seed': 1,
        'wandb_mode': 'disabled',
        'wandb_project': 'GNNome-dev',

      
        'pred_threshold': 0.65,
        'use_dynamic_threshold': False,
        'max_pos_weight': 300,

     
        'use_focal_loss': False,
        'focal_gamma': 2.0,

       
        'use_asl': True,
        'asl_gamma_pos': 3.0,
        'asl_gamma_neg': 1.0,
        'asl_clip': 0.0,

        'train_pos_keep_ratio': 5.0,

     
        'neg_aux_weight': 0.5,

      
        'chr_overfit': 0,
        'plot_nga50_during_training': False,
        'eval_frequency': 20,
        'batch_norm': True,

       
        'use_similarities': False,

      
        'dim_latent': 128,
        'num_gnn_layers': 8,
        'node_features': 4,
        'edge_features': 4,
        'training': True,
        'hidden_ne_features': 64,
        'hidden_edge_scores': 32,
        'hidden_edge_features': 64,
        'nb_pos_enc': 128,
        'type_pos_enc': 'none',
        'normalization': 'batch',
        'dropout': 0.2,
        "metis_num_parts": 8,
        "metis_max_parts": 128,
        "metis_halo_hops": 3,
        "metis_balance_edges": True,


    
        'num_epochs': 50,
        'lr': 1e-4,

        'use_symmetry_loss': False,
        'alpha': 0.1,

        'num_nodes_per_cluster': 3000,
        'k_extra_hops': 1,

        'patience': 4,
        'decay': 0.95,

        'masking': False,
        'masking_valid': False,
        'mask_frac_low': 80,
        'mask_frac_high': 100,

   
        'lambda_path': 0.05,
        'lambda_conf': 0.02,
        'conf_warmup_epochs': 8,

        'train_neg_ratio': 3.0,

    
        'strategy': 'greedy',
        'num_decoding_paths': 100,
        'decode_with_labels': False,
        'load_checkpoint': True,
        'num_threads': 32,
        'B': 1,
        'len_threshold': 70_000,

       
        'model_type': 'full',
        'directed': True,

        'num_transformer_layers': 4,
        'num_mamba_layers': 4,  
        
    }

    

    return hp
