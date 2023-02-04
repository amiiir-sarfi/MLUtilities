from config import Config
from feature_extractor import Feature_Extractor
from knn_probe import KNN_Probe
from transfer import Trainer as TransferTrainer
import wandb


if __name__ == "__main__":
    cfg = Config().parse(None)
    
    # Feature extraction:    
    if cfg.task == 'transfer_linear':
        proj_name = 'transfer'
    if cfg.task == 'knn_probe':
        proj_name = 'probe'
    proj_name = proj_name if cfg.wandb_project_name=='' else cfg.wandb_project_name
    
    if not cfg.no_wandb:
        wandb.init(project=proj_name, entity=cfg.wandb_entity, name=cfg.wandb_experiment_name, config=cfg, mode = cfg.wandb_offline)
    
    if cfg.extract_features:
        for ckpt_path, feature_path in zip(cfg.ckpt_full_paths, cfg.features_full_paths):
            Feature_Extractor(cfg, ckpt_path, feature_path).extract()
        
    for ckpt_path, feature_path, ckpt_info in zip(cfg.ckpt_full_paths, cfg.features_full_paths, cfg.ckpt_info):    
        if cfg.task == 'knn_probe':
            KNN_Probe(cfg, feature_path, ckpt_info).probe()
        elif 'transfer' in cfg.task:
            TransferTrainer(cfg, ckpt_path, ckpt_info).fit()
            

    