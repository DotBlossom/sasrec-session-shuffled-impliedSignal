from dataclasses import dataclass

# =====================================================================
# [Config] 파이프라인 설정 
# =====================================================================
@dataclass
class PipelineConfig:
    # Paths
    base_dir: str = r"D:\trainDataset\localprops"
    model_dir: str = r"C:\Users\candyform\Desktop\inferenceCode\models"
    ft_model_dir: str = r"C:\Users\candyform\Desktop\inferenceCode\models\finetune"

    # Hyperparameters
    batch_size: int = 768
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    
    # Model Args (SASRecUserTower용)
    d_model: int = 128
    max_len: int = 50
    dropout: float = 0.3
    pretrained_dim: int = 128 # 사전학습 아이템 벡터 차원 
    nhead: int = 4
    num_layers: int = 2
    
    # Loss Penalties
    lambda_logq: float = 1.0
    lambda_sup: float = 0.1
    lambda_cl: float = 0.2
   
    # [신규] HNM 제어 파라미터
    top_k_percent: float = 0.01 # 상위 15% 하드 네거티브 사용 (10~20% 사이 권장)
    hnm_threshold: float = 0.90
    hard_margin: float = 0.01

    # model 관리
    freeze_item_tower: bool = True
    item_tower_pth_name: str = "encoder_ep03_loss0.8129.pth"
    # 자동 할당될 메타데이터 크기
    num_items: int = 0
    num_prod_types: int = 0
    num_colors: int = 0
    num_graphics: int = 0
    num_sections: int = 0
    num_age_groups: int = 10

    max_target_len: int = 10