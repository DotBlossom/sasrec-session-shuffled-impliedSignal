            
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import sys

from sheduler import EarlyStopping, get_cosine_schedule_with_warmup


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)
import wandb
from preprocessor.preprocessor_v2 import FeatureProcessor_v3, dataset_peek_v3
from v2_usetower_model import SASRecDataset_v2, SASRecDataset_v3, SASRecUserTower_v2, SASRecUserTower_v3
from v1_refine_usertower import FeatureProcessor, SASRecDataset, SASRecUserTower, dataset_peek, duorec_loss_refined, full_batch_hard_emphasis_loss, inbatch_corrected_logq_loss, inbatch_corrected_logq_loss_with_hard_neg, inbatch_corrected_logq_loss_with_hard_neg_margin, inbatch_corrected_logq_loss_with_shared_hard_neg, inbatch_hnm_corrected_loss_with_stats,inbatch_mixed_hnm_loss_with_stats


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
# =====================================================================
# Phase 1: Environment Setup
# =====================================================================
def setup_environment(seed: int = 42):
    """난수 고정 및 디바이스 설정 (Airflow Task 독립성 보장)"""
    print("\n⚙️ [Phase 1] Setting up environment...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device set to: {device}")
    return device




# =====================================================================
# Phase 2: Data Preparation
# =====================================================================
def prepare_features(cfg: PipelineConfig):
    """FeatureProcessor 초기화 및 메타데이터 업데이트 (로컬 캐싱 적용)"""
    print("\n📊 [Phase 2] Loading Processors...")
    
    # 1. 캐시 파일 경로 설정
    cache_path = os.path.join(cfg.base_dir, "processor_cache.pkl")
    
    # 2. 캐시 존재 여부 확인 및 로드
    if os.path.exists(cache_path):
        print(f"   ✅ [Cache Hit] Found cached processors at {cache_path}")
        print("   ⏳ Loading from local storage...")
        with open(cache_path, 'rb') as f:
            train_proc, val_proc = pickle.load(f)
            
    # 3. 캐시가 없을 경우: 원본 파라켓 로드 및 생성
    else:
        print("   ⚠️ [Cache Miss] Cache not found. Processing from Parquet files...")
        
        # 경로 설정
        user_path = os.path.join(cfg.base_dir, "features_user_w_meta_nonleak.parquet") 
        item_path = os.path.join(cfg.base_dir, "features_item.parquet")
        #seq_path = os.path.join(cfg.base_dir, "features_sequence_cleaned.parquet")
        
        TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")
        USER_VAL_FEAT_PATH = os.path.join(cfg.base_dir, "features_user_w_meta_nonleak_val.parquet")
        #SEQ_VAL_DATA_PATH = os.path.join(cfg.base_dir, "features_sequence_val.parquet")
        
        # Processor 초기화 
        train_proc = FeatureProcessor_v3(user_path, item_path)
        val_proc = FeatureProcessor_v3(USER_VAL_FEAT_PATH, item_path, base_processor=train_proc)
        
        # [신규] 생성된 Processor 객체를 로컬 파일로 저장 (HIGHEST_PROTOCOL로 속도/용량 최적화)
        print("   💾 Saving processors to local cache for future use...")
        with open(cache_path, 'wb') as f:
            pickle.dump((train_proc, val_proc), f, protocol=pickle.HIGHEST_PROTOCOL)

    # 4. Config 업데이트 (캐시에서 불러왔든 새로 만들었든 동일하게 적용)
    cfg.num_items = train_proc.num_items
    
    ####### 실제 item metadata id랑 묶인상태로 가져와야하고 연결 필요 #######
    cfg.num_prod_types = int(train_proc.i_side_arr[:, 0].max()) + 1
    cfg.num_colors = int(train_proc.i_side_arr[:, 1].max()) + 1
    cfg.num_graphics = int(train_proc.i_side_arr[:, 2].max()) + 1
    cfg.num_sections = int(train_proc.i_side_arr[:, 3].max()) + 1
    
    print(f"✅ Max hash needed: {cfg.num_prod_types+cfg.num_colors + cfg.num_graphics +  cfg.num_sections}")
    print(f"✅ Features Loaded. Total Items: {cfg.num_items}")
    
    return train_proc, val_proc, cfg
# =====================================================================
# Phase 3: Embedding Alignment & DataLoader
# =====================================================================
def load_aligned_pretrained_embeddings(processor, model_dir, pretrained_dim):
    """Dataset에서 사용할 수 있도록 정렬된 사전학습 벡터(N+1, Dim) 생성"""
    print(f"\n🔄 [Phase 3-1] Aligning Pretrained Item Embeddings...")
    emb_path = os.path.join(model_dir, "pretrained_item_matrix.pt")
    ids_path = os.path.join(model_dir, "item_ids.pt")

    num_embeddings = processor.num_items + 1 
    aligned_weight = torch.randn(num_embeddings, pretrained_dim) * 0.01 
    aligned_weight[0] = 0.0 # Padding
    
    try:
        pretrained_emb = torch.load(emb_path, map_location='cpu')
        if isinstance(pretrained_emb, dict):
            pretrained_emb = pretrained_emb.get('weight', pretrained_emb.get('item_content_emb.weight'))
        pretrained_ids = torch.load(ids_path, map_location='cpu')
        
        pretrained_map = {str(iid.item()) if isinstance(iid, torch.Tensor) else str(iid): pretrained_emb[idx] 
                          for idx, iid in enumerate(pretrained_ids)}
        
        matched = 0
        for i, current_id_str in enumerate(processor.item_ids):
            if current_id_str in pretrained_map:
                aligned_weight[i + 1] = pretrained_map[current_id_str]
                matched += 1
                
        print(f"✅ Matched: {matched}/{len(processor.item_ids)}")
    except Exception as e:
        print(f"⚠️ [Warning] Failed to load Pretrained files: {e}. Using random init.")
        
    return aligned_weight

def create_dataloaders(processor, cfg: PipelineConfig, global_now_str, aligned_pretrained_vecs=None, is_train=True):
    """Dataset 및 DataLoader 인스턴스화"""
    mode_str = "Train" if is_train else "Validation"
    print(f"\n📦 [Phase 3-2] Creating {mode_str} DataLoaders...")
    
    # 💡 1. is_train 파라미터 전달
    dataset = SASRecDataset_v3(processor, global_now_str = global_now_str, max_len=cfg.max_len, is_train=is_train)
    
    # Dataset 인스턴스에 정렬된 pretrained vector 룩업 테이블 주입
    dataset.pretrained_lookup = aligned_pretrained_vecs 
    
    loader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        # 💡 2. 검증 시에는 셔플을 끄고, 자투리 데이터(마지막 배치)도 버리지 않고 모두 평가
        shuffle=is_train, 
        num_workers=0, 
        pin_memory=True,
        drop_last=is_train 
    )
    
    print(f"✅ {mode_str} Loader Ready: {len(loader)} batches/epoch")
    return loader

def load_item_tower_state_dict(model_dir: str, pth_filename: str, device):
    """
    [Data/IO] 물리적 파일(.pth)을 읽어 메모리(state_dict)로 올리는 순수 IO 역할.
    모델 구조나 학습 상태(Freeze 여부)에는 절대 관여하지 않음.
    """
    file_path = os.path.join(model_dir, pth_filename)
    
    if not os.path.exists(file_path):
        print(f"⚠️ [IO Warning] Item Tower file not found: {file_path}")
        print("   -> Random initialization will be used.")
        return None
        
    print(f"📥 [IO] Loading Item Tower weights from {pth_filename}...")
    
    try:
        # map_location을 통해 CPU/GPU 메모리 매핑 최적화
        state_dict = torch.load(file_path, map_location=device)
        return state_dict
    except Exception as e:
        print(f"❌ [IO Error] Failed to load .pth file: {e}")
        return None
    
import hashlib
import json

def get_hash_id(text, hash_size):
    """문자열을 일관된 정수 ID(1 ~ hash_size)로 해싱 (0은 Padding)"""
    if not text or str(text).lower() in ['unknown', 'nan', 'none']:
        return 0
    # MD5를 사용하여 파이썬 세션이 바뀌어도 항상 동일한 해시값 보장
    hash_obj = hashlib.md5(str(text).strip().lower().encode('utf-8'))
    # 16진수를 정수로 변환 후 hash_size로 나눈 나머지 + 1
    return (int(hash_obj.hexdigest(), 16) % hash_size) + 1

def load_item_metadata_hashed(processor, base_dir, hash_size=1000):
    """JSON 파일을 읽어 정렬된 메타데이터 해시 텐서(N+1, 4)를 생성"""
    print("\n🏷️ [Phase 3-2] Loading and Hashing Item Metadata...")
    json_path = os.path.join(base_dir, "filtered_data_reinforced.json")
    
    num_items = processor.num_items + 1
    # 0번 인덱스는 패딩을 위해 0으로 유지 (N+1, 4차원 배열)
    item_side_arr = np.zeros((num_items, 4), dtype=np.int64)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            item_data = json.load(f)
    except Exception as e:
        print(f"❌ [Error] Failed to load JSON: {e}")
        return torch.tensor(item_side_arr, dtype=torch.long)
    
    # 빠른 검색을 위해 O(1) Lookup Dictionary 생성
    # int형 article_id를 string으로 변환하여 매핑
    metadata_dict = {str(item.get('article_id', '')): item for item in item_data}
    
    matched = 0
    for i, current_id_str in enumerate(processor.item_ids):
        idx = i + 1 # 1-based indexing
        
        if current_id_str in metadata_dict:
            meta = metadata_dict[current_id_str]
            
            # 카테고리 매핑 및 해싱 (해당 키가 없으면 빈 문자열 반환)
            type_val = meta.get("product_type_name", "")
            color_val = meta.get("colour_group_name", "")
            graphic_val = meta.get("graphical_appearance_name", "")
            section_val = meta.get("section_name", "")
            
            item_side_arr[idx, 0] = get_hash_id(type_val, hash_size)
            item_side_arr[idx, 1] = get_hash_id(color_val, hash_size)
            item_side_arr[idx, 2] = get_hash_id(graphic_val, hash_size)
            item_side_arr[idx, 3] = get_hash_id(section_val, hash_size)
            
            matched += 1

    print(f"✅ Metadata Matched & Hashed: {matched}/{len(processor.item_ids)} (Hash Size: {hash_size})")
    
    del item_data
    del metadata_dict
    import gc
    gc.collect()
    
    return torch.tensor(item_side_arr, dtype=torch.long)
# =====================================================================
# Phase 4: Model Setup
# =====================================================================
class SASRecItemTower(nn.Module):
    def __init__(self, num_items, d_model, log_q_tensor=None):
        super().__init__()
        
        # 💡 단순히 임베딩이라기보다 '미세조정 가능한 아이템 벡터 행렬'임을 명시
        self.item_matrix = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        
        if log_q_tensor is not None:
            self.register_buffer('log_q', log_q_tensor)
        else:
            self.register_buffer('log_q', torch.zeros(num_items + 1))

    def get_all_embeddings(self):
        return self.item_matrix.weight

    def get_log_q(self):
        return self.log_q
        
    def set_freeze_state(self, freeze: bool):
        for param in self.parameters():
            param.requires_grad = not freeze
        # SASRecItemTower 클래스 내부에 추가
    def get_embeddings(self, item_ids):
        """
        특정 아이템 ID들에 대해서만 임베딩을 추출 (VRAM 절약용)
        item_ids: [num_unique_ids] 형태의 텐서
        """
        # item_matrix 레이어를 직접 호출하여 인덱싱된 임베딩을 가져옵니다.
        return self.item_matrix(item_ids)
    # 💡 [핵심] 밖에서 억지로 쑤셔넣지 않고, 클래스 스스로 추론 벡터를 받아 초기화하는 메서드
    def init_from_pretrained(self, pretrained_vecs):
        """추론된 사전학습 벡터를 미세조정 가능한 파라미터(Weight)로 초기화"""
        with torch.no_grad():
            self.item_matrix.weight.copy_(pretrained_vecs)
        print("✅ Pretrained item vectors successfully loaded into learnable matrix!")
    
def setup_models(cfg: PipelineConfig, device, item_state_dict=None, log_q_tensor=None):
    print(f"\n🧠 [Phase 4] Initializing Models...")
    
    # 1. User Tower 생성
    user_tower = SASRecUserTower_v3(cfg).to(device)
    
    # 2. Item Tower 뼈대 생성
    item_tower = SASRecItemTower(
        num_items=cfg.num_items, 
        d_model=cfg.d_model, 
        log_q_tensor=log_q_tensor
    ).to(device)
    
    # 3. Data 주입 (IO 데이터 -> Architecture)
    if item_state_dict is not None:
        try:
            # strict=False 옵션: 저장된 모델과 현재 구조의 키 이름이 조금 달라도 유연하게 로드
            missing, unexpected = item_tower.load_state_dict(item_state_dict, strict=False)
            print(f"✅ Item Tower weights successfully loaded!")
            if unexpected:
                print(f"   - Ignored extra keys from .pth: {unexpected[:3]}...")
            if missing:
                print(f"   ⚠️ [CRITICAL WARNING] Missing keys: {missing}")
        except Exception as e:
            print(f"❌ [Error] Weight injection failed: {e}")

    # 4. 학습 상태(Freeze/Unfreeze) 통제 적용
    item_tower.set_freeze_state(cfg.freeze_item_tower)
    
    # 직관적인 로깅
    mode_str = "FROZEN ❄️ (Speed Optimized)" if cfg.freeze_item_tower else "UNFROZEN 🔥 (Joint Fine-tuning)"
    print(f"✅ Item Tower State: {mode_str}")
    
    return user_tower, item_tower



# =====================================================================
# Phase 5: Training Loop 
# =====================================================================
def train_user_tower(epoch, model, item_tower, log_q_tensor, dataloader, optimizer, scaler, cfg, device, seq_labels, static_labels):
    """단일 에포크 훈련 함수 (실제 Loss 계산 및 로그 모니터링 적용)"""
    model.train()
    total_loss_accum = 0.0
    main_loss_accum = 0.0
    cl_loss_accum = 0.0
    
        
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # -------------------------------------------------------
        # 1. Data Unpacking (Dictionary to Device)
        # -------------------------------------------------------
        item_ids = batch['item_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        time_bucket_ids = batch['time_bucket_ids'].to(device)
        
        type_ids = batch['type_ids'].to(device)
        color_ids = batch['color_ids'].to(device)
        graphic_ids = batch['graphic_ids'].to(device)
        section_ids = batch['section_ids'].to(device)
        
        age_bucket = batch['age_bucket'].to(device)
        price_bucket = batch['price_bucket'].to(device)
        cnt_bucket = batch['cnt_bucket'].to(device)
        recency_bucket = batch['recency_bucket'].to(device)
        
        channel_ids = batch['channel_ids'].to(device)
        club_status_ids = batch['club_status_ids'].to(device)
        news_freq_ids = batch['news_freq_ids'].to(device)
        fn_ids = batch['fn_ids'].to(device)
        active_ids = batch['active_ids'].to(device)
        
        cont_feats = batch['cont_feats'].to(device)
        
        # Pretrained Vector 룩업 처리
        if 'pretrained_vecs' in batch:
            pretrained_vecs = batch['pretrained_vecs'].to(device)
        else:
            pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device)
            
        forward_kwargs = {
            'pretrained_vecs': pretrained_vecs,
            'item_ids': item_ids,
            'time_bucket_ids': time_bucket_ids,
            'type_ids': type_ids,
            'color_ids': color_ids,
            'graphic_ids': graphic_ids,
            'section_ids': section_ids,
            'age_bucket': age_bucket,
            'price_bucket': price_bucket,
            'cnt_bucket': cnt_bucket,
            'recency_bucket': recency_bucket,
            'channel_ids': channel_ids,
            'club_status_ids': club_status_ids,
            'news_freq_ids': news_freq_ids,
            'fn_ids': fn_ids,
            'active_ids': active_ids,
            'cont_feats': cont_feats,
            'padding_mask': padding_mask,
            'training_mode': True
        }

        # =======================================================
        # [모니터링 로그] 첫 배치에서만 데이터 상태 점검
        # =======================================================
        if batch_idx == 0:
            print(f"\n📦 [Batch 0 Monitor]")
            print(f"   - Item IDs: Shape {item_ids.shape} | Min {item_ids.min()} | Max {item_ids.max()}")
            print(f"   - Time Buckets: Min {time_bucket_ids.min()} | Max {time_bucket_ids.max()}")
            pad_ratio = (padding_mask.sum().item() / padding_mask.numel()) * 100
            print(f"   - Padding Ratio: {pad_ratio:.1f}%")
            print(f"   - Cont Feats Mean: {cont_feats.mean().item():.3f} | Std: {cont_feats.std().item():.3f}")
            
            print("\n🎯 [First User Data State Check]")
            print("-" * 50)
            print(f"👤 [User Profile]")
            print(f"   - Age Bucket ID:    {age_bucket[0].item()} (Target Age Group)")
            print(f"   - Price Bucket ID:  {price_bucket[0].item()} (Spending Power)")
            print(f"   - News Freq ID:     {news_freq_ids[0].item()} (Marketing Sensitivity)")
            
            valid_indices = torch.where(~padding_mask[0])[0]
            if len(valid_indices) > 0:
                print(f"\n🛍️ [Item History - Last 3 Items]")
                sample_indices = valid_indices[-3:] 
                sample_types = type_ids[0][sample_indices].tolist()
                sample_times = time_bucket_ids[0][sample_indices].tolist()
                for i, (t_id, time_id) in enumerate(zip(sample_types, sample_times)):
                    print(f"   - Item {i+1}: Type Hash ID [{t_id}] | Time Bucket ID [{time_id}]")
            else:
                print("\n⚠️ [Warning] This user has NO valid sequence (All Padded).")
            print("-" * 50)

        # -------------------------------------------------------
        # 2. Forward & Real Loss Calculation (AMP)
        # -------------------------------------------------------
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            
            
            
            # A. First View
            output_1 = model(**forward_kwargs)
            # B. Second View (Dropout 마스크가 달라짐)
            output_2 = model(**forward_kwargs)

            # (1) Main Loss (All Time Steps)
            #valid_mask = ~padding_mask.view(-1)
            #flat_output = output_1.view(-1, cfg.d_model)[valid_mask]
            #flat_targets = target_ids.view(-1)[valid_mask]
            
            last_output_1 = output_1[:, -1, :] # (Batch, Dim)
            last_targets = target_ids[:, -1]   # (Batch,)
            last_valid_mask = ~padding_mask[:, -1]
            
            valid_user_emb = last_output_1[last_valid_mask]
            valid_targets = last_targets[last_valid_mask]
            
            hnm_stats = {}
            
            if valid_user_emb.size(0) > 0:
                valid_user_emb = F.normalize(valid_user_emb, p=2, dim=1)
                
                # 💡 [핵심 추가] 평가때와 동일하게 item_tower에서 실시간으로 벡터 추출!
                # 나중에 Joint Training을 켤 때 아이템 벡터가 업데이트되려면 여기서 뽑아야 합니다.
                full_item_embeddings = item_tower.get_all_embeddings()
                norm_item_embeddings = F.normalize(full_item_embeddings, p=2, dim=1)
                main_loss, hnm_stats = full_batch_hard_emphasis_loss(
                    user_emb=valid_user_emb,
                    item_tower_emb=norm_item_embeddings, 
                    target_ids=valid_targets,
                    log_q_tensor=log_q_tensor,
                    top_k_percent=cfg.top_k_percent,
                    hard_margin=cfg.hard_margin,
                    hnm_threshold=cfg.hnm_threshold,   # Config에서 가져온 Threshold (예: 0.85)
                    temperature=0.15, 
                    lambda_logq=cfg.lambda_logq        # 상향된 1.0 적용
                )
            else:
                main_loss = torch.tensor(0.0, device=device)
                hnm_stats = {"avg_hn_similarity": 0.0, "num_active_hard_negs": 0}
            
            # (2) DuoRec Loss (Last Time Step Only)
            last_output_1 = output_1[:, -1, :] 
            last_output_2 = output_2[:, -1, :]
            last_targets = target_ids[:, -1]

            cl_loss = duorec_loss_refined(
                user_emb_1=last_output_1,
                user_emb_2=last_output_2,
                target_ids=last_targets,
                temperature = 0.07,
                lambda_sup=cfg.lambda_sup
            )

            # 최종 Loss 조합
            total_loss = main_loss + (cfg.lambda_cl * cl_loss)

        # -------------------------------------------------------
        # 3. Backward & Optimizer Step
        # -------------------------------------------------------
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        # 기울기 폭발 방지를 위한 정규화 (5.0은 트랜스포머에서 많이 쓰이는 여유있는 값)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        # 누적
        total_loss_accum += total_loss.item()
        main_loss_accum += main_loss.item()
        cl_loss_accum += cl_loss.item()
        
        pbar.set_postfix({
            'Loss': f"{total_loss.item():.4f}",
            'Main': f"{main_loss.item():.4f}",
            'CL': f"{cl_loss.item():.4f}"
        })
        
        # 100배치마다 로깅
        if batch_idx % 100 == 0:
            print(f"   [Epoch {epoch}] Batch {batch_idx:04d}/{len(dataloader)} | Total Loss: {total_loss.item():.4f} (Main: {main_loss.item():.4f}, CL: {cl_loss.item():.4f})")
        if batch_idx % 100 == 0:
            wandb.log({
                "Train/Main_Loss_Step": main_loss.item(),
                "HNM/Avg_Hard_Negative_Sim": hnm_stats.get("avg_hn_similarity", 0),
                "HNM/Num_K": hnm_stats.get("num_active_hard_negs", 0),
                "Step": epoch * len(dataloader) + batch_idx
            })
        # -------
        
    avg_loss = total_loss_accum / len(dataloader)
    avg_main = main_loss_accum / len(dataloader)
    avg_cl = cl_loss_accum / len(dataloader)

    with torch.no_grad():
        s_weights = torch.sigmoid(model.seq_gate).cpu().numpy()
        u_weights = torch.sigmoid(model.static_gate).cpu().numpy()
            
            # 딕셔너리 형태로 변환하여 WandB에 전송
    gate_log = {f"Gate/Seq_{label}": w for label, w in zip(seq_labels, s_weights)}
    gate_log.update({f"Gate/Static_{label}": w for label, w in zip(static_labels, u_weights)})
    wandb.log(gate_log)

    
    print(f"🏁 Epoch {epoch} Completed | Avg Total: {avg_loss:.4f} (Main: {avg_main:.4f}, CL: {avg_cl:.4f})")
    return avg_loss


import torch
import torch
import torch.nn.functional as F
from tqdm import tqdm



# 💡 인자에 processor를 추가했습니다.
def evaluate_model(model, item_tower, dataloader, target_df_path, device, processor, k_list=[10, 20, 200,500]):
    """
    Validation 데이터셋과 정답지(target_dict)를 이용해 Recall@K를 평가하는 함수
    """
    model.eval()
    item_tower.eval()
    print(f"🎯 Loading targets from: {target_df_path}")
    target_df = pd.read_parquet(target_df_path)
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    del target_df
    # K값 중 가장 큰 값을 기준으로 한 번만 Top-K 연산을 수행하여 GPU 연산 절약
    max_k = max(k_list)
    
    total_hits = {k: 0.0 for k in k_list}
    total_valid_users = 0
    
    with torch.no_grad():
        # 1. 전체 아이템 임베딩 로드 및 정규화 (루프 밖에서 한 번만 수행)
        full_item_embeddings = item_tower.get_all_embeddings()
        norm_item_embeddings = F.normalize(full_item_embeddings, p=2, dim=1)
        
        '''
        print("\n🔍 [Eval Monitor] Item Tower Check")
        print(f"   - Shape: {full_item_embeddings.shape}")
        print(f"   - Mean: {full_item_embeddings.mean().item():.6f} | Std: {full_item_embeddings.std().item():.6f}")
            # 인덱스 1번(첫 번째 실제 아이템)의 앞 5개 차원 값 출력
        if full_item_embeddings.size(0) > 1:
            print(f"   - Item [1] Sample: {full_item_embeddings[1][:5].tolist()}")
        '''
        
        
        
        
        # tqdm을 이용해 진행 시간 및 상태 표시
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Dataloader에서 'user_ids'를 문자열 리스트로 바로 가져옴
            user_ids = batch['user_ids'] 
            
            item_ids = batch['item_ids'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            time_bucket_ids = batch['time_bucket_ids'].to(device)
            type_ids = batch['type_ids'].to(device)
            color_ids = batch['color_ids'].to(device)
            graphic_ids = batch['graphic_ids'].to(device)
            section_ids = batch['section_ids'].to(device)
            age_bucket = batch['age_bucket'].to(device)
            price_bucket = batch['price_bucket'].to(device)
            cnt_bucket = batch['cnt_bucket'].to(device)
            recency_bucket = batch['recency_bucket'].to(device)
            channel_ids = batch['channel_ids'].to(device)
            club_status_ids = batch['club_status_ids'].to(device)
            news_freq_ids = batch['news_freq_ids'].to(device)
            fn_ids = batch['fn_ids'].to(device)
            active_ids = batch['active_ids'].to(device)
            cont_feats = batch['cont_feats'].to(device)        
            recency_offset = batch['recency_offset'].to(device)
            current_week = batch['current_week'].to(device)
            
            # Pretrained Vector 룩업 처리
            if 'pretrained_vecs' in batch:
                pretrained_vecs = batch['pretrained_vecs'].to(device)
                print("pretrained_vecs has been loaded")
            else:
                pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device)
            
            # =======================================================
            '''
            if batch_idx == 0:
                print(f"\n🔍 [Eval Monitor] Pretrained Vecs Check (Batch 0)")
                print(f"   - Shape: {pretrained_vecs.shape}")
                print(f"   - Mean: {pretrained_vecs.mean().item():.6f} | Std: {pretrained_vecs.std().item():.6f}")
                
                # 패딩(0)이 아닌 실제 아이템 ID 하나를 찾아 해당 벡터의 값 확인
                valid_mask = item_ids[0] != 0
                if valid_mask.any():
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0][0]
                    sample_item_id = item_ids[0][valid_idx].item()
                    print(f"   - Item [{sample_item_id}] Sample: {pretrained_vecs[0][valid_idx][:5].tolist()}")
            '''
            forward_kwargs = {
                'pretrained_vecs': pretrained_vecs,
                'item_ids': item_ids,
                'time_bucket_ids': time_bucket_ids,
                'type_ids': type_ids,
                'color_ids': color_ids,
                'graphic_ids': graphic_ids,
                'section_ids': section_ids,
                'age_bucket': age_bucket,
                'price_bucket': price_bucket,
                'cnt_bucket': cnt_bucket,
                'recency_bucket': recency_bucket,
                'channel_ids': channel_ids,
                'club_status_ids': club_status_ids,
                'news_freq_ids': news_freq_ids,
                'fn_ids': fn_ids,
                'active_ids': active_ids,
                'cont_feats': cont_feats,
                'recency_offset': recency_offset, 'current_week': current_week,
                'padding_mask': padding_mask,
                'training_mode': False # Dropout 비활성화
            }

            # 2. User Tower Forward
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output = model(**forward_kwargs) # (Batch, Seq_Len, Dim)
                
            # 3. 실제 마지막 유효 시점의 벡터 추출 (단순히 -1이 아니라 Padding을 고려)
            if output.dim() == 3:
                lengths = (~padding_mask).sum(dim=1)
                last_indices = (lengths - 1).clamp(min=0)
                batch_range = torch.arange(output.size(0), device=device)
                last_user_emb = output[batch_range, last_indices]
            else:
                last_user_emb = output
                
            # L2 정규화
            last_user_emb = F.normalize(last_user_emb, p=2, dim=1)
            
            # 4. 정답지(target_dict)에 존재하는 유효한 유저만 필터링
            valid_idx_list = [i for i, uid in enumerate(user_ids) if uid in target_dict and len(target_dict[uid]) > 0]
            if not valid_idx_list: 
                continue 
                
            v_idx = torch.tensor(valid_idx_list, device=device)
            valid_user_emb = last_user_emb[v_idx]
            
            # 5. 전체 아이템과 내적하여 Score 계산
            scores = torch.matmul(valid_user_emb, norm_item_embeddings.T)
            
            # 6. Top-K 인덱스 추출
            _, topk_indices = torch.topk(scores, k=max_k, dim=-1)
            pred_ids = topk_indices.cpu().numpy() 
            
            # 7. 실제 정답(Set)과 교집합 비교하여 Recall@K 측정
            for i, original_idx in enumerate(valid_idx_list):
                u_id = user_ids[original_idx]
                
                # 💡 [안전 장치] 정답이 단일 문자열이든 리스트든 무조건 리스트로 취급하게 만듦
                raw_targets = target_dict[u_id]
                if isinstance(raw_targets, str) or not hasattr(raw_targets, '__iter__'):
                    raw_targets = [raw_targets]
                
                # 💡 리스트로 만들어진 raw_targets를 순회
                actual_indices = set(processor.item2id[iid] for iid in raw_targets if iid in processor.item2id)
                
                # 만약 정답 아이템들이 모델이 모르는(OOT/Unseen) 아이템이라 매핑 후 세트가 비어있다면, 
                # 맞출 가능성이 0이므로 평가 타겟 유저에서 제외 (분모 증가 방지)
                if not actual_indices:
                    continue
                
                total_valid_users += 1
                for k in k_list:
                    # 1. Top-K 예측 리스트를 Set으로 변환
                    pred_set = set(pred_ids[i, :k])
                    
                    # 2. 교집합(맞춘 아이템들) 추출
                    hit_items = actual_indices.intersection(pred_set)
                    
                    # 3. (맞춘 개수 / 실제 구매한 전체 개수) 비율 계산
                    user_recall = len(hit_items) / len(actual_indices)
                    
                    # 4. 해당 비율을 누적
                    total_hits[k] += user_recall
            
            
            if 'scores' in locals():
                del scores, topk_indices
            
            # 2. Forward 연산 결과 텐서 해제
            del output, last_user_emb
            if 'valid_user_emb' in locals():
                del valid_user_emb
                
            # 3. 입력 피처 및 kwargs 일괄 해제
            del item_ids, padding_mask, time_bucket_ids, pretrained_vecs
            del type_ids, color_ids, graphic_ids, section_ids
            del age_bucket, price_bucket, cnt_bucket, recency_bucket
            del channel_ids, club_status_ids, news_freq_ids, fn_ids, active_ids, cont_feats
            del recency_offset, current_week
            del forward_kwargs

    # 최종 Recall 퍼센티지 계산
    results = {}
    if total_valid_users > 0:
        for k in k_list:
            results[f'Recall@{k}'] = (total_hits[k] / total_valid_users) * 100
            
    print(f"\n📈 [Validation Results] Valid Users: {total_valid_users}")
    for k in k_list:
        print(f"   - Recall@{k:03d}: {results.get(f'Recall@{k}', 0):.2f}%")
        
    del full_item_embeddings, norm_item_embeddings
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


from tqdm import tqdm
import wandb
from torch.nn.attention import SDPBackend, sdpa_kernel
def train_user_tower_all_time(epoch, model, item_tower, log_q_tensor, dataloader, optimizer, scaler, cfg, device, hard_neg_pool_tensor, scheduler, seq_labels=None, static_labels=None):
    """단일 에포크 훈련 함수 (All Time Steps + Same-User Masking 적용) + Gradient Accumulation"""
    model.train()
    total_loss_accum = 0.0
    main_loss_accum = 0.0
    cl_loss_accum = 0.0
    
    # 💡 [핵심 1] 누적 스텝 설정 (384 * 2 = 768)
    accumulation_steps = 1
    
    seq_labels = seq_labels or []
    static_labels = static_labels or []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    # 💡 [핵심 2] 루프 시작 전, 혹시 남아있을지 모르는 이전 에포크의 그래디언트 초기화
    #optimizer.zero_grad()
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(pbar):
        # ❌ optimizer.zero_grad() # 매 미니배치마다 초기화하면 안 되므로 삭제!

        # -------------------------------------------------------
        # 1. Data Unpacking (기존과 동일)
        # -------------------------------------------------------
        item_ids = batch['item_ids'].to(device, non_blocking=True)
        target_ids = batch['target_ids'].to(device, non_blocking=True)
        padding_mask = batch['padding_mask'].to(device, non_blocking=True)
        time_bucket_ids = batch['time_bucket_ids'].to(device, non_blocking=True)
        
        type_ids = batch['type_ids'].to(device, non_blocking=True)
        color_ids = batch['color_ids'].to(device, non_blocking=True)
        graphic_ids = batch['graphic_ids'].to(device, non_blocking=True)
        section_ids = batch['section_ids'].to(device, non_blocking=True)
        
        age_bucket = batch['age_bucket'].to(device, non_blocking=True)
        price_bucket = batch['price_bucket'].to(device, non_blocking=True)
        cnt_bucket = batch['cnt_bucket'].to(device, non_blocking=True)
        recency_bucket = batch['recency_bucket'].to(device, non_blocking=True)
        
        channel_ids = batch['channel_ids'].to(device, non_blocking=True)
        club_status_ids = batch['club_status_ids'].to(device, non_blocking=True)
        news_freq_ids = batch['news_freq_ids'].to(device, non_blocking=True)
        fn_ids = batch['fn_ids'].to(device, non_blocking=True)
        active_ids = batch['active_ids'].to(device, non_blocking=True)
        cont_feats = batch['cont_feats'].to(device, non_blocking=True)
        
        recency_offset = batch['recency_offset'].to(device, non_blocking=True)
        current_week = batch['current_week'].to(device, non_blocking=True)
        
        if 'pretrained_vecs' in batch:
            pretrained_vecs = batch['pretrained_vecs'].to(device, non_blocking=True)
        else:
            pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device, non_blocking=True)
            
        forward_kwargs = {
            'pretrained_vecs': pretrained_vecs,
            'item_ids': item_ids,
            'time_bucket_ids': time_bucket_ids,
            'type_ids': type_ids, 'color_ids': color_ids,
            'graphic_ids': graphic_ids, 'section_ids': section_ids,
            'age_bucket': age_bucket, 'price_bucket': price_bucket,
            'cnt_bucket': cnt_bucket, 'recency_bucket': recency_bucket,
            'channel_ids': channel_ids, 'club_status_ids': club_status_ids,
            'news_freq_ids': news_freq_ids, 'fn_ids': fn_ids,
            'active_ids': active_ids, 'cont_feats': cont_feats,
            'recency_offset': recency_offset, 'current_week': current_week,
            'padding_mask': padding_mask,
            'training_mode': True
        }
        # =======================================================
        # 🕵️‍♂️ [Data Peek] 첫 번째 에포크, 첫 번째 배치에서 데이터 캡처!
        # =======================================================
        if epoch == 1 and batch_idx == 0:
            print("\n" + "="*70)
            print("🕵️‍♂️ [Tensor Peek] First Batch, First User Verification")
            print("="*70)
            
            u_idx = 0 # 배치의 첫 번째 유저
            seq_len = item_ids.shape[1]
            valid_len = (~padding_mask[u_idx]).sum().item()
            
            print(f"✅ User Index in Batch: {u_idx} | Valid Length: {valid_len} / {seq_len}")
            
            # 1. 1명의 유저 시퀀스 데이터 확인 (패딩 포함 전체 리스트 출력)
            print("\n[1. Sequence Alignment Check (Left Padding Expected)]")
            print(f"📦 item_ids      : {item_ids[u_idx].tolist()}")
            print(f"🎯 target_ids    : {target_ids[u_idx].tolist()}")
            print(f"⏳ time_buckets  : {time_bucket_ids[u_idx].tolist()}")
            print(f"📆 recency_offset: {recency_offset[u_idx].tolist()}")
            print(f"💰 price_bucket  : {price_bucket[u_idx].tolist()}")
            
            # 2. forward_kwargs 형태(Shape) 확인
            print("\n[2. forward_kwargs Shape Check]")
            for k, v in forward_kwargs.items():
                if isinstance(v, torch.Tensor):
                    # 텐서인 경우 차원(Shape) 출력
                    print(f" - {k:<15}: {list(v.shape)}")
                else:
                    # 텐서가 아닌 경우 (예: training_mode) 값 출력
                    print(f" - {k:<15}: {v}")
            print("="*70 + "\n")
        # -------------------------------------------------------
        # 2. Forward & Loss Calculation (AMP)
        # -------------------------------------------------------
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            '''
            allowed_backends = [
                SDPBackend.FLASH_ATTENTION, 
                SDPBackend.EFFICIENT_ATTENTION
            ]
    
            with sdpa_kernel(allowed_backends):
            '''
            output_1 = model(**forward_kwargs)
            # output_2 = model(**forward_kwargs) # (DuoRec 제거되었으므로 주석 처리 권장)
            
            valid_mask = ~padding_mask 
            batch_size, seq_len = item_ids.shape
            
            seq_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            last_indices = torch.max(seq_positions.masked_fill(~valid_mask, -1), dim=1)[0]
            last_indices = last_indices.clamp(min=0) 
            
            batch_range = torch.arange(batch_size, device=device)
            is_last_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            is_last_mask[batch_range, last_indices] = True
            
            flat_output = output_1[valid_mask] 
            flat_targets = target_ids[valid_mask]
            flat_is_last = is_last_mask[valid_mask] 
            
            batch_row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
            flat_user_ids = batch_row_indices[valid_mask] 
            
            if flat_output.size(0) > 0:
                flat_user_emb = F.normalize(flat_output, p=2, dim=1)
                last_hard_neg_ids = None 
                
                all_item_emb = item_tower.get_all_embeddings()
                norm_item_embeddings = F.normalize(all_item_emb, p=2, dim=1)
                
                main_loss, b_metrics = inbatch_corrected_logq_loss_with_hard_neg_margin(
                    user_emb=flat_user_emb, item_tower_emb=norm_item_embeddings,
                    target_ids=flat_targets, user_ids=flat_user_ids,
                    log_q_tensor=log_q_tensor, last_hard_neg_ids=last_hard_neg_ids, 
                    flat_is_last=flat_is_last, temperature=0.1, 
                    lambda_logq=cfg.lambda_logq, margin=0.00, alpha=1, return_metrics=True 
                )
            else:
                main_loss = torch.tensor(0.0, device=device)
            
            total_loss = main_loss 

            # 💡 [핵심 3] 그래디언트 누적을 위한 Loss 스케일링
            # 두 번 누적할 것이므로, 1번 더해질 때마다 절반의 크기로 더해져야 
            # 배치 768을 한 번에 처리한 것과 수학적으로 동일한 스케일이 됩니다.
            scaled_loss = total_loss / accumulation_steps

        # -------------------------------------------------------
        # 3. Backward (Loss 누적)
        # -------------------------------------------------------
        # 주의: scaled_loss로 backward를 수행해야 그래디언트 크기가 유지됩니다.
        scaler.scale(scaled_loss).backward()

        # -------------------------------------------------------
        # 4. Optimizer Step & Scheduler (누적 주기에 도달했을 때만)
        # -------------------------------------------------------
        # 마지막 남은 자투리 배치(len(dataloader)와 같을 때)도 잊지 않고 업데이트해야 함
        if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 여기서 그래디언트를 비워줍니다.
            #optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            # 스케줄러도 "유효 배치(Effective Batch)"가 끝났을 때만 스텝을 밟습니다.
            if scheduler is not None:
                scheduler.step()

        # -------------------------------------------------------
        # 5. Logging (주의: 모니터링은 무조건 Unscaled Loss로!)
        # -------------------------------------------------------
        # 누적용 변수나 화면에 보여줄 때는 쪼개지 않은 원래 total_loss를 써야 착시가 없습니다.
        total_loss_accum += total_loss.item()
        main_loss_accum += main_loss.item()
        
        pbar.set_postfix({
            'Loss': f"{total_loss.item():.4f}",
            'Main': f"{main_loss.item():.4f}"
        })
        
        # 100번 단위 로깅도 미니배치(384) 기준 100번이 됩니다.
        if batch_idx % 100 == 0:
            wandb.log({
                "Train/Main_Loss": main_loss.item(), # Unscaled Loss
            })

        # mini-batch deloc
        del output_1, flat_output, flat_targets, flat_is_last, flat_user_ids
        del main_loss, total_loss, scaled_loss
        
        # 2. 부피가 큰 입력 피처 텐서 일괄 해제
        del item_ids, target_ids, padding_mask, time_bucket_ids, pretrained_vecs
        del type_ids, color_ids, graphic_ids, section_ids
        del age_bucket, price_bucket, cnt_bucket, recency_bucket
        del channel_ids, club_status_ids, news_freq_ids, fn_ids, active_ids, cont_feats
        del recency_offset, current_week
        if 'flat_user_emb' in locals():
            del flat_user_emb, all_item_emb, norm_item_embeddings
        
    # 에포크 종료 후 평균 Loss 계산
    avg_loss = total_loss_accum / len(dataloader)
    avg_main = main_loss_accum / len(dataloader)
    avg_cl = cl_loss_accum / len(dataloader) if cl_loss_accum > 0 else 0.0

    # Gate Weights Logging
    with torch.no_grad():
        s_weights = torch.sigmoid(model.seq_gate).cpu().numpy()
        u_weights = torch.sigmoid(model.static_gate).cpu().numpy()
        
        gate_log = {}
        if seq_labels and len(seq_labels) == len(s_weights):
            gate_log.update({f"Gate/Seq_{label}": w for label, w in zip(seq_labels, s_weights)})
        if static_labels and len(static_labels) == len(u_weights):
            gate_log.update({f"Gate/Static_{label}": w for label, w in zip(static_labels, u_weights)})
            
        if gate_log:
            wandb.log(gate_log)

    print(f"🏁 Epoch {epoch} Completed | Avg Total: {avg_loss:.4f} (Main: {avg_main:.4f}, CL: {avg_cl:.4f})")
    return avg_loss
# =====================================================================
# Main Execution Pipeline
# =====================================================================

def run_resume_pipeline_v2():
    """기존 체크포인트에서 이어서 학습(Fine-tuning)하는 파이프라인"""
    print("🚀 Starting User Tower RESUME Pipeline...")
    

    SEQ_LABELS = ['item_id', 'time', 'type', 'color', 'graphic', 'section', 'recency_curr', 'week_curr']
    STATIC_LABELS = [
        'age', 'price', 'cnt', 'recency',      # Buckets (4)
        'channel', 'club', 'news', 'fn', 'active', # Categoricals (5)
        'cont_price_std', 'cont_last_diff', 'cont_repurch', 'cont_weekend' # Continuous (4) [추가]
    ] 
    # 1. Config & Env
    cfg = PipelineConfig()
    device = setup_environment()
    processor, val_processor, cfg = prepare_features(cfg)
    
    # 1-1. h-params 
    cfg.lr = 1e-3
    item_finetune_lr = cfg.lr * 0.05
    
    # item metadata cfg
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")

    
    # 2. Data 가져오기
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    
    item_state_dict = load_item_tower_state_dict(cfg.model_dir, cfg.item_tower_pth_name, device)
    log_q_tensor = processor.get_logq_probs(device)
    
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    
    train_loader = create_dataloaders(processor, cfg, "2020-09-16", aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, "2020-09-22",aligned_vecs, is_train=False)
    dataset_peek(train_loader.dataset, processor)
    
    
    
        
    wandb.init(
        project="SASRec-User-Tower-causality-Optimization-v2", # 프로젝트명
        name=f"run_lr_{cfg.lr}_epoch_{cfg.epochs}_time_sperad_f1", # 실험 이름
        config=cfg.__dict__ # 하이퍼파라미터 저장
    )
    
    
    # -----------------------------------------------------------
    # 3. Models & Optimizer Setup (체크포인트 로드)
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, item_state_dict, log_q_tensor)
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")
    
    # 💡 이전 Best 가중치 로드
    user_checkpoint_path = os.path.join(cfg.model_dir, "best_user_tower_v2_time.pth")
    item_checkpoint_path = os.path.join(cfg.model_dir, "best_item_tower_v2_time.pth")
    
    user_tower.load_state_dict(torch.load(user_checkpoint_path, map_location=device))
    item_tower.load_state_dict(torch.load(item_checkpoint_path, map_location=device))
    print(f"✅ Successfully loaded best weights from disk for resuming!")
    
    # 💡 이미 학습이 진행된 상태이므로, 아이템 타워 동결 없이 처음부터 Joint Training
    item_tower.set_freeze_state(False)
    print(f"🔥 Resuming Joint Training! (User LR: {cfg.lr}, Item LR: {item_finetune_lr})")
    
    # User와 Item 파라미터를 처음부터 함께 묶어 옵티마이저 생성
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': cfg.lr},
        {'params': item_tower.parameters(), 'lr': item_finetune_lr}
    ], weight_decay=cfg.weight_decay)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # 💡 [스케줄러 변경] threshold 0.001 추가 및 patience 약간 여유 있게(3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, threshold=0.001
    )
    
    # 💡 이전 최고 기록(20.08)을 수동으로 입력해주면 기존 기록부터 이어갈 수 있지만,
    # 새로운 체크포인트를 만들기 위해 0.0부터 시작합니다. (필요 시 20.08로 변경 가능)
    best_recall_100 = 0.0 

    # -----------------------------------------------------------
    # 4. Training Loop
    # -----------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        
        # ------------------- 훈련 (Train) -------------------
        avg_loss = train_user_tower_all_time(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower, 
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=None, # 하드 네거티브가 필요하다면 이 부분을 수정
            scheduler=None,
            seq_labels=SEQ_LABELS,
            static_labels=STATIC_LABELS,
        )
        
        # ------------------- 평가 (Evaluate) -------------------
        val_metrics = evaluate_model(
            model=user_tower, 
            item_tower=item_tower, 
            dataloader=val_loader,
            target_df_path=TARGET_VAL_PATH,
            device=device,
            processor=processor,
            k_list=[20, 100, 500]
        )
        
        current_recall_100 = val_metrics.get('Recall@100', 0.0)
        
        # ------------------- 스케줄러 & Best Model 저장 -------------------
        scheduler.step(current_recall_100)
        
        if current_recall_100 > best_recall_100:
            print(f"🌟 [New Best!] Recall@100 updated: {best_recall_100:.2f}% -> {current_recall_100:.2f}%")
            best_recall_100 = current_recall_100
            
            # 💡 [수정] 새로운 이름으로 체크포인트 저장
            torch.save(user_tower.state_dict(), os.path.join(cfg.ft_model_dir, "best_user_tower_v2_time_ft.pth"))
            torch.save(item_tower.state_dict(), os.path.join(cfg.ft_model_dir, "best_item_tower_v2_time_ft.pth"))
            print("   💾 Best model weights saved to disk (v1/v2).")
        else:
            print(f"   - (Current Best: {best_recall_100:.2f}%)")
            
    print("\n🎉 Resume Pipeline Execution Finished Successfully!")



def run_pipeline_opt_v2():
    """Airflow DAG나 MLflow Run에서 직접 호출하는 엔트리 포인트"""
    print("🚀 Starting User Tower Training Pipeline...")
    
    SEQ_LABELS = ['item_id', #'time',
                  'type', #'color', 'graphic', 'section', 
                  'recency_curr', 'week_curr']
    STATIC_LABELS = [
        'age', 'price', #'cnt', 'recency',      # Buckets (4)
        'channel', 'club', 'news', 'fn', 'active', # Categoricals (5)
        'cont_price_std', 'cont_last_diff', 'cont_repurch', 'cont_weekend' # Continuous (4) [추가]
    ] 
    
    # 1. Config & Env
    cfg = PipelineConfig()
    device = setup_environment()
    processor, val_processor, cfg = prepare_features(cfg)
    
    # 1-1. h-params 세팅 (Epoch 대폭 상향)
    cfg.lr = 2e-3
    cfg.epochs = 40  # 💡 [핵심] 충분한 탐색을 위해 에포크를 50으로 상향 (실제론 Early Stop 됨)
    
    # item metadata cfg
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")

    # 2. Data 가져오기
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    
    item_state_dict = load_item_tower_state_dict(cfg.model_dir, cfg.item_tower_pth_name, device)
    log_q_tensor = processor.get_logq_probs(device)
    
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    val_processor.i_side_arr = processor.i_side_arr
    
    train_loader = create_dataloaders(processor, cfg, "2020-09-16", aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, "2020-09-22", aligned_vecs, is_train=False)
    dataset_peek_v3(train_loader.dataset, processor)
    
    print(f"✅ Final Model Config - Total Hash Nodes: {cfg.num_prod_types + cfg.num_colors + cfg.num_graphics + cfg.num_sections}")

    wandb.init(
        project="SASRec-User-Tower-causality-Optimization-v2",
        name=f"run_lr_{cfg.lr}_epoch_{cfg.epochs}_shuffle_session_correction", 
        config=cfg.__dict__ 
    )
    
    # -----------------------------------------------------------
    # 3. Models & Optimizer Setup (초기 상태: Epoch 1용 세팅)
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, item_state_dict, log_q_tensor)
    item_tower.init_from_pretrained(aligned_vecs.to(device))
    item_tower.set_freeze_state(True)
    print(f"❄️ Epoch 1: Item Tower FROZEN! (User Tower LR: {cfg.lr})")
    optimizer = torch.optim.AdamW(
        user_tower.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay,
        fused=True  # 💡 다중 CUDA 커널을 하나로 융합하여 속도 극대화
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # 💡 [신규 스케줄러 & 얼리스탑 세팅] 
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * 0.05) # 전체 훈련의 5% 구간 웜업 (50 에포크 기준)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01)
    
    # Patience 7 부여 (7 에포크 동안 안 오르면 종료)
    early_stopping = EarlyStopping(patience=5, mode='max')

    # -----------------------------------------------------------
    # 4. Training Loop
    # -----------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        # 💡 [Dynamic Unfreeze 시점 변경] 에포크가 늘어났으므로 5 에포크에 파인튜닝 시작
        if epoch == 5: 
            print("\n🔥 [Dynamic Unfreeze] Epoch 4: Item Tower Joint Training 시작!")
            item_tower.set_freeze_state(False)
            item_finetune_lr = cfg.lr * 0.05 
            
            optimizer.add_param_group({
                'params': item_tower.parameters(), 
                'lr': item_finetune_lr 
            })
            print(f"   - User Tower LR: {cfg.lr}")
            print(f"   - Item Tower LR: {item_finetune_lr} (Fine-tuning mode)")

        # ------------------- 훈련 (Train) -------------------
        avg_loss = train_user_tower_all_time(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower, 
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=None,
            scheduler=scheduler, # 배치 단위 스텝을 위해 전달
            seq_labels = SEQ_LABELS,
            static_labels = STATIC_LABELS,
        )
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # ------------------- 평가 (Evaluate) -------------------
        val_metrics = evaluate_model(
            model=user_tower, 
            item_tower=item_tower, 
            dataloader=val_loader,
            target_df_path=TARGET_VAL_PATH,
            device=device,
            processor=processor,
            k_list=[10, 20, 100, 500]
        )
        
        current_recall_100 = val_metrics.get('Recall@100', 0.0)
        
        # ------------------- 스케줄러 & Best Model 저장 -------------------
        early_stopping(current_recall_100)
        
        if early_stopping.is_best:
            print(f"🌟 [New Best!] Recall@100 updated: {current_recall_100:.2f}%")
            
            # 최고 성능 달성 시 파라미터 덮어쓰기 저장
            torch.save(user_tower.state_dict(), os.path.join(cfg.model_dir, "best_user_tower_v3_allseq_session_v2_test.pth"))
            torch.save(item_tower.state_dict(), os.path.join(cfg.model_dir, "best_item_tower_v3_allseq_session_v2.test_pth"))
            print("   💾 Best model weights saved to disk.")
            
        if early_stopping.early_stop:
            print(f"\n🛑 조기 종료 발동! {early_stopping.patience} Epoch 동안 Recall@100이 개선되지 않았습니다.")
            print(f"🏆 최종 최고 Recall@100: {early_stopping.best_score:.2f}%")
            break
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
            
    print("\n🎉 Pipeline Execution Finished Successfully!")

if __name__ == "__main__":
    # 5에포크까지 학습했으므로 6번부터 재개
    #run_resume_pipeline(resume_epoch=16, last_best_recall=22.60)
    run_pipeline_opt_v2()
    #run_resume_pipeline_v2()
