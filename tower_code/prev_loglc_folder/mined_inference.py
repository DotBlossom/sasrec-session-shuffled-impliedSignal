import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
import pandas as pd
import numpy as np
import os
import random
import math
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import gc
import warnings
import logging

warnings.filterwarnings("ignore", message="Support for mismatched src_key_padding_mask and mask is deprecated")

# ==========================================
# ⚙️ 설정 & 경로
# ==========================================
#TEMPERATURE = 0.2
LAMBDA_LOGQ = 0.1
BATCH_SIZE = 896
EMBED_DIM = 128
MAX_SEQ_LEN = 50
DROPOUT = 0.3
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = r"D:\trainDataset\localprops"
MODEL_DIR = r"C:\Users\candyform\Desktop\inferenceCode\models"
CACHE_DIR = os.path.join(BASE_DIR, "cache")

ITEM_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_item.parquet")
USER_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_user.parquet")
SEQ_DATA_PATH_PQ = os.path.join(BASE_DIR, "features_sequence_cleaned.parquet")
TARGET_VAL_PATH = os.path.join(BASE_DIR, "features_target_val.parquet")
USER_VAL_FEAT_PATH = os.path.join(BASE_DIR, "features_user_val.parquet")
SEQ_VAL_DATA_PATH = os.path.join(BASE_DIR, "features_sequence_val.parquet")

SAVE_PATH_BEST = os.path.join(MODEL_DIR, "user_tower_phase3_best_ft_0.19x.pth")

class SmartLogger:
    def __init__(self, verbosity=1): self.verbosity = verbosity
    def log(self, level, msg):
        if self.verbosity >= level: print(f"[{'ℹ️' if level==1 else '📊'}] {msg}")

logger = SmartLogger(verbosity=1)

# ==========================================
# 1. Feature Processor & Dataset
# ==========================================
class FeatureProcessor:
    def __init__(self, user_path, item_path, seq_path, scaler=None):
        print(f"🔄 Loading Data from {user_path}...")
        self.users = pd.read_parquet(user_path)
        self.users = self.users.drop_duplicates(subset=['customer_id']).set_index('customer_id')
        
        self.items = pd.read_parquet(item_path).set_index('article_id')
        self.seqs = pd.read_parquet(seq_path).set_index('customer_id')
        
        # 인덱스 강제 문자열 변환
        self.users.index = self.users.index.astype(str)
        self.items.index = self.items.index.astype(str)
        self.seqs.index = self.seqs.index.astype(str)

        self.user_ids = self.users.index.tolist()
        self.user2id = {uid: i + 1 for i, uid in enumerate(self.user_ids)} # 1-based
        self.item_ids = self.items.index.tolist()
        self.item2id = {iid: i + 1 for i, iid in enumerate(self.item_ids)} # 1-based
        
        self.u_dense_cols = ['user_avg_price_log', 'total_cnt_log', 'recency_log']
        self.users_scaled = self.users.copy()
        
        if scaler is None: 
            self.user_scaler = StandardScaler()
            scaled_data = self.user_scaler.fit_transform(self.users[self.u_dense_cols])
        else: 
            self.user_scaler = scaler
            scaled_data = self.user_scaler.transform(self.users[self.u_dense_cols])
        
        self.users_scaled[self.u_dense_cols] = np.nan_to_num(scaled_data, nan=0.0)

    def get_user_tensor(self, user_id):
        if user_id not in self.users_scaled.index:
            return torch.zeros(len(self.u_dense_cols)), torch.tensor(0, dtype=torch.long)
            
        row = self.users_scaled.loc[user_id]
        dense = torch.tensor(row[self.u_dense_cols].values.astype(np.float32), dtype=torch.float32)
        # preferred_channel이 1~N이라고 가정하고 0-based로 변환 (-1)
        cat = torch.tensor(int(row['preferred_channel']) - 1, dtype=torch.long)
        return dense, cat
    def get_logq_probs(self, device):
        """
        모델의 Embedding(N+1, D) 구조와 일치하도록 인덱스 보정된 log_q 생성
        """
        # 1. raw_probability 추출 (0-based)
        raw_probs = self.items['raw_probability'].reindex(self.item_ids).values
        
        # 2. Smoothing 및 처리
        eps = 1e-6
        sorted_probs = np.nan_to_num(raw_probs, nan=0.0) + eps
        sorted_probs /= sorted_probs.sum()
        
        # 3. 로그 계산
        log_q_values = np.log(sorted_probs).astype(np.float32)
        
        # 4. [중요] 1-based 인덱싱 대응을 위한 Padding 추가
        # 0번 인덱스는 사용하지 않으므로 아주 작은 확률(또는 0)의 로그값으로 채움
        full_log_q = np.zeros(len(self.item_ids) + 1, dtype=np.float32)
        full_log_q[1:] = log_q_values  # 1번 인덱스부터 실제 값 채우기
        full_log_q[0] = -20.0          # 0번 인덱스(Padding)는 낮은 값으로 설정
    
        return torch.tensor(full_log_q, dtype=torch.float32).to(device)
class UserTowerDataset(Dataset):
    def __init__(self, processor, max_seq_len=50, is_training=True):
        self.processor = processor
        self.user_ids = processor.user_ids 
        self.max_len = max_seq_len
        self.is_training = is_training
        self.min_cut_len = 3      

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        u_id_str = self.user_ids[idx]
        u_dense, u_cat = self.processor.get_user_tensor(u_id_str)
        
        processed_tokens = []
        processed_deltas = []
        
        if u_id_str in self.processor.seqs.index:
            seq_row = self.processor.seqs.loc[u_id_str]
            # Series일 경우 처리
            if isinstance(seq_row, pd.DataFrame): seq_row = seq_row.iloc[0]
                
            for i, d in zip(seq_row['sequence_ids'], seq_row['sequence_deltas']):
                 token = self.processor.item2id.get(str(i), 0) # str 변환 안전장치
                 if token == 0: continue
                 processed_tokens.append(token)
                 processed_deltas.append(d)

        seq_len = len(processed_tokens)
        input_seq = []
        target_seq = [] 

        if seq_len > 0:
            if self.is_training:
                can_sample = seq_len > self.min_cut_len
                if not can_sample or random.random() < 0.8:
                    input_seq = processed_tokens[:-1]
                    target_seq = processed_tokens[1:]
                else:
                    max_cut = seq_len - 1
                    cut_idx = seq_len if max_cut < self.min_cut_len else random.randint(self.min_cut_len, max_cut)
                    full_slice = processed_tokens[:cut_idx+1]
                    input_seq = full_slice[:-1]
                    target_seq = full_slice[1:]
            else:
                input_seq = processed_tokens[:]
                target_seq = [0] * len(input_seq)

        input_ids = input_seq[-self.max_len:]
        target_ids = target_seq[-self.max_len:]
        input_deltas = processed_deltas[:len(input_seq)][-self.max_len:]

        return {
            'user_idx': torch.tensor(idx + 1, dtype=torch.long),
            'user_dense': u_dense, 'user_cat': u_cat,
            'seq_ids': torch.tensor(input_ids, dtype=torch.long),
            'seq_deltas': torch.tensor(input_deltas, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

def user_tower_collate_fn(batch):
    u_idx = torch.stack([b['user_idx'] for b in batch])
    u_dense = torch.stack([b['user_dense'] for b in batch])
    u_cat = torch.stack([b['user_cat'] for b in batch])
    seq_ids = pad_sequence([b['seq_ids'] for b in batch], batch_first=True, padding_value=0)
    seq_deltas = pad_sequence([b['seq_deltas'] for b in batch], batch_first=True, padding_value=0)
    target_ids = pad_sequence([b['target_ids'] for b in batch], batch_first=True, padding_value=0)
    seq_mask = (seq_ids != 0).long()
    last_target = torch.tensor([b['target_ids'][-1] if len(b['target_ids']) > 0 else 0 for b in batch], dtype=torch.long)
    return u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, target_ids, last_target

# ==========================================
# 2. Alignment Functions (Alignment)
# ==========================================
def load_and_align_embeddings(model, processor, model_dir, device):
    """ Content Item Embedding Alignment (Pretrained -> model.item_content_emb) """
    print(f"\n🔄 [Content Alignment] Starting Item Embedding Alignment...")
    emb_path = os.path.join(model_dir, "pretrained_item_matrix.pt")
    ids_path = os.path.join(model_dir, "item_ids.pt")

    try:
        pretrained_emb = torch.load(emb_path, map_location='cpu')
        if isinstance(pretrained_emb, dict):
            pretrained_emb = pretrained_emb.get('weight', pretrained_emb.get('item_content_emb.weight'))
        pretrained_ids = torch.load(ids_path, map_location='cpu')
    except Exception as e:
        print(f"❌ [Error] Failed to load Content files: {e}")
        return model

    pretrained_map = {str(item_id.item()) if isinstance(item_id, torch.Tensor) else str(item_id): pretrained_emb[idx] for idx, item_id in enumerate(pretrained_ids)}
    
    num_embeddings = len(processor.item_ids) + 1 
    new_weight = torch.randn(num_embeddings, pretrained_emb.shape[1]) * 0.01 
    new_weight[0] = 0.0 
    
    matched = 0
    for i, current_id_str in enumerate(processor.item_ids):
        if current_id_str in pretrained_map:
            new_weight[i + 1] = pretrained_map[current_id_str]
            matched += 1
            
    with torch.no_grad():
        model.item_content_emb = nn.Embedding.from_pretrained(new_weight.to(device), freeze=False)
        
    print(f"✅ [Content Alignment] Matched: {matched}/{len(processor.item_ids)}")
    return model



def load_and_align_gnn_items2(model, processor, base_dir, device):
    """
    [Fixed] GNN 학습 결과(simgcl_trained.pth) - ID Mapping Only
    """
    print(f"\n🔄 [GNN Alignment] Starting GNN Item Embedding Alignment (ID Only)...")
    
    # ... (경로 설정 및 파일 로드 부분 동일) ...
    cache_dir = os.path.join(base_dir, "cache")
    model_path = os.path.join(MODEL_DIR , "simgcl_trained.pth")
    maps_path = os.path.join(cache_dir, "id_maps_train.pt")

    try:
        maps = torch.load(maps_path, map_location='cpu')
        gnn_item2id = maps['item2id']
        
        gnn_state_dict = torch.load(model_path, map_location='cpu')
        gnn_emb_weight = gnn_state_dict['embedding_item.weight']
        
    except Exception as e:
        print(f"❌ [Error] Failed to load GNN files: {e}")
        return model

    # ... (매트릭스 생성 및 매핑 부분 동일) ...
    num_embeddings = len(processor.item_ids) + 1 
    emb_dim = gnn_emb_weight.shape[1]
    new_weight = torch.randn(num_embeddings, emb_dim) * 0.01
    new_weight[0] = 0.0

    matched_count = 0
    for i, current_id_str in enumerate(processor.item_ids):
        target_idx = i + 1 
        if current_id_str in gnn_item2id:
            gnn_idx = gnn_item2id[current_id_str]
            new_weight[target_idx] = gnn_emb_weight[gnn_idx]
            matched_count += 1
            
    # 5. 모델 주입 (수정됨!)
    target_layer_name = 'gnn_item_emb'  # ✅ CORRECT: User -> Item으로 변경
    
    with torch.no_grad():
        if hasattr(model, target_layer_name):
            setattr(model, target_layer_name, nn.Embedding.from_pretrained(new_weight.to(device), freeze=False))
            print(f"  ✅ Injected aligned vectors into 'model.{target_layer_name}'")
        else:
            print(f"❌ [Critical] Could not find '{target_layer_name}' in User Tower.")
            return model

    print(f"✅ [GNN Alignment] Complete! Matched: {matched_count}/{len(processor.item_ids)}")
    return model





import torch
import torch.nn as nn
import os

class ResidualAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        
        # 1. [Main Path] 학습을 통해 '변형'될 특징 (Interaction 정보 반영)
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        )
        
        # 2. [Shortcut Path] 원본 메타데이터의 특성을 유지하는 경로
        self.shortcut = nn.Linear(input_dim, output_dim, bias=False)
        
        # 3. [Gate Layer] (NEW)
        # 입력(x)을 보고 0~1 사이의 중요도(alpha)를 산출
        # input_dim -> output_dim 크기의 게이트를 만들어 차원별로 조절 가능하게 함
        self.gate_layer = nn.Linear(input_dim, output_dim)
        
        # --- Initialization ---
        # A. Shortcut: Identity에 가깝게 초기화 (원본 보존)
        if input_dim == output_dim:
            nn.init.eye_(self.shortcut.weight)
        else:
            nn.init.xavier_uniform_(self.shortcut.weight)

        # B. MLP: 초기 출력을 작게 하여 학습 초기 충격 방지
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

        # C. Gate: 초기에는 "원본(Shortcut)"을 더 신뢰하도록 설정
        # Bias를 양수(2.0)로 설정하면 Sigmoid 통과 후 약 0.88이 됨
        # 즉, 학습 초기에는 원본 88%, 변형 12% 비율로 시작
        nn.init.xavier_uniform_(self.gate_layer.weight, gain=0.01)
        nn.init.constant_(self.gate_layer.bias, 2.0) 

    def forward(self, x):
        # 1. 변형된 특징 (Learned Context)
        transformed = self.mlp(x)
        
        # 2. 원본 보존 특징 (Original Metadata)
        original = self.shortcut(x)
        
        # 3. Gate 계수 계산 (0.0 ~ 1.0)
        # alpha가 높을수록 '원본 메타데이터'를 유지하려는 성향이 강함
        alpha = torch.sigmoid(self.gate_layer(x))
        
        # 4. Gated Mixing (Convex Combination)
        # "메타데이터가 확실하면 원본을 쓰고, 구매 패턴이 특이하면 변형된 값을 써라"
        return alpha * original + (1 - alpha) * transformed

def load_and_align_gnn_items(model, processor, base_dir, device):
    """ 
    GNN Item Embedding Alignment (Residual Adapter Aware)
    저장된 Adapter 가중치를 사용해 최종 임베딩을 계산한 뒤 정렬하여 주입 
    """
    print(f"\n🔄 [GNN Item Alignment] Starting (Adapter Mode)...")
    
    # 경로 설정
    # MODEL_DIR 전역 변수가 없다면 base_dir 기준으로 설정하거나 인자로 받으세요.
    # 여기서는 안전하게 base_dir/models 혹은 직접 지정된 경로 사용 가정
    model_dir = globals().get('MODEL_DIR', os.path.join(base_dir, 'models')) 
    cache_dir = os.path.join(base_dir, "cache")
    
    model_path = os.path.join(model_dir, "model_ver2_ep2_tune.pth") # 혹은 최신 에포크
    maps_path = os.path.join(cache_dir, "id_maps_train.pt")

    
        # 1. ID Map 로드
    maps = torch.load(maps_path, map_location='cpu')
    gnn_item2id = maps['item2id']
        
        # 2. State Dict 로드
    state_dict = torch.load(model_path, map_location='cpu')
        
        # 3. [핵심] Adapter 가중치와 Pretrained Feature 복원
        # 저장된 모델에 item_features가 없다면, GNN 학습 시 사용한 원본 feature 파일을 로드해야 함
        # 하지만 GNNTrainer에서 'item_features'를 buffer나 parameter로 저장했다면 state_dict에 있음
        
        # (A) Pretrained Input Feature 찾기
    if 'item_features' in state_dict:
        raw_features = state_dict['item_features'] # (N_gnn, 128)
    else:
            # 만약 state_dict에 없다면 외부 파일에서 로드 (예외 처리)
            # 여기서는 편의상 state_dict에 있다고 가정하거나, 없으면 에러
        raise ValueError("❌ 'item_features' not found in checkpoint! (Did you save it?)")

        # (B) Adapter & Bias 로드 및 계산
    input_dim = raw_features.shape[1] # 128
    output_dim = 64 # GNN output dim
        
    adapter = ResidualAdapter(input_dim, output_dim).to('cpu')
        
        # Adapter 키 매핑 (접두사 'item_adapter.' 제거 필요할 수 있음)
    adapter_state = {}
    for k, v in state_dict.items():
        if k.startswith('item_adapter.'):
            adapter_state[k.replace('item_adapter.', '')] = v
        
    adapter.load_state_dict(adapter_state)
        
        # Bias 로드
    bias = state_dict.get('item_bias', torch.zeros(len(gnn_item2id), output_dim))
        
        # (C) 최종 임베딩 생성 (Forward)
    with torch.no_grad():
        adapter.eval()
            # Feature -> Adapter -> + Bias
        final_gnn_embeddings = adapter(raw_features) + bias
        

    # 4. 정렬 및 주입 (기존 로직과 동일)
    num_embeddings = len(processor.item_ids) + 1 
    new_weight = torch.randn(num_embeddings, output_dim) * 0.01
    new_weight[0] = 0.0

    matched = 0
    for i, current_id_str in enumerate(processor.item_ids):
        if current_id_str in gnn_item2id:
            # GNN 학습 시의 ID로 임베딩 조회
            gnn_idx = gnn_item2id[current_id_str]
            if gnn_idx < len(final_gnn_embeddings):
                new_weight[i + 1] = final_gnn_embeddings[gnn_idx]
                matched += 1
            
    with torch.no_grad():
        model.gnn_item_emb = nn.Embedding.from_pretrained(new_weight.to(device), freeze=False)
        print(f"✅ Injected into 'model.gnn_item_emb' (Dim: {output_dim})")

    print(f"✅ [GNN Item Alignment] Matched: {matched}/{len(processor.item_ids)}")
    return model

def load_and_align_gnn_user_embeddings(model, processor, base_dir, device):
    """ GNN User Embedding Alignment (Standard Embedding) """
    print(f"\n🔄 [GNN User Alignment] Starting...")
    
    model_dir = globals().get('MODEL_DIR', os.path.join(base_dir, 'models'))
    cache_dir = os.path.join(base_dir, "cache")
    
    model_path = os.path.join(model_dir, "simgcl_trained.pth")
    maps_path = os.path.join(cache_dir, "id_maps_train.pt")

    try:
        maps = torch.load(maps_path, map_location='cpu')
        gnn_user2id = maps['user2id']
        state_dict = torch.load(model_path, map_location='cpu')
        
        # 유저 임베딩 키 찾기 ('embedding_user.weight'가 일반적)
        user_key = next((k for k in state_dict.keys() if 'embedding_user' in k), None)
        
        if user_key is None:
            raise ValueError(f"User embedding key not found. Keys: {list(state_dict.keys())[:5]}")
            
        gnn_user_weight = state_dict[user_key]

    except Exception as e:
        print(f"❌ [Error] Failed to load GNN User files: {e}")
        return model

    num_users = len(processor.user_ids) + 1
    embed_dim = gnn_user_weight.shape[1]
    new_weight = torch.randn(num_users, embed_dim) * 0.01
    new_weight[0] = 0.0
    
    matched = 0
    for i, current_id_str in enumerate(processor.user_ids):
        if current_id_str in gnn_user2id:
            gnn_idx = gnn_user2id[current_id_str]
            if gnn_idx < len(gnn_user_weight):
                new_weight[i + 1] = gnn_user_weight[gnn_idx]
                matched += 1
            
    with torch.no_grad():
        model.gnn_user_emb = nn.Embedding.from_pretrained(new_weight.to(device), freeze=False)
        print(f"✅ Injected into 'model.gnn_user_emb'")

    print(f"✅ [GNN User Alignment] Matched: {matched}/{len(processor.user_ids)}")
    return model

def verify_gnn_checkpoint_keys(model_path):
    print(f"\n🔎 [Inspection] Checking keys in: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ File not found!")
        return

    try:
        state_dict = torch.load(model_path, map_location='cpu')
        keys = list(state_dict.keys())
        
        print(f"   -> Total Keys: {len(keys)}")
        print("   -> Key Examples:")
        
        # User 관련 키 확인
        user_keys = [k for k in keys if 'user' in k]
        print(f"      👤 User Keys: {user_keys}")
        
        # Item/Adapter 관련 키 확인
        item_keys = [k for k in keys if 'item' in k or 'adapter' in k]
        print(f"      📦 Item/Adapter Keys: {item_keys}")
        
        # Shape 확인
        if user_keys:
            print(f"      Shape of {user_keys[0]}: {state_dict[user_keys[0]].shape}")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def verify_embedding_alignment(model, processor, model_dir):
    # (생략: 기존 코드와 동일, 필요시 추가)
    pass

# ==========================================
# 3. Model Definition (Fixed)
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceCentricFusion(nn.Module):
    """
    [설계 철학]
    1. 경쟁(Softmax)을 제거합니다. Sequence는 무조건 1.0의 비중을 가집니다.
    2. GNN과 Meta는 Sequence 벡터를 Query로 사용하여, 
       Sequence가 '필요하다고 판단할 때만' 정보가 더해(Add)집니다.
    3. 초기에는 GNN/Meta 반영률을 0에 수렴하게 하여 Sequence 학습을 강제합니다.
    """
    def __init__(self, dim=128):
        super().__init__()
        
        # Sequence가 GNN/Meta를 얼마나 가져올지 결정하는 Gate
        # 입력: Sequence (Context)
        # 출력: 2 (GNN gate, Meta gate) -> Softmax 아님! Sigmoid 사용
        self.context_gate = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 2), # [0]: GNN Gate, [1]: Meta Gate
            nn.Sigmoid()      # 0.0 ~ 1.0 독립적인 확률
        )
        
        # 차원 투영 (Projector)
        self.gnn_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        
        self.meta_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        
        # 최종 정리는 LayerNorm만 (MLP 통과 X -> 정보 희석 방지)
        self.final_ln = nn.LayerNorm(dim)

        # 🔥 [핵심 초기화]
        # Gate의 마지막 레이어 바이어스를 음수로 설정하여
        # 초기 Sigmoid 출력이 0에 가깝게 만듦 (예: -5 -> sigmoid(-5) ≈ 0.006)
        # 이렇게 하면 첫 Epoch에는 GNN/Meta가 거의 반영되지 않고 Sequence만 학습됨.
        nn.init.zeros_(self.context_gate[-2].weight)
        nn.init.constant_(self.context_gate[-2].bias, -5.0) 

    def forward(self, v_gnn, v_seq, v_meta):
        # 1. Gate 계산 (Sequence가 결정함)
        # gates: (Batch, Seq_Len, 2)
        gates = self.context_gate(v_seq)
        
        g_gnn = gates[..., 0:1]
        g_meta = gates[..., 1:2]
        
        # 2. Residual Addition (경쟁하지 않고 더하기만 함)
        # v_seq (Main) + (Gate * GNN) + (Gate * Meta)
        # Sequence는 계수가 1로 고정이므로 절대 무시되지 않음
        fused = v_seq + (g_gnn * self.gnn_proj(v_gnn)) + (g_meta * self.meta_proj(v_meta))
        
        # 3. Norm & Return
        # Gate 가중치도 리턴하여 로깅 (평균값)
        gnn_ratio = g_gnn.mean().item()
        meta_ratio = g_meta.mean().item()
        gate_weights = [gnn_ratio, meta_ratio]

        return self.final_ln(fused), gate_weights

# ==========================================
# 🧩 3. Parallel Adapter (유지)
# ==========================================
class ParallelAdapter(nn.Module):
    def __init__(self, content_dim=128, gnn_dim=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.content_proj = nn.Sequential(
            nn.Linear(content_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, v_content, v_gnn):
        # [수정] Content Embedding에 Residual Connection 추가 (+ v_content)
        # v_content(원본)가 Adapter를 통과한 결과와 더해짐 -> 원본 정보 보존
        merged = (self.content_proj(v_content) + v_content) + self.gnn_proj(v_gnn)
        return merged

# ==========================================
# 🏰 Hybrid User Tower (수정됨)
# ==========================================
class HybridUserTower(nn.Module):
    def __init__(self, num_users, num_items, gnn_user_init, gnn_item_init, item_content_init):
        super().__init__()
        self.embed_dim = 128

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 1. Embeddings
        self.gnn_user_emb = nn.Embedding.from_pretrained(gnn_user_init, freeze=False)
        self.gnn_item_emb = nn.Embedding.from_pretrained(gnn_item_init, freeze=False)
        self.item_content_emb = nn.Embedding.from_pretrained(item_content_init, freeze=False)
        
        # 2. Adapters
        self.gnn_projector = nn.Sequential(
            nn.Linear(gnn_user_init.shape[1], 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128), nn.LayerNorm(128)
        )
        
        # [수정] ParallelAdapter 사용
        self.seq_adapter = ParallelAdapter(
            content_dim=128, 
            gnn_dim=64, 
            out_dim=128, 
            dropout=DROPOUT
        )
        
        # 3. Sequence Modeling
        self.time_emb = nn.Embedding(1001, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=2, dim_feedforward=512, 
            dropout=DROPOUT, batch_first=True, norm_first=True
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 4. Meta & Fusion
        self.channel_emb = nn.Embedding(2, 32)
        self.meta_mlp = nn.Sequential(
            nn.Linear(35, 128), nn.GELU(),  # Target Layer Monitoring
            nn.Linear(128, 128), nn.LayerNorm(128)
        )
        self.fusion_layer = SequenceCentricFusion(dim=128)
        
        
        
        
    def get_current_temperature(self, clamp_min):
        # 사용할 때는 exp를 취해서 양수로 만듦
        # 1 / exp(scale) = temperature
        # 하지만 보통 계산 효율을 위해 (Cosine Sim * Scale) 방식으로 곱해버림
        # 여기서는 기존 Loss 함수와의 호환성을 위해 Temperature 값으로 변환해서 리턴
        
        # logit_scale을 최대 100(exp(4.6))까지만 커지게 제한 (CLIP 논문 테크닉 - 발산 방지)
        scale = self.logit_scale.exp().clamp(clamp_min, max=100.0)
        
        #clamp(min=14.3)
        # Scale = 1 / Temperature 이므로,
        # Temperature = 1 / Scale
        return 1.0 / scale
    
    def forward(self, u_idx, seq_ids, seq_deltas, seq_mask, u_dense, u_cat):
        B, L = seq_ids.shape
        
        # 1. GNN User
        v_gnn = self.gnn_projector(self.gnn_user_emb(u_idx))
        v_gnn_seq = F.normalize(v_gnn, p=2, dim=1).unsqueeze(1).expand(-1, L, -1)
        v_gnn_seq = torch.zeros_like(v_gnn_seq)
        if self.training:
            drop_prob = 0.4  # 40% 확률로 GNN을 버림
            keep_prob = 1 - drop_prob
            
            # 배치별 마스크 생성 (B, 1, 1)
            mask = torch.bernoulli(torch.full((B, 1, 1), keep_prob, device=v_gnn_seq.device))
            
            # Inverted Dropout: 살아남은 신호는 keep_prob로 나눠서 스케일 유지
            v_gnn_seq = (v_gnn_seq * mask) / keep_prob
        
        # =========================================================
        # [수정된 부분] 2. Dual-View Sequence (Parallel Adapter)
        # =========================================================
        # (1) 임베딩 꺼내기
        raw_content = self.item_content_emb(seq_ids) # (B, L, 128)
        raw_gnn = self.gnn_item_emb(seq_ids)         # (B, L, 64)
        
        # (2) Adapter 통과 (인자 2개 전달!)
        # 기존에는 cat으로 합쳐서 넣었지만, 이제는 따로 넣어야 합니다.
        seq_input = self.seq_adapter(raw_content, raw_gnn) # <--- 여기가 수정됨!
        
        # (3) Time Embedding
        seq_input = seq_input  * math.sqrt(self.embed_dim) + self.time_emb(seq_deltas.clamp(max=1000))
        
        # =========================================================
        
        causal_mask = torch.triu(torch.ones(L, L, device=seq_ids.device) * float('-inf'), diagonal=1)
        key_padding_mask = (seq_mask == 0)
        
        seq_out = self.seq_encoder(seq_input, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        v_seq = F.normalize(seq_out, p=2, dim=2)

        cat_vec = self.channel_emb(u_cat)
        v_meta = self.meta_mlp(torch.cat([u_dense, cat_vec], dim=1))
        v_meta_seq = F.normalize(v_meta, p=2, dim=1).unsqueeze(1).expand(-1, L, -1)
        
        output, gate_weights = self.fusion_layer(v_gnn_seq, v_seq, v_meta_seq)
        output = F.normalize(output, p=2, dim=2)
        return output, v_seq, gate_weights
    def get_meta_feature_importance(self):
        """
        Meta MLP의 첫 번째 Linear Layer 가중치를 분석하여
        어떤 Feature가 가장 영향력이 큰지 계산합니다.
        """
        # 첫 번째 Linear Layer의 가중치: (Out_Dim, In_Dim) -> (128, 35)
        weight_matrix = self.meta_mlp[0].weight.abs().detach().cpu()
        
        # Input Dimension Slicing
        # Price: 0~32, Cnt: 32~64, Recency: 64~96, Channel: 96~112
        imp_price = weight_matrix[:, 0:32].mean().item()
        imp_cnt = weight_matrix[:, 32:64].mean().item()
        imp_recency = weight_matrix[:, 64:96].mean().item()
        imp_channel = weight_matrix[:, 96:112].mean().item()
        
        # 정규화 (비율로 보기 위해)
        total = imp_price + imp_cnt + imp_recency + imp_channel + 1e-9
        return {
            "Price": imp_price / total,
            "Count": imp_cnt / total,
            "Recency": imp_recency / total,
            "Channel": imp_channel / total
        }
# ==========================================
# 4. Loss & Eval
# ==========================================
def logq_correction_loss(user_emb, item_emb, pos_item_ids, item_probs, temperature=0.07, lambda_logq=0.0):
    scores = torch.matmul(user_emb, item_emb.T)
    if lambda_logq > 0.0:
        
        log_q = torch.log(item_probs[pos_item_ids] + 1e-4).view(1, -1)
        scores = scores - (lambda_logq * log_q)
    logits = scores / temperature
    is_collision = (pos_item_ids.unsqueeze(1) == pos_item_ids.unsqueeze(0))
    mask = is_collision.fill_diagonal_(False)
    logits = logits.masked_fill(mask, -1e4)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def efficient_corrected_logq_loss(
    user_emb, 
    item_emb, 
    pos_item_ids, 
    precomputed_log_q, 
    temperature=0.1, 
    lambda_logq=0.1
):
    # 인덱스 범위 체크 (디버깅용, 실제 학습시 성능 영향 미미)
    assert pos_item_ids.max() < precomputed_log_q.size(0), "pos_item_ids contains out-of-bounds index!"
    logits = torch.matmul(user_emb, item_emb.T)
    logits.div_(temperature) # logits /= temperature (In-place)
    
    if lambda_logq > 0.0:
        # 2. LogQ Correction (In-place)
        # precomputed_log_q에서 현재 배치의 값만 슬라이싱 (View 생성)
        batch_log_q = precomputed_log_q[pos_item_ids].view(1, -1)
        
        # In-place subtraction: 새로운 텐서 할당 최소화
        logits.sub_(batch_log_q * lambda_logq)
        
        # 3. Positive Recovery (RecSys 2025)
        # torch.sum 대신 einsum을 쓰면 가끔 특정 CUDA 버전에서 더 효율적입니다.
        pos_logits_raw = torch.einsum('bd,bd->b', user_emb, item_emb).div_(temperature)
        logits.diagonal().copy_(pos_logits_raw)

    # 4. Collision Masking (메모리 절약형)
    with torch.no_grad():
        is_collision = (pos_item_ids.unsqueeze(1) == pos_item_ids.unsqueeze(0))
        mask = is_collision.fill_diagonal_(False)
    
    # FP16 AMP 사용 시 -3e4가 안전 (Underflow 방지)
    mask_value = -30000.0 if logits.dtype == torch.float16 else -1e9
    logits.masked_fill_(mask, mask_value)

    # 5. Labels 생성 (매번 생성하지 않고 재사용 가능하지만, 이 정도는 미미함)
    labels = torch.arange(logits.size(0), device=logits.device)
    
    return F.cross_entropy(logits, labels)







def evaluate_multi_vector_ensemble(
    seq_model, processor, target_df_path, gnn_user_matrix, gnn_item_matrix, 
    device, k_list=[20, 100, 500], batch_size=4096, 
    alpha_step=0.2
):
    """
    Multi-Vector Retrieval Ensemble Evaluation
    
    Logic:
      1. GNN과 Seq Model이 각각 독립적으로 User Vector와 Item Vector를 생성.
      2. 각각 전체 아이템에 대해 Score 계산 후 Top-K(Max) 추출.
      3. 지정된 비율(Alpha)에 따라 GNN의 상위 N개와 Seq의 상위 M개를 혼합하여 최종 추천 리스트 구성.
      
    Args:
      alpha: GNN의 반영 비율 (1.0 = GNN Only, 0.0 = Seq Only)
    """
    max_k = max(k_list)
    print(f"\n🚀 Starting Multi-Vector Retrieval Ensemble (Max K: {max_k})...")
    
    seq_model.eval()
    
    # ---------------------------------------------------------
    # 1. Target Data Load & Valid Loader
    # ---------------------------------------------------------
    target_df = pd.read_parquet(target_df_path)
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    
    val_loader = DataLoader(
        UserTowerDataset(processor, is_training=False), 
        batch_size=batch_size, shuffle=False, collate_fn=user_tower_collate_fn
    )
    
    # ---------------------------------------------------------
    # 2. Pre-computation (Global Item Vectors)
    # ---------------------------------------------------------
    print("⚡ Pre-computing Item Vectors for both models...")
    with torch.no_grad():
        all_item_ids = torch.arange(1, len(processor.item_ids)+1).to(device)
        
        # [Seq Model] Item Vectors
        seq_item_vecs_list = []
        for i in range(0, len(all_item_ids), 4096):
            chunk = all_item_ids[i:i+4096]
            c_emb = seq_model.item_content_emb(chunk)
            g_emb = seq_model.gnn_item_emb(chunk)
            c_vec = seq_model.seq_adapter(c_emb, g_emb)
            seq_item_vecs_list.append(F.normalize(c_vec, p=2, dim=1))
        all_seq_item_vecs = torch.cat(seq_item_vecs_list, dim=0)

        # [GNN Model] Item Vectors
        # index 0 is padding, so start from 1
        all_gnn_item_vecs = F.normalize(gnn_item_matrix[1:].to(device), p=2, dim=1)

    # ---------------------------------------------------------
    # 3. Setup Evaluation Metrics
    # ---------------------------------------------------------
    # Alpha: 1.0 (GNN 100%) ~ 0.0 (Seq 100%)
    alphas = [round(x, 1) for x in np.arange(1.0, -0.01, -alpha_step)]
    
    # Results Container: {Alpha: {K: Count}}
    results = {a: {k: 0 for k in k_list} for a in alphas}
    total_users = 0
    
    # ---------------------------------------------------------
    # 4. Evaluation Loop
    # ---------------------------------------------------------
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="   -> Multi-Vector Retrieval"):
            u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, _, _ = [x.to(device) for x in batch]
            
            # 유효 유저 필터링
            batch_uids = [processor.user_ids[i-1] for i in u_idx.cpu().numpy()]
            valid_idx_list = [i for i, uid in enumerate(batch_uids) if uid in target_dict]
            
            if not valid_idx_list: continue
            
            # 유효한 인덱스만 텐서로 변환
            v_idx = torch.tensor(valid_idx_list).to(device)
            current_batch_size = len(v_idx)
            
            # =========================================================
            # [Step A] Independent Inference (Vector Generation)
            # =========================================================
            
            # A-1. GNN User Vectors
            user_gnn_vecs = gnn_user_matrix[u_idx[v_idx]].to(device)
            user_gnn_vecs = F.normalize(user_gnn_vecs, p=2, dim=1)
            
            # A-2. Seq User Vectors
            output = seq_model(
                u_idx[v_idx], seq_ids[v_idx], seq_deltas[v_idx], seq_mask[v_idx], u_dense[v_idx], u_cat[v_idx]
            )
            if isinstance(output, tuple): output = output[0]
            lengths = seq_mask[v_idx].sum(dim=1)
            last_indices = (lengths - 1).clamp(min=0)
            user_seq_vecs = output[torch.arange(current_batch_size), last_indices]
            user_seq_vecs = F.normalize(user_seq_vecs, p=2, dim=1) # Normalize for Cosine Sim
            
            # =========================================================
            # [Step B] Independent Retrieval (Top-K per Model)
            # =========================================================
            
            # B-1. GNN Retrieval (Batch Matmul)
            # (Batch, Dim) @ (Num_Items, Dim).T -> (Batch, Num_Items)
            scores_gnn = torch.matmul(user_gnn_vecs, all_gnn_item_vecs.T)
            # 가장 큰 K에 대해서만 미리 추출 (CPU로 옮겨서 병합 연산 부하 줄이기 위함)
            _, indices_gnn = torch.topk(scores_gnn, k=max_k, dim=1)
            indices_gnn = indices_gnn.cpu().numpy()
            
            # B-2. Seq Retrieval
            scores_seq = torch.matmul(user_seq_vecs, all_seq_item_vecs.T)
            _, indices_seq = torch.topk(scores_seq, k=max_k, dim=1)
            indices_seq = indices_seq.cpu().numpy()
            
            # =========================================================
            # [Step C] Ratio-based Merging & Scoring
            # =========================================================
            
            # 타겟 정답지 준비
            batch_target_sets = []
            for original_idx in valid_idx_list:
                u_id = batch_uids[original_idx]
                # item_id to index (0-based) conversion needed if item_vecs are 0-based
                # Note: indices_gnn/seq returns 0-based index of all_item_ids.
                # all_item_ids[k] corresponds to ItemID k+1.
                # So indices match (ItemID - 1).
                actual_indices = set(processor.item2id[tid] - 1 for tid in target_dict[u_id] if tid in processor.item2id)
                batch_target_sets.append(actual_indices)
            
            # Alpha Loop
            for alpha in alphas:
                # 각 K에 대해 혼합 비율 적용
                for k in k_list:
                    # 비율 계산 (GNN 개수, Seq 개수)
                    n_gnn = int(k * alpha)
                    n_seq = k - n_gnn
                    
                    # Batch 내 각 유저별로 혼합 수행
                    for u in range(current_batch_size):
                        actual = batch_target_sets[u]
                        if not actual: continue
                        
                        # 1. GNN에서 상위 n_gnn개 추출
                        # 2. Seq에서 상위 n_seq개 추출
                        # 3. 합집합 (순서는 상관없음, Recall 측정용 Set)
                        
                        # 슬라이싱 시 n=0이면 빈 배열 반환
                        set_gnn = set(indices_gnn[u, :n_gnn]) if n_gnn > 0 else set()
                        set_seq = set(indices_seq[u, :n_seq]) if n_seq > 0 else set()
                        
                        # Multi-Source Merge
                        pred_set = set_gnn | set_seq
                        
                        # Recall Check
                        # (교집합이 하나라도 있으면 Hit)
                        if not actual.isdisjoint(pred_set):
                            results[alpha][k] += 1

            total_users += len(valid_idx_list)

    # ---------------------------------------------------------
    # 5. Report
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"📊 Multi-Vector Ensemble Report (Total Users: {total_users})")
    print(f"{'-'*80}")
    
    header = f"{'Alpha(GNN)':<12} | {'GNN:Seq Ratio':<15}"
    for k in k_list:
        header += f" | {f'Recall@{k}':<12}"
    print(header)
    print(f"{'-'*80}")
    
    best_alpha = -1
    best_score = -1
    
    # Sort alphas desc (1.0 -> 0.0)
    for alpha in sorted(results.keys(), reverse=True):
        scores = {}
        for k in k_list:
            scores[k] = results[alpha].get(k, 0) / total_users if total_users > 0 else 0
            
        row_str = f"{alpha:<12.1f} | {f'{int(alpha*10)} : {int((1-alpha)*10)}':<15}"
        for k in k_list:
            row_str += f" | {scores[k]:<12.4f}"
        print(row_str)
        
        # Best Selection (Based on smallest K usually, or largest K)
        # Here we use the first K (Recall@20) as primary metric
        if scores[k_list[0]] > best_score:
            best_score = scores[k_list[0]]
            best_alpha = alpha
            
    print(f"{'='*80}")
    print(f"🏆 Best Ensemble Ratio (GNN): {best_alpha}")
    
    return best_alpha

import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def evaluate_weighted_score_ensemble(
    seq_model, processor, target_df_path, gnn_user_matrix, gnn_item_matrix, 
    device, k_list=[20, 100, 500], batch_size=4096, 
    alpha_step=0.1, candidate_pool_size=1000
):
    """
    Weighted Score Fusion Ensemble Evaluation (Late Fusion)
    
    Logic:
      1. 각 모델별로 넉넉한 후보군(candidate_pool_size)을 추출하여 합집합(Union)을 만듭니다.
      2. 합쳐진 후보군 아이템에 대해 두 모델의 Score를 각각 계산합니다.
      3. Min-Max Normalization을 적용하여 점수 스케일을 맞춥니다.
      4. Alpha 비율대로 가중 합산(Weighted Sum)하여 최종 랭킹을 산출합니다.
    """
    max_k = max(k_list)
    # 후보군 사이즈는 목표 K보다 커야 앙상블 효과가 납니다. (보통 2~5배 추천)
    pool_k = max(candidate_pool_size, max_k * 2)
    
    print(f"\n🚀 Starting Weighted Score Ensemble (Pool K: {pool_k} -> Select Top K)...")
    
    seq_model.eval()
    
    # ---------------------------------------------------------
    # 1. Target Data & Loader
    # ---------------------------------------------------------
    target_df = pd.read_parquet(target_df_path)
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    
    val_loader = DataLoader(
        UserTowerDataset(processor, is_training=False), 
        batch_size=batch_size, shuffle=False, collate_fn=user_tower_collate_fn
    )
    
    # ---------------------------------------------------------
    # 2. Pre-computation (Item Vectors)
    # ---------------------------------------------------------
    print("⚡ Pre-computing Item Vectors...")
    with torch.no_grad():
        all_item_ids = torch.arange(1, len(processor.item_ids)+1).to(device)
        
        # [Seq Model]
        seq_item_vecs_list = []
        for i in range(0, len(all_item_ids), 4096):
            chunk = all_item_ids[i:i+4096]
            c_emb = seq_model.item_content_emb(chunk)
            g_emb = seq_model.gnn_item_emb(chunk)

            c_vec = seq_model.seq_adapter(c_emb, g_emb)
            seq_item_vecs_list.append(F.normalize(c_vec, p=2, dim=1))
        all_seq_item_vecs = torch.cat(seq_item_vecs_list, dim=0)

        # [GNN Model]
        all_gnn_item_vecs = F.normalize(gnn_item_matrix[1:].to(device), p=2, dim=1)

    # ---------------------------------------------------------
    # 3. Evaluation Setup
    # ---------------------------------------------------------
    alphas = [round(x, 1) for x in np.arange(1.0, -0.01, -alpha_step)]
    results = {a: {k: 0 for k in k_list} for a in alphas}
    total_users = 0
    
    
    
    # ---------------------------------------------------------
    # 4. Main Loop
    # ---------------------------------------------------------
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="   -> Weighted Score Fusion"):
            u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, _, _ = [x.to(device) for x in batch]
            
            # 유효 유저 필터링
            batch_uids = [processor.user_ids[i-1] for i in u_idx.cpu().numpy()]
            valid_idx_list = [i for i, uid in enumerate(batch_uids) if uid in target_dict]
            
            if not valid_idx_list: continue
            v_idx = torch.tensor(valid_idx_list).to(device)
            current_batch_size = len(v_idx)
            
            # =========================================================
            # [Step A] User Vector Generation
            # =========================================================
            # GNN User Vec
            user_gnn_vecs = gnn_user_matrix[u_idx[v_idx]].to(device)
            user_gnn_vecs = F.normalize(user_gnn_vecs, p=2, dim=1)
            
            # Seq User Vec
            output = seq_model(
                u_idx[v_idx], seq_ids[v_idx], seq_deltas[v_idx], seq_mask[v_idx], u_dense[v_idx], u_cat[v_idx]
            )
            if isinstance(output, tuple): output = output[0]
            lengths = seq_mask[v_idx].sum(dim=1)
            last_indices = (lengths - 1).clamp(min=0)
            user_seq_vecs = output[torch.arange(current_batch_size), last_indices]
            # Normalize는 Cosine Similarity 계산을 위해 필수 (특히 Score Fusion에서 스케일 영향 최소화)
            user_seq_vecs = F.normalize(user_seq_vecs, p=2, dim=1)

            # =========================================================
            # [Step B] Candidate Pool Generation (Union of Top-M)
            # =========================================================
            # 모든 아이템에 대해 계산하면 느리므로, 각 모델의 Top-M개를 뽑아 합집합을 만듭니다.
            
            # GNN Global Scores
            scores_gnn_all = torch.matmul(user_gnn_vecs, all_gnn_item_vecs.T)
            _, indices_gnn_top = torch.topk(scores_gnn_all, k=pool_k, dim=1)
            
            # Seq Global Scores
            scores_seq_all = torch.matmul(user_seq_vecs, all_seq_item_vecs.T)
            _, indices_seq_top = torch.topk(scores_seq_all, k=pool_k, dim=1)
            
            # Union Indices (Smart Gathering)
            # 배치 내 각 유저별로 후보 아이템 인덱스를 모읍니다.
            # 효율적인 병렬 처리를 위해 gather 방식을 사용합니다.
            
            # (Batch, 2 * pool_k) 형태로 병합
            combined_indices = torch.cat([indices_gnn_top, indices_seq_top], dim=1)
            
            # =========================================================
            # [Step C] Score Calculation on Union Set
            # =========================================================
            # 각 유저별로 선택된 아이템들의 Vector만 가져와서 내적 (Efficient)
            
            # 1. Gather Item Vectors based on Combined Indices
            # combined_indices: (Batch, Pool_Size) -> flattened for gathering
            flat_indices = combined_indices.view(-1)
            
            # (Batch * Pool_Size, Dim)
            batch_gnn_items = all_gnn_item_vecs[flat_indices].view(current_batch_size, -1, all_gnn_item_vecs.shape[1])
            batch_seq_items = all_seq_item_vecs[flat_indices].view(current_batch_size, -1, all_seq_item_vecs.shape[1])
            
            # 2. Recalculate Scores (User * Item)
            # (Batch, 1, Dim) * (Batch, Pool_Size, Dim) -> sum -> (Batch, Pool_Size)
            s_gnn = (user_gnn_vecs.unsqueeze(1) * batch_gnn_items).sum(dim=-1)
            s_seq = (user_seq_vecs.unsqueeze(1) * batch_seq_items).sum(dim=-1)
            
            # =========================================================
            # [Step D] Min-Max Normalization (Crucial!)
            # =========================================================
            # 모델마다 점수 분포(평균, 분산)가 다르므로 0~1 사이로 맞춰줍니다.
            def min_max_norm(tensor):
                min_val = tensor.min(dim=1, keepdim=True)[0]
                max_val = tensor.max(dim=1, keepdim=True)[0]
                return (tensor - min_val) / (max_val - min_val + 1e-9)
            
            s_gnn_norm = min_max_norm(s_gnn)
            s_seq_norm = min_max_norm(s_seq)
            
            # 타겟 정답지 준비
            batch_targets = []
            for original_idx in valid_idx_list:
                u_id = batch_uids[original_idx]
                actual_indices = set(processor.item2id[tid] - 1 for tid in target_dict[u_id] if tid in processor.item2id)
                batch_targets.append(actual_indices)
                
            # combined_indices (Local Index -> Global Index 매핑용)
            combined_indices_cpu = combined_indices.cpu().numpy()
            
            # =========================================================
            # [Step E] Alpha Sweep & Metric
            # =========================================================
            for alpha in alphas:
                # Weighted Sum
                final_scores = alpha * s_gnn_norm + (1.0 - alpha) * s_seq_norm
                
                # Top-K Selection (Local Index)
                # 여기서 중복된 아이템이 있을 수 있음 (Union 과정에서) -> 하지만 점수는 동일하므로 문제 없음
                # 다만 완벽을 위해 TopK 후 Global ID로 변환하여 중복 제거 필요할 수 있으나,
                # topk가 충분히 크지 않으면 큰 영향 없음. 정석대로라면 unique 처리가 필요.
                # 여기서는 속도를 위해 바로 TopK 후 검증 단계에서 Set으로 처리.
                
                _, local_topk_indices = torch.topk(final_scores, k=max_k + 20, dim=1) # 넉넉히 추출 (중복 대비)
                local_topk_indices = local_topk_indices.cpu().numpy()
                
                for i, actual_indices in enumerate(batch_targets):
                    if not actual_indices: continue
                    
                    # Local Index -> Global Item ID 복원
                    # combined_indices[i] : 해당 유저의 후보군 글로벌 ID들
                    # local_topk_indices[i] : 그 후보군 안에서의 등수
                    pred_global_ids = combined_indices_cpu[i][local_topk_indices[i]]
                    
                    # 중복 제거하면서 Top-K 유지 (unique_preserve_order)
                    _, unique_idx = np.unique(pred_global_ids, return_index=True)
                    pred_unique = pred_global_ids[np.sort(unique_idx)]
                    
                    for k in k_list:
                        # 상위 K개만 잘라서 정답 확인
                        top_k_items = pred_unique[:k]
                        if not actual_indices.isdisjoint(top_k_items):
                            results[alpha][k] += 1
                            
            total_users += len(valid_idx_list)

    # ---------------------------------------------------------
    # 5. Report
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"📊 Weighted Score Fusion Report (Pool: {pool_k})")
    print(f"{'-'*80}")
    
    header = f"{'Alpha(GNN)':<12} | {'GNN:Seq Ratio':<15}"
    for k in k_list:
        header += f" | {f'Recall@{k}':<12}"
    print(header)
    print(f"{'-'*80}")
    
    best_alpha = -1
    best_score = -1
    
    for alpha in sorted(results.keys(), reverse=True):
        scores = {}
        for k in k_list:
            scores[k] = results[alpha].get(k, 0) / total_users if total_users > 0 else 0
            
        row_str = f"{alpha:<12.1f} | {f'{int(alpha*10)} : {int((1-alpha)*10)}':<15}"
        for k in k_list:
            row_str += f" | {scores[k]:<12.4f}"
        print(row_str)
        
        # Best Metric Update (Recall@20 기준)
        if scores[k_list[0]] > best_score:
            best_score = scores[k_list[0]]
            best_alpha = alpha
            
    print(f"{'='*80}")
    print(f"🏆 Best Weighted Alpha: {best_alpha}")
    
    return best_alpha
# ==========================================
# 6. Main Execution Flow
# ==========================================
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def evaluate_rrf_ensemble(
    seq_model, processor, target_df_path, gnn_user_matrix, gnn_item_matrix, 
    device, k_list=[20, 100, 500], batch_size=4096, 
    alpha_step=0.1, candidate_pool_size=1000, k_rrf=200
):
    """
    Weighted RRF (Reciprocal Rank Fusion) Ensemble Evaluation
    
    Logic:
      1. GNN과 Seq 모델이 각각 Top-N 후보를 뽑아 합집합(Union)을 만듭니다.
      2. 합쳐진 후보군에 대해 두 모델의 점수를 다시 계산합니다.
      3. 점수 대신 **등수(Rank)**를 산출합니다.
      4. RRF 공식: Score = alpha * (1 / (k_rrf + rank1)) + (1-alpha) * (1 / (k_rrf + rank2))
      
    Args:
      k_rrf: RRF 상수로, 보통 60을 많이 사용합니다. (랭킹이 낮아도 점수가 너무 0이 되지 않게 완화)
    """
    max_k = max(k_list)
    pool_k = max(candidate_pool_size, max_k * 2)
    
    print(f"\n🚀 Starting Weighted RRF Ensemble (Pool K: {pool_k}, RRF Constant: {k_rrf})...")
    
    seq_model.eval()
    
    # ---------------------------------------------------------
    # 1. Target Data & Loader (기존과 동일)
    # ---------------------------------------------------------
    target_df = pd.read_parquet(target_df_path)
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    
    val_loader = DataLoader(
        UserTowerDataset(processor, is_training=False), 
        batch_size=batch_size, shuffle=False, collate_fn=user_tower_collate_fn
    )
    
    # ---------------------------------------------------------
    # 2. Pre-computation (Item Vectors) (기존과 동일)
    # ---------------------------------------------------------
    print("⚡ Pre-computing Item Vectors...")
    with torch.no_grad():
        all_item_ids = torch.arange(1, len(processor.item_ids)+1).to(device)
        
        # [Seq Model]
        seq_item_vecs_list = []
        for i in range(0, len(all_item_ids), 4096):
            chunk = all_item_ids[i:i+4096]
            c_emb = seq_model.item_content_emb(chunk)
            g_emb = seq_model.gnn_item_emb(chunk)
            c_vec = seq_model.seq_adapter(c_emb, g_emb)
            seq_item_vecs_list.append(F.normalize(c_vec, p=2, dim=1))
        all_seq_item_vecs = torch.cat(seq_item_vecs_list, dim=0)

        # [GNN Model]
        all_gnn_item_vecs = F.normalize(gnn_item_matrix[1:].to(device), p=2, dim=1)

    # ---------------------------------------------------------
    # 3. Evaluation Setup
    # ---------------------------------------------------------
    alphas = [round(x, 1) for x in np.arange(1.0, -0.01, -alpha_step)]
    results = {a: {k: 0 for k in k_list} for a in alphas}
    total_users = 0
    
    # RRF 랭킹 계산을 위한 헬퍼 텐서 (배치 처리를 위함)
    # 미리 만들지 않고 배치 루프 안에서 생성
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="   -> Weighted RRF Ranking"):
            u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, _, _ = [x.to(device) for x in batch]
            
            # 유효 유저 필터링
            batch_uids = [processor.user_ids[i-1] for i in u_idx.cpu().numpy()]
            valid_idx_list = [i for i, uid in enumerate(batch_uids) if uid in target_dict]
            
            if not valid_idx_list: continue
            v_idx = torch.tensor(valid_idx_list).to(device)
            current_batch_size = len(v_idx)
            
            # =========================================================
            # [Step A] User Vector Generation
            # =========================================================
            user_gnn_vecs = F.normalize(gnn_user_matrix[u_idx[v_idx]].to(device), p=2, dim=1)
            
            output = seq_model(u_idx[v_idx], seq_ids[v_idx], seq_deltas[v_idx], seq_mask[v_idx], u_dense[v_idx], u_cat[v_idx])
            if isinstance(output, tuple): output = output[0]
            lengths = seq_mask[v_idx].sum(dim=1)
            last_indices = (lengths - 1).clamp(min=0)
            user_seq_vecs = F.normalize(output[torch.arange(current_batch_size), last_indices], p=2, dim=1)

            # =========================================================
            # [Step B] Candidate Pool Generation (Union)
            # =========================================================
            # 각 모델별 Top-K 추출 (속도를 위해)
            scores_gnn_all = torch.matmul(user_gnn_vecs, all_gnn_item_vecs.T)
            _, indices_gnn_top = torch.topk(scores_gnn_all, k=pool_k, dim=1)
            
            scores_seq_all = torch.matmul(user_seq_vecs, all_seq_item_vecs.T)
            _, indices_seq_top = torch.topk(scores_seq_all, k=pool_k, dim=1)
            
            # Union Indices (Batch, 2 * pool_k)
            combined_indices = torch.cat([indices_gnn_top, indices_seq_top], dim=1)
            
            # =========================================================
            # [Step C] Score Recalculation (For Ranking)
            # =========================================================
            flat_indices = combined_indices.view(-1)
            batch_gnn_items = all_gnn_item_vecs[flat_indices].view(current_batch_size, -1, all_gnn_item_vecs.shape[1])
            batch_seq_items = all_seq_item_vecs[flat_indices].view(current_batch_size, -1, all_seq_item_vecs.shape[1])
            
            # (Batch, Pool_Size) - 점수 계산
            raw_scores_gnn = (user_gnn_vecs.unsqueeze(1) * batch_gnn_items).sum(dim=-1)
            raw_scores_seq = (user_seq_vecs.unsqueeze(1) * batch_seq_items).sum(dim=-1)
            
            # =========================================================
            # [Step D] Convert Scores to Ranks (핵심 변경 부분)
            # =========================================================
            # RRF를 위해서는 점수가 아니라 '순위(Rank)'가 필요합니다.
            # torch.argsort(descending=True)를 두 번 쓰면 랭크를 얻을 수 있습니다.
            # 예: scores=[0.1, 0.9, 0.5] -> argsort1=[1, 2, 0] (인덱스) -> argsort2=[2, 0, 1] (랭크: 0.9가 0등)
            
            # 1. Sort하여 인덱스 확보
            _, sorted_idx_gnn = torch.sort(raw_scores_gnn, dim=1, descending=True)
            _, sorted_idx_seq = torch.sort(raw_scores_seq, dim=1, descending=True)
            
            # 2. 원래 위치에 랭크(등수) 할당 (Scatter)
            # 0등부터 시작하므로 +1은 나중에 RRF 공식에서 처리하거나 여기서 미리 처리
            rank_gnn = torch.zeros_like(raw_scores_gnn)
            rank_seq = torch.zeros_like(raw_scores_seq)
            
            # arange를 배치 크기만큼 확장 (0, 1, ..., Pool_Size-1)
            ranks_range = torch.arange(combined_indices.size(1)).to(device).expand(current_batch_size, -1)
            
            # sorted_idx 위치에 0, 1, 2... 순서대로 값을 뿌려줌
            rank_gnn.scatter_(1, sorted_idx_gnn, ranks_range.float())
            rank_seq.scatter_(1, sorted_idx_seq, ranks_range.float())
            
            # =========================================================
            # [Step E] RRF & Alpha Sweep
            # =========================================================
            # RRF Formula: 1 / (k + rank + 1)  (Rank는 0-based라 가정시 +1 필요, 혹은 k에 포함)
            # 여기서는 Rank가 0부터 시작하므로 (k_rrf + rank + 1)로 계산
            
            rrf_score_gnn = 1.0 / (k_rrf + rank_gnn + 1.0)
            rrf_score_seq = 1.0 / (k_rrf + rank_seq + 1.0)
            
            combined_indices_cpu = combined_indices.cpu().numpy()
            
            # 타겟 준비
            batch_targets = []
            for original_idx in valid_idx_list:
                u_id = batch_uids[original_idx]
                actual_indices = set(processor.item2id[tid] - 1 for tid in target_dict[u_id] if tid in processor.item2id)
                batch_targets.append(actual_indices)

            for alpha in alphas:
                # Weighted RRF
                final_rrf_scores = (alpha * rrf_score_gnn) + ((1.0 - alpha) * rrf_score_seq)
                
                # Top-K Selection
                _, local_topk_indices = torch.topk(final_rrf_scores, k=max_k + 20, dim=1)
                local_topk_indices = local_topk_indices.cpu().numpy()
                
                for i, actual_indices in enumerate(batch_targets):
                    if not actual_indices: continue
                    
                    # Local Index -> Global Item ID
                    pred_global_ids = combined_indices_cpu[i][local_topk_indices[i]]
                    
                    # 중복 제거 (Unique)
                    _, unique_idx = np.unique(pred_global_ids, return_index=True)
                    pred_unique = pred_global_ids[np.sort(unique_idx)]
                    
                    for k in k_list:
                        if not actual_indices.isdisjoint(pred_unique[:k]):
                            results[alpha][k] += 1
                            
            total_users += len(valid_idx_list)

    # ---------------------------------------------------------
    # 4. Report
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"📊 Weighted RRF Ensemble Report (k_rrf: {k_rrf})")
    print(f"{'-'*80}")
    
    header = f"{'Alpha(GNN)':<12} | {'GNN:Seq Ratio':<15}"
    for k in k_list:
        header += f" | {f'Recall@{k}':<12}"
    print(header)
    print(f"{'-'*80}")
    
    best_alpha = -1
    best_score = -1
    
    for alpha in sorted(results.keys(), reverse=True):
        scores = {}
        for k in k_list:
            scores[k] = results[alpha].get(k, 0) / total_users if total_users > 0 else 0
            
        row_str = f"{alpha:<12.1f} | {f'{int(alpha*10)} : {int((1-alpha)*10)}':<15}"
        for k in k_list:
            row_str += f" | {scores[k]:<12.4f}"
        print(row_str)
        
        if scores[k_list[0]] > best_score:
            best_score = scores[k_list[0]]
            best_alpha = alpha
            
    print(f"{'='*80}")
    print(f"🏆 Best RRF Alpha: {best_alpha}")
    
    return best_alpha

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

def evaluate_gnn_standalone(
    model, processor, target_df_path, device, 
    k_list=[20, 100, 500], batch_size=4096
):
    """
    Pure GNN Retrieval Evaluation
    
    Logic:
      1. 모델 내 주입된 GNN User Embedding과 GNN Item Embedding을 추출합니다.
      2. User Vector와 전체 Item Vector 간의 Cosine Similarity를 계산합니다.
      3. Top-K를 추출하여 정답(Target)과 비교합니다.
      (Sequence Model의 Logit이나 Score는 전혀 사용하지 않습니다.)
    """
    max_k = max(k_list)
    print(f"\n🚀 Starting Standalone GNN Evaluation (Max K: {max_k})...")
    
    model.eval()
    
    # ---------------------------------------------------------
    # 1. Target Data Load
    # ---------------------------------------------------------
    target_df = pd.read_parquet(target_df_path)
    # customer_id -> [target_item_id1, target_item_id2, ...]
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    
    # ---------------------------------------------------------
    # 2. Valid Loader Setup
    # ---------------------------------------------------------
    val_loader = DataLoader(
        UserTowerDataset(processor, is_training=False), 
        batch_size=batch_size, shuffle=False, collate_fn=user_tower_collate_fn
    )
    
    # ---------------------------------------------------------
    # 3. Pre-computation (GNN Item Matrix)
    # ---------------------------------------------------------
    print("⚡ Extracting & Normalizing GNN Item Vectors...")
    with torch.no_grad():
        # 모델에 저장된 GNN 아이템 임베딩 전체를 가져옵니다.
        # Index 0은 Padding이므로 제외하거나 포함해도 0벡터라 영향 없음 (여기선 1부터 사용)
        # 하지만 Indexing 편의를 위해 전체를 가져오고 0번은 무시하는 전략 사용
        all_gnn_items = model.gnn_item_emb.weight.data.clone().detach().to(device)
        
        # Cosine Similarity를 위한 L2 Normalization
        # (Batch Matmul 시 Dot Product만 하면 됨)
        all_gnn_items_norm = F.normalize(all_gnn_items, p=2, dim=1)

    # ---------------------------------------------------------
    # 4. Evaluation Loop
    # ---------------------------------------------------------
    results = {k: 0 for k in k_list}
    total_users = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="   -> GNN Retrieval"):
            u_idx, _, _, _, _, _, _, _ = [x.to(device) for x in batch]
            
            # 유효 유저 필터링 (Target이 있는 유저만)
            batch_uids = [processor.user_ids[i-1] for i in u_idx.cpu().numpy()]
            valid_idx_list = [i for i, uid in enumerate(batch_uids) if uid in target_dict]
            
            if not valid_idx_list: continue
            
            v_idx = torch.tensor(valid_idx_list).to(device)
            valid_u_idx = u_idx[v_idx]
            current_batch_size = len(v_idx)
            
            # =========================================================
            # [Step A] User Vector Extraction
            # =========================================================
            # 모델의 GNN User Embedding에서 조회
            batch_gnn_user = model.gnn_user_emb(valid_u_idx)
            batch_gnn_user_norm = F.normalize(batch_gnn_user, p=2, dim=1)
            
            # =========================================================
            # [Step B] Retrieval (Dot Product)
            # =========================================================
            # (Batch, Dim) @ (Num_Items, Dim).T -> (Batch, Num_Items)
            # all_gnn_items_norm은 (N+1, Dim) 형태 (0번은 패딩)
            scores = torch.matmul(batch_gnn_user_norm, all_gnn_items_norm.T)
            
            # 0번 인덱스(Padding)가 검색되지 않도록 마스킹 (선택사항, 안전장치)
            scores[:, 0] = -float('inf')
            
            # Top-K 추출
            _, topk_indices = torch.topk(scores, k=max_k, dim=1)
            topk_indices = topk_indices.cpu().numpy()
            
            # =========================================================
            # [Step C] Metric Calculation
            # =========================================================
            for i, original_idx in enumerate(valid_idx_list):
                u_id = batch_uids[original_idx]
                
                # 정답 ID -> Index 변환
                # processor.item2id는 String ID -> 1-based Index 매핑
                actual_indices = set(
                    processor.item2id[tid] 
                    for tid in target_dict[u_id] 
                    if tid in processor.item2id
                )
                
                if not actual_indices: continue
                
                # Recall Check
                pred_items = topk_indices[i] # 이미 1-based index (Embedding Index와 동일)
                
                for k in k_list:
                    # 상위 k개 중에 정답이 하나라도 있는가?
                    if not actual_indices.isdisjoint(pred_items[:k]):
                        results[k] += 1
            
            total_users += len(valid_idx_list)

    # ---------------------------------------------------------
    # 5. Report
    # ---------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"📊 GNN Standalone Performance Report")
    print(f"   (Total Users: {total_users})")
    print(f"{'-'*60}")
    
    header = f"{'Metric':<15} | {'Value':<10}"
    print(header)
    print(f"{'-'*60}")
    
    for k in sorted(k_list):
        recall = results[k] / total_users if total_users > 0 else 0
        print(f"{f'Recall@{k}':<15} | {recall:.4f}")
            
    print(f"{'='*60}\n")

# 사용 예시
# evaluate_gnn_standalone(model, valid_proc, TARGET_VAL_PATH, DEVICE)
def main():
    # 1. 초기화: Feature Processor 로드
    # (Train Processor를 로드하여 ID 매핑 기준을 잡음)
    print("1️⃣ Initializing Processors...")
    train_proc = FeatureProcessor(USER_FEAT_PATH_PQ, ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ)
    valid_proc = FeatureProcessor(
        USER_VAL_FEAT_PATH,  # 검증 유저 피처
        ITEM_FEAT_PATH_PQ,   # 아이템 피처 (공유)
        SEQ_VAL_DATA_PATH,   # ⭐ 핵심: 검증용 시퀀스 (Target 제외)
        scaler=train_proc.user_scaler # Scaler 공유
    )
    
    # [중요] ID 매핑을 Train과 동일하게 강제 일치
    # (새로운 아이템/유저가 있으면 무시하거나 처리하기 위해)
    valid_proc.user2id = train_proc.user2id
    valid_proc.item2id = train_proc.item2id
    valid_proc.user_ids = train_proc.user_ids 
    valid_proc.item_ids = train_proc.item_ids
    num_users = len(train_proc.user_ids) + 1
    num_items = len(train_proc.item_ids) + 1
    
    # 2. 모델 초기화 (Dummy Init)
    print("2️⃣ Initializing Model...")
    # 실제로는 Pretrained Tensor를 로드해서 넣어야 함
    dummy_gnn_u = torch.randn(num_users, 64)
    dummy_gnn_i = torch.randn(num_items, 64)
    dummy_content = torch.randn(num_items, 128)
    
    model = HybridUserTower(
        num_users=num_users,
        num_items=num_items,
        gnn_user_init=dummy_gnn_u,
        gnn_item_init=dummy_gnn_i,
        item_content_init=dummy_content
    ).to(DEVICE)
    
    # 3. 임베딩 정렬 & 로드
    model = load_and_align_embeddings(model, train_proc, MODEL_DIR, DEVICE)

    model = load_and_align_gnn_items(model, train_proc, BASE_DIR, DEVICE)
    # 4. 학습된 가중치 로드
    if os.path.exists(SAVE_PATH_BEST):
        print(f"3️⃣ Loading Trained Weights form {SAVE_PATH_BEST}...")
        # strict=False: 임베딩 사이즈 등이 미세하게 다를 경우 유연하게 로드
        model.load_state_dict(torch.load(SAVE_PATH_BEST), strict=False)
        model = load_and_align_gnn_items(model, train_proc, BASE_DIR, DEVICE)
        gnn_item_matrix = model.gnn_item_emb.weight.data.clone().detach()
    else:
        print("⚠️ Trained weights not found. Using random init.")
    
    model = load_and_align_gnn_user_embeddings(model, train_proc, BASE_DIR, DEVICE)
    # 5. GNN 매트릭스 추출 (앙상블용)
    # 모델 내부에 정렬되어 저장된 GNN 임베딩을 꺼내서 사용
    print("4️⃣ Extracting GNN Matrices...")
    gnn_user_matrix = model.gnn_user_emb.weight.data.clone().detach()


    # 6. 앙상블 평가 수행
    # valid_proc 대신 train_proc을 사용 (데이터 경로만 Validation용으로 지정하면 됨)
    # 실제로는 Valid Set에 대한 Processor를 따로 만드는 것이 정석이지만, 
    # 여기서는 편의상 ID 매핑이 동일한 train_proc 사용 + Valid Target Path 주입
    '''
    evaluate_ensemble_sweep(
        seq_model=model,
        processor=valid_proc, 
        target_df_path=TARGET_VAL_PATH, 
        gnn_user_matrix=gnn_user_matrix, 
        gnn_item_matrix=gnn_item_matrix,
        device=DEVICE
    )
    
    evaluate_rrf_ensemble_sweep(        seq_model=model,
        processor=valid_proc, 
        target_df_path=TARGET_VAL_PATH, 
        gnn_user_matrix=gnn_user_matrix, 
        gnn_item_matrix=gnn_item_matrix,
        device=DEVICE)
        '''
    evaluate_weighted_score_ensemble(
        seq_model=model,
        processor=valid_proc, 
        target_df_path=TARGET_VAL_PATH, 
        gnn_user_matrix=gnn_user_matrix, 
        gnn_item_matrix=gnn_item_matrix,
        device=DEVICE,
        k_list=[20, 100, 500], # 평가할 K 사이즈
        alpha_step=0.2         # 비율 변경 단위
    )


    
if __name__ == "__main__":
    main()





