import torch
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler


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
