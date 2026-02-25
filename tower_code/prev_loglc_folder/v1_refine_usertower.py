import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm





def dataset_peek(dataset, processor):
    """Dataset에서 1개 샘플을 꺼내 로직이 정합한지 검수"""
    print("\n🧐 [Data Peek] Checking Sequence Integrity...")
    sample = dataset[0]
    
    # 1. 시퀀스 Shift 확인
    ids = sample['item_ids'].tolist()
    targets = sample['target_ids'].tolist()
    
    # 0이 아닌 첫 번째 실제 데이터 인덱스 찾기
    first_idx = next((i for i, x in enumerate(ids) if x != 0), None)
    
    if first_idx is not None and first_idx < len(ids) - 1:
        print(f"   - Input Seq  (t):   ... {ids[first_idx:first_idx+3]}")
        print(f"   - Target Seq (t+1): ... {targets[first_idx:first_idx+3]}")
        if ids[first_idx+1] == targets[first_idx]:
            print("   ✅ Shift Logic: OK (Input[t+1] == Target[t])")
        else:
            print("   ❌ Shift Logic: ERROR! Target is not shifted correctly.")

    # 2. 유저 스태틱 피처 확인
    print(f"   - Age Bucket ID: {sample['age_bucket'].item()}")
    print(f"   - Cont Feats Shape: {sample['cont_feats'].shape}")



class FeatureProcessor:
    def __init__(self, user_path, item_path, seq_path,base_processor=None):
        print("🚀 Loading preprocessed features...")
        self.users = pd.read_parquet(user_path).drop_duplicates(subset=['customer_id']).set_index('customer_id')
        self.items = pd.read_parquet(item_path).drop_duplicates(subset=['article_id']).set_index('article_id')
        self.seqs = pd.read_parquet(seq_path).set_index('customer_id')

        # 인덱스 타입 강제 (String)
        self.users.index = self.users.index.astype(str)
        self.items.index = self.items.index.astype(str)
        self.seqs.index = self.seqs.index.astype(str)

        # =================================================================
        # 1. ID Mappings (1-based, 0 is Padding)
        # =================================================================
        self.user_ids = self.seqs.index.tolist() # 시퀀스가 존재하는 유저만 대상
        self.user2id = {uid: i + 1 for i, uid in enumerate(self.users.index)}

    
        
        
        
        
        
        
        if base_processor is None:
            # Train일 때: 새롭게 아이템 번호표 생성
            self.item_ids = self.items.index.tolist()
            self.item2id = {iid: i + 1 for i, iid in enumerate(self.item_ids)}
            self.num_items = len(self.item_ids)
        else:
            # Validation일 때: Train의 번호표를 그대로 물려받음
            self.item_ids = base_processor.item_ids
            self.item2id = base_processor.item2id
            self.num_items = base_processor.num_items
        
        
        self.num_items = len(self.item_ids)
        
        

        # =================================================================
        # 2. Fast Lookup Arrays for Dataset (__getitem__ 속도 최적화)
        # =================================================================
        print("⚡ Building fast lookup tables...")
        
        # [A] User Features (유저 ID 1~N으로 바로 접근할 수 있도록 배열화)
        num_users_total = len(self.users) + 1
        
        # Bucket / Categorical (LongTensor용)
        self.u_bucket_arr = np.zeros((num_users_total, 4), dtype=np.int64) 
        self.u_cat_arr = np.zeros((num_users_total, 5), dtype=np.int64)
        # Continuous (FloatTensor용)
        self.u_cont_arr = np.zeros((num_users_total, 4), dtype=np.float32)
        
        self.u_last_date_arr = np.zeros(num_users_total, dtype=np.int64)
        
        # 매핑 수행
        for uid, row in self.users.iterrows():
            if uid not in self.user2id: continue
            uidx = self.user2id[uid]
            
            # Buckets: age, price, cnt, recency
            self.u_bucket_arr[uidx] = [
                row['age_bucket'], row['user_avg_price_bucket'], 
                row['total_cnt_bucket'], row['recency_bucket']
            ]
            # Categoricals: channel, club, news, fn, active
            self.u_cat_arr[uidx] = [
                row['preferred_channel'], row['club_member_status_idx'],
                row['fashion_news_frequency_idx'], row['FN'], row['Active']
            ]
            # Continuous Scaled: price_std, last_diff, repurch, weekend
            self.u_cont_arr[uidx] = [
                row['price_std_scaled'], row['last_price_diff_scaled'],
                row['repurchase_ratio_scaled'], row['weekend_ratio_scaled']
            ]
            if pd.notnull(row['last_purchase_date']):
                dt_val = pd.to_datetime(row['last_purchase_date'])
                self.u_last_date_arr[uidx] = dt_val.toordinal()
        # [B] Item Side Info Lookup (아이템 ID 1~N으로 바로 접근)
        # 아이템 데이터 프레임에 type_id, color_id 등이 있다고 가정
        self.i_side_arr = np.zeros((self.num_items + 1, 4), dtype=np.int64)
        for iid, row in self.items.iterrows():
            if iid not in self.item2id: continue
            idx = self.item2id[iid]
            # 전처리된 아이템 피처에 맞춰 컬럼명 수정 필요
            self.i_side_arr[idx] = [
                row.get('type_id', 0), row.get('color_id', 0), 
                row.get('graphic_id', 0), row.get('section_id', 0)
            ]

    def get_logq_probs(self, device):
        """Negative Sampling이나 Loss 보정을 위한 아이템 등장 확률 Log 반환"""
        raw_probs = self.items['raw_probability'].reindex(self.item_ids).values
        eps = 1e-6
        sorted_probs = np.nan_to_num(raw_probs, nan=0.0) + eps
        sorted_probs /= sorted_probs.sum()
        
        log_q_values = np.log(sorted_probs).astype(np.float32)
        
        full_log_q = np.zeros(self.num_items + 1, dtype=np.float32)
        full_log_q[1:] = log_q_values 
        full_log_q[0] = -20.0 # Padding Index
    
        return torch.tensor(full_log_q, dtype=torch.float32).to(device)


    # FeatureProcessor 클래스 내부에 추가할 메서드
    def analyze_distributions(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        print("\n📊 [Data Distribution Analysis]")
        print("-" * 50)

        # 1. 시퀀스 길이 분포 (max_len 결정의 핵심 근거)
        seq_lengths = self.seqs['sequence_ids'].apply(len)
        
        print(f"🔹 Sequence Length Stats:")
        print(f"   - Mean: {seq_lengths.mean():.2f}")
        print(f"   - Median: {seq_lengths.median()}")
        print(f"   - P90: {seq_lengths.quantile(0.9):.1f}")
        print(f"   - P95: {seq_lengths.quantile(0.95):.1f}")
        print(f"   - Max: {seq_lengths.max()}")

        plt.figure(figsize=(12, 5))
        
        # Left: Sequence Length Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(seq_lengths, bins=50, kde=True, color='skyblue')
        plt.axvline(seq_lengths.quantile(0.95), color='red', linestyle='--', label='P95')
        plt.title("User Sequence Length Distribution")
        plt.xlabel("Length")
        plt.legend()

        # 2. 주요 유저 카테고리 분포 (Age, Price Bucket 등)
        # u_bucket_arr에서 0번(Age), 1번(Price) 컬럼 추출 (Padding 제외하고 1번 인덱스부터)
        plt.subplot(1, 2, 2)
        ages = self.u_bucket_arr[1:, 0]
        sns.countplot(x=ages, palette='viridis')
        plt.title("User Age Bucket Distribution")
        plt.xlabel("Age Bucket ID")

        plt.tight_layout()
        plt.show()

        # 3. 아이템 등장 빈도 (Long-tail 확인)
        all_items_in_seqs = [iid for subseq in self.seqs['sequence_ids'] for iid in subseq]
        item_counts = pd.Series(all_items_in_seqs).value_counts()
        
        print(f"\n🔹 Item Interaction Stats:")
        print(f"   - Total Unique Items in Seqs: {len(item_counts)}")
        print(f"   - Top 10% items cover {item_counts.iloc[:int(len(item_counts)*0.1)].sum() / len(all_items_in_seqs) * 100:.1f}% of interactions")
        
        # 4. ID Mapping Coverage 확인 (디버깅용)
        missing_items = [iid for iid in item_counts.index if iid not in self.item2id]
        if missing_items:
            print(f"⚠️ Warning: {len(missing_items)} items in sequences are NOT in item master!")
        else:
            print("✅ Success: All items in sequences are correctly mapped.")
        
class SASRecDataset(Dataset):
    def __init__(self, processor: FeatureProcessor, global_now_str ="2020-09-22", max_len=30, is_train=True):
        self.processor = processor
        self.max_len = max_len
        self.is_train = is_train
        self.user_ids = processor.user_ids
        
        global_now_dt = pd.to_datetime(global_now_str)
        self.now_ordinal = global_now_dt.toordinal()
        self.now_week = global_now_dt.isocalendar().week

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        u_mapped_id = self.processor.user2id.get(user_id, 0)
        
        # 1. 시퀀스 로드 
        seq_raw = self.processor.seqs.loc[user_id, 'sequence_ids']
        
        # 1-1. time deltas : 1년전 동일계절에 구매했던건? 최근은? 등등을 매핑
        time_deltas_raw = self.processor.seqs.loc[user_id, 'sequence_deltas']
        bins = np.array([0, 3, 7, 14, 30, 60, 180, 330, 395])
        time_buckets = np.digitize(time_deltas_raw, bins, right=False).tolist()
        
        
        seq = [self.processor.item2id.get(item, 0) for item in seq_raw]
        
        # =========================================================
        # 2. Causality Split (SASRec Shift Logic)
        # =========================================================
        if self.is_train:
            # 학습 시: input과 target을 위해 max_len + 1 개를 가져옴
            seq = seq[-(self.max_len + 1):]
            time_buckets = time_buckets[-(self.max_len + 1):] # [신규 추가] 타임 버킷도 동일하게 슬라이싱
            if len(seq) > 1:
                input_seq = seq[:-1]  # t 시점까지의 입력
                target_seq = seq[1:]  # t+1 시점의 정답
                input_time = time_buckets[:-1] # [신규 추가] t 시점의 시간 간격
            else:
                input_seq = seq
                target_seq = seq # 방어 코드 (길이가 1인 경우)
                input_time = time_buckets
        else:
            # 추론/검증 시: 최신 max_len 개를 입력으로 사용 (다음 1개를 예측하기 위해)
            input_seq = seq[-self.max_len:]
            target_seq = [] # Test loop에서 정답을 별도로 처리
            input_time = time_buckets[-self.max_len:] # [신규 추가]

        # =========================================================
        # 3. Left Padding
        # =========================================================
        # 최근 행동이 배열의 끝에 오도록 Left Padding을 적용
        pad_len = self.max_len - len(input_seq)
        input_padded = [0] * pad_len + input_seq
        time_padded = [0] * pad_len + input_time
        if self.is_train:
            target_padded = [0] * pad_len + target_seq
        else:
            target_padded = [0] * self.max_len

        # =========================================================
        # 4. Item Side Info Lookup (Sequence)
        # =========================================================
        # padding(0)인 경우 Lookup 배열의 0번째 인덱스(0,0,0,0)를 가져옴
        item_side_info = self.processor.i_side_arr[input_padded]
        
        type_ids = item_side_info[:, 0]
        color_ids = item_side_info[:, 1]
        graphic_ids = item_side_info[:, 2]
        section_ids = item_side_info[:, 3]

        # Padding Mask (True면 Transformer에서 무시)
        padding_mask = [True] * pad_len + [False] * len(input_seq)

        # =========================================================
        # 5. User Features Lookup (Static)
        # =========================================================
        u_buckets = self.processor.u_bucket_arr[u_mapped_id]
        u_cats = self.processor.u_cat_arr[u_mapped_id]
        u_conts = self.processor.u_cont_arr[u_mapped_id]

        # ====================
        # global time 
        # =====================
        last_ordinal = self.processor.u_last_date_arr[u_mapped_id]
    
        if last_ordinal == 0: # 패딩이거나 구매 이력이 아예 없는 경우
            recency_offset = 365
            current_week = self.now_week # 예비용 기본값
        else:
            # 1. Recency Offset (기존 유지: 오늘로부터 얼마나 낡았는가?)
            recency_offset = self.now_ordinal - last_ordinal
            recency_offset = max(0, min(365, recency_offset))
            
            # 🔥 2. Current Week (핵심 수정: 유저의 마지막 구매 시점의 주차!)
            last_date = datetime.date.fromordinal(last_ordinal)
            current_week = last_date.isocalendar().week
        
        # =========================================================
        # 6. Return Tensors
        # =========================================================
        return {
            
            'user_ids': user_id,
            # Sequence
            'item_ids': torch.tensor(input_padded, dtype=torch.long),
            'target_ids': torch.tensor(target_padded, dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'time_bucket_ids': torch.tensor(time_padded, dtype=torch.long),
            
            # Item Side Info
            'type_ids': torch.tensor(type_ids, dtype=torch.long),
            'color_ids': torch.tensor(color_ids, dtype=torch.long),
            'graphic_ids': torch.tensor(graphic_ids, dtype=torch.long),
            'section_ids': torch.tensor(section_ids, dtype=torch.long),
            
            # User Buckets
            'age_bucket': torch.tensor(u_buckets[0], dtype=torch.long),
            'price_bucket': torch.tensor(u_buckets[1], dtype=torch.long),
            'cnt_bucket': torch.tensor(u_buckets[2], dtype=torch.long),
            'recency_bucket': torch.tensor(u_buckets[3], dtype=torch.long),
            
            # User Categoricals
            'channel_ids': torch.tensor(u_cats[0], dtype=torch.long),
            'club_status_ids': torch.tensor(u_cats[1], dtype=torch.long),
            'news_freq_ids': torch.tensor(u_cats[2], dtype=torch.long),
            'fn_ids': torch.tensor(u_cats[3], dtype=torch.long),
            'active_ids': torch.tensor(u_cats[4], dtype=torch.long),
            
            # User Continuous
            'cont_feats': torch.tensor(u_conts, dtype=torch.float32),
            
            # 💡 [추가] Global Context (모델의 Early Injection 입력용)
            'recency_offset': torch.tensor(recency_offset, dtype=torch.long),
            'current_week': torch.tensor(current_week, dtype=torch.long)
        }
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRecUserTower(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.max_len = args.max_len
        self.dropout_rate = args.dropout

        # ==================================================================
        # 1. Sequence Embeddings (Dynamic: Short-term Intent)
        # ==================================================================
        self.item_proj = nn.Linear(args.pretrained_dim, self.d_model)
        self.item_id_emb = nn.Embedding(args.num_items + 1, self.d_model, padding_idx=0)
        
        self.type_emb = nn.Embedding(args.num_prod_types + 1, self.d_model, padding_idx=0)
        self.color_emb = nn.Embedding(args.num_colors + 1, self.d_model, padding_idx=0)
        self.graphic_emb = nn.Embedding(args.num_graphics + 1, self.d_model, padding_idx=0)
        self.section_emb = nn.Embedding(args.num_sections + 1, self.d_model, padding_idx=0)

        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        # 시퀀스 피처용 (item_id, time, type, color, graphic, section) -> 6개
        self.seq_gate = nn.Parameter(torch.ones(6)) 
        
        # 스테틱 피처용 (age, price, cnt, rec, chan, club, news, fn, act, cont) -> 10개
        self.static_gate = nn.Parameter(torch.ones(10))
        # [업데이트] Time-Aware 버킷 임베딩
        num_time_buckets = 12 
        self.time_emb = nn.Embedding(num_time_buckets, self.d_model, padding_idx=0)
        
        self.emb_ln = nn.LayerNorm(self.d_model)
        self.emb_dropout = nn.Dropout(self.dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=args.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=self.dropout_rate,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)

        # ==================================================================
        # 2. Static Embeddings (Global: Long-term Preference)
        # ==================================================================
        #  (A) Categorical Embeddings (Cardinality에 따른 효율적 차원 할당)
        
        # 10구간 Bucket 피처들 (상대적으로 정보량이 많음) -> 16차원
        mid_dim = 16
        self.age_emb = nn.Embedding(11, mid_dim, padding_idx=0)      
        self.price_emb = nn.Embedding(11, mid_dim, padding_idx=0)    
        self.cnt_emb = nn.Embedding(11, mid_dim, padding_idx=0)      
        self.recency_emb = nn.Embedding(11, mid_dim, padding_idx=0)  

        # Binary 및 Low-Cardinality 피처들 -> 4차원
        low_dim = 4
        self.channel_emb = nn.Embedding(4, low_dim, padding_idx=0)   
        self.club_status_emb = nn.Embedding(4, low_dim, padding_idx=0) 
        self.news_freq_emb = nn.Embedding(3, low_dim, padding_idx=0)   
        self.fn_emb = nn.Embedding(3, low_dim, padding_idx=0)        
        self.active_emb = nn.Embedding(3, low_dim, padding_idx=0)    

        # (B) Continuous Features Projection
        # 4차원의 연속형 데이터를 16차원으로 키워 임베딩과 볼륨을 맞춤
        self.num_cont_feats = 4
        cont_proj_dim = 16
        self.cont_proj = nn.Linear(self.num_cont_feats, cont_proj_dim)

        # 모든 Static Feature의 Concat 후 총 차원 계산
        # (16 * 4) + (4 * 5) + 16 = 64 + 20 + 16 = 100
        total_static_input_dim = (mid_dim * 4) + (low_dim * 5) + cont_proj_dim
        
        self.static_mlp = nn.Sequential(
            nn.Linear(total_static_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

        # ==================================================================
        # 3. Final Fusion & Output
        # ==================================================================
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    def get_causal_mask(self, seq_len, device):
        # float('-inf') 대신 dtype=torch.bool을 사용하여 True/False 행렬로 생성
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    def forward(self, 
                # Sequence Inputs (Batch, Seq)
                pretrained_vecs, item_ids, 
                time_bucket_ids, 
                type_ids, color_ids, graphic_ids, section_ids,
                # Static Categorical Inputs (Batch, )
                age_bucket, price_bucket, cnt_bucket, recency_bucket,
                channel_ids, club_status_ids, news_freq_ids, fn_ids, active_ids,
                # Static Continuous Inputs (Batch, 4)
                cont_feats, 
                padding_mask=None,
                training_mode=True
                ):
        
        device = item_ids.device
        seq_len = item_ids.size(1)
        
        s_g_raw = torch.sigmoid(self.seq_gate) 
        u_g_raw = torch.sigmoid(self.static_gate)
        
        s_mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], device=s_g_raw.device)
        s_g = s_g_raw * s_mask  # 곱셈 연산은 새로운 텐서를 생성하므로 안전합니다.

        u_mask = torch.ones_like(u_g_raw)
        # u_mask[6:9] = 0.0 # 필요한 경우 주석 해제
        u_g = u_g_raw * u_mask
        
        # -----------------------------------------------------------
        # Phase 1: Sequence Encoding (Short-term)
        # -----------------------------------------------------------
        seq_emb = self.item_proj(pretrained_vecs) 
        seq_emb += self.item_id_emb(item_ids) * s_g[0]
        seq_emb += self.time_emb(time_bucket_ids) * s_g[1]
        seq_emb += self.type_emb(type_ids) * s_g[2]
        seq_emb += self.color_emb(color_ids) * s_g[3]
        seq_emb += self.graphic_emb(graphic_ids) * s_g[4]
        seq_emb += self.section_emb(section_ids) * s_g[5]
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        seq_emb += self.pos_emb(positions)
        
        seq_emb = self.emb_ln(seq_emb)
        seq_emb = self.emb_dropout(seq_emb)

        causal_mask = self.get_causal_mask(seq_len, device)
        output = self.transformer_encoder(
            seq_emb, 
            mask=causal_mask, 
            src_key_padding_mask=padding_mask
        )

        # -----------------------------------------------------------
        # Phase 2: Static Encoding (Long-term)
        # -----------------------------------------------------------
        #  Dataset에서 전달받은 모든 피처들을 개별 임베딩
        emb_age = self.age_emb(age_bucket) * u_g[0]
        emb_price = self.price_emb(price_bucket) * u_g[1]
        emb_cnt = self.cnt_emb(cnt_bucket) * u_g[2]
        emb_rec = self.recency_emb(recency_bucket) * u_g[3]
        
        emb_chan = self.channel_emb(channel_ids) * u_g[4]
        emb_club = self.club_status_emb(club_status_ids) * u_g[5]
        emb_news = self.news_freq_emb(news_freq_ids) * u_g[6]
        emb_fn = self.fn_emb(fn_ids) * u_g[7]
        emb_act = self.active_emb(active_ids) * u_g[8]
        
        # 연속형 변수에도 게이트 적용 가능
        cont_proj_vec = F.relu(self.cont_proj(cont_feats)) * u_g[9]
        
        # Concat All Static Features
        static_input = torch.cat([
            emb_age, emb_price, emb_cnt, emb_rec,
            emb_chan, emb_club, emb_news, emb_fn, emb_act,
            cont_proj_vec
        ], dim=1)
        
        # MLP Processing
        user_profile_vec = self.static_mlp(static_input) # (Batch, d_model)

        # -----------------------------------------------------------
        # Phase 3: Late Fusion
        # -----------------------------------------------------------
        if training_mode:
            user_profile_expanded = user_profile_vec.unsqueeze(1).expand(-1, seq_len, -1)
            final_vec = torch.cat([output, user_profile_expanded], dim=-1)
            final_vec = self.output_proj(final_vec)
            
            return F.normalize(final_vec, p=2, dim=-1)
        else:
            user_intent_vec = output[:, -1, :] 
            final_vec = torch.cat([user_intent_vec, user_profile_vec], dim=-1)
            final_vec = self.output_proj(final_vec)
            
            return F.normalize(final_vec, p=2, dim=-1)
        # -----------------------------------------------------------
        # SEQ + pretrained vec -> Transformer -> User Intent Vector late fusion
        # -----------------------------------------------------------
    
    
    
# ==========================================
# 1. Loss Functions (In-Batch Negative + LogQ)
# ==========================================
def inbatch_corrected_logq_loss(user_emb, item_tower_emb, target_ids, log_q_tensor, temperature=0.1, lambda_logq=1.0):
    """
    In-Batch Negative Sampling과 LogQ 보정이 적용된 효율적인 CrossEntropy Loss
    
    Args:
        user_emb: (N, Dim) - Batch 단위 유저 벡터 (Flatten 적용됨)
        item_tower_emb: (Num_Items, Dim) - 전체 아이템 임베딩
        target_ids: (N, ) - 정답 아이템 ID (Flatten 적용됨)
        log_q_tensor: (Num_Items, ) - 전체 아이템의 등장 확률(Log)
        temperature: (float) - Softmax Temperature
        lambda_logq: (float) - 편향 제어 강도 (보통 1.0)
    """
    N = user_emb.size(0)
    
    # 1. 배치 내 등장한 정답 아이템들의 임베딩만 추출 (N, Dim)
    # 전체 47,062개가 아닌 배치 내 N개만 사용하여 메모리를 극도로 절약합니다.
    batch_item_emb = item_tower_emb[target_ids]
    
    # 2. In-Batch Logits 계산 (N, N)
    # i번째 유저 벡터와 j번째 아이템 벡터의 내적 (대각선 원소가 정답)
    logits = torch.matmul(user_emb, batch_item_emb.T)
    logits.div_(temperature)

    # 3. LogQ 편향 보정 (Sampling Bias Correction)
    if lambda_logq > 0.0:
        # 배치 내 등장한 아이템들의 LogQ 값 추출 (N,)
        batch_log_q = log_q_tensor[target_ids]
        
        # Google RecSys 논문 수식: s^c(x, y) = s(x, y) - log(P(y))
        # 정답이든 오답이든 해당 아이템의 인기도(LogQ)만큼 로짓을 깎아줌
        # Broadcasting: (N, N) 행렬의 각 열(Column)에서 해당 아이템의 LogQ를 뺌
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # 4. 정답 Label 생성 (대각선 인덱스: 0, 1, 2, ..., N-1)
    # i번째 유저의 정답은 배치 내 i번째 아이템임
    # 1. 배치 내에 동일한 아이템이 있는지 확인하는 (N, N) True/False 마스크
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    
    # 2. 대각선(진짜 자신의 정답)은 유지해야 하므로 제외할 마스크 생성
    diag_mask = torch.eye(N, dtype=torch.bool, device=user_emb.device)
    
    # 3. 진짜 정답이 아니면서 아이템 ID만 겹치는 '억울한 오답(False Negatives)' 추출
    false_neg_mask = same_item_mask & ~diag_mask
    
    # 4. 억울한 오답들의 로짓을 -inf로 덮어씌워 모델이 페널티를 주지 못하게 차단
    logits.masked_fill_(false_neg_mask, float('-inf'))
    
    
    
    
    labels = torch.arange(N, device=user_emb.device)
    
    # 5. 최종 CrossEntropyLoss 계산
    return F.cross_entropy(logits, labels)


def duorec_loss_refined(user_emb_1, user_emb_2, target_ids, temperature=0.1, lambda_sup=0.1):
    """
    Supervised Contrastive Learning (SupCon) + NaN 방지 및 패딩 처리 완료
    """
    batch_size = user_emb_1.size(0)
    device = user_emb_1.device
    
    # 1. 벡터 정규화
    z_i = F.normalize(user_emb_1, dim=1)
    z_j = F.normalize(user_emb_2, dim=1)
    
    # 2. Unsupervised Loss (InfoNCE)
    logits_unsup = torch.matmul(z_i, z_j.T) / temperature
    labels = torch.arange(batch_size, device=device)
    loss_unsup = F.cross_entropy(logits_unsup, labels)
    
    # 3. Supervised Loss
    loss_sup = torch.tensor(0.0, device=device)
    
    if lambda_sup > 0:
        targets = target_ids.view(-1, 1)
        
        # 같은 타겟을 공유하는 유저 Mask (Batch, Batch)
        mask = torch.eq(targets, targets.T).float()
        
        # [Fix 1: Padding 오인 방지] 타겟이 0(Padding)인 유저들은 전부 마스크 0으로 초기화
        pad_mask = (targets == 0).float()
        mask = mask * (1 - pad_mask) 
        
        # 자기 자신 제외
        mask.fill_diagonal_(0)
        
        if mask.sum() > 0:
            logits_sup = torch.matmul(z_i, z_i.T) / temperature
            diag_mask = torch.eye(batch_size, device=device).bool()
            
            # 대각선을 -inf로 마스킹 (자기 자신 제외)
            logits_sup.masked_fill_(diag_mask, float('-inf'))
            
            # Log Softmax 계산
            log_prob = F.log_softmax(logits_sup, dim=1)
            
            # [Fix 2: NaN 폭탄 방지] 대각선의 -inf가 mask(0)와 곱해져 NaN이 되는 것을 막기 위해 0.0으로 덮어씀
            log_prob = log_prob.masked_fill(diag_mask, 0.0)
            
            # Positive Sample이 존재하는 유저만 필터링
            valid_rows = mask.sum(1) > 0
            if valid_rows.sum() > 0:
                loss_sup_batch = -(mask[valid_rows] * log_prob[valid_rows]).sum(1) / mask[valid_rows].sum(1)
                loss_sup = loss_sup_batch.mean()
                
    return loss_unsup + (lambda_sup * loss_sup)


# ======================================================

def inbatch_hnm_corrected_loss_with_stats(
    user_emb, item_tower_emb, target_ids, log_q_tensor, 
    top_k_percent=0.01, hnm_threshold=0.90, temperature=0.1, lambda_logq=0.7, lambda_cl=0.2
):
    """
    Refactored HNM: Selection(Mining)과 Correction(LogQ)을 분리하여 '진짜 매운맛' 추출
    """
    N = user_emb.size(0)
    device = user_emb.device
    
    # 1. 정규화 및 기본 유사도 계산
    u_norm = F.normalize(user_emb, p=2, dim=1)
    i_batch_norm = F.normalize(item_tower_emb[target_ids], p=2, dim=1)
    
    # 순수 코사인 유사도 matrix (N, N)
    cos_sim = torch.matmul(u_norm, i_batch_norm.T) 

    # 2. 마스킹 (False Negative & Too Similar)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        item_sim = torch.matmul(i_batch_norm, i_batch_norm.T)
        too_similar_mask = (item_sim > hnm_threshold) & ~diag_mask
    
    ignore_mask = same_item_mask | too_similar_mask
    
    # 3. [핵심] 하드 네거티브 '선택' (Mining)
    # LogQ를 배제하고 오직 '유사도'가 높은 순서대로 K개를 뽑습니다.
    mining_logits = (cos_sim / temperature).detach().clone()
    mining_logits.masked_fill_(ignore_mask, float('-inf'))
    
    # 가용한 네거티브 개수 내에서 K 설정
    available_negs = (~ignore_mask).sum(dim=1)
    num_k = max(1, min(int((N - 1) * top_k_percent), available_negs.min().item()))
    
    _, top_k_indices = torch.topk(mining_logits, k=num_k, dim=1)
    
    # 4. [보정] 최종 로짓 구성 및 LogQ 적용
    # 선택은 유사도로 했지만, Loss를 계산할 때는 인기도 편향을 제거합니다.
    logits = cos_sim / temperature
    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        # Google 수식: s/temp - lambda * logQ
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # 5. 최종 Loss용 로짓 수집
    pos_logits = torch.diagonal(logits).unsqueeze(1)
    hard_neg_logits = torch.gather(logits, 1, top_k_indices)
    
    final_logits = torch.cat([pos_logits, hard_neg_logits], dim=1)
    labels = torch.zeros(N, dtype=torch.long, device=device)
    
    loss = F.cross_entropy(final_logits, labels)
    
    # 6. '매운맛' 통계 (보정 전 순수 유사도 기준)
    with torch.no_grad():
        hard_hn_sims = torch.gather(cos_sim, 1, top_k_indices)
        avg_hn_sim = hard_hn_sims.mean().item()

    return loss, {"avg_hn_similarity": avg_hn_sim, "num_active_hard_negs": num_k}


def inbatch_mixed_hnm_loss_with_stats(
    user_emb, item_tower_emb, target_ids, log_q_tensor, 
    top_k_percent=0.01, random_sample_size=100, 
    hnm_threshold=0.90, temperature=0.1, lambda_logq=0.7
):
    """
    Mixed Strategy: Hard Negatives (Top-K) + Random Negatives (M)
    """
    N = user_emb.size(0)
    device = user_emb.device
    
    # 1. 정규화 및 유사도 계산
    u_norm = F.normalize(user_emb, p=2, dim=1)
    i_batch_norm = F.normalize(item_tower_emb[target_ids], p=2, dim=1)
    cos_sim = torch.matmul(u_norm, i_batch_norm.T) 

    # 2. 마스킹 (False Negative & Too Similar)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        item_sim = torch.matmul(i_batch_norm, i_batch_norm.T)
        too_similar_mask = (item_sim > hnm_threshold) & ~diag_mask
    
    ignore_mask = same_item_mask | too_similar_mask
    
    # 3. [Mining] Hard Negative Selection (Top-K)
    mining_logits = (cos_sim / temperature).detach().clone()
    mining_logits.masked_fill_(ignore_mask, float('-inf'))
    
    num_k = max(1, int((N - 1) * top_k_percent))
    _, top_k_indices = torch.topk(mining_logits, k=num_k, dim=1)
    
    # 4. [Mining] Random Negative Selection
    # 하드 네거티브가 아닌 나머지 중에서 랜덤하게 추출
    # 구현 편의상 전체 배치에서 무작위로 뽑되, 마스킹된 것들은 이후 Loss에서 제외됨
    random_indices = torch.randint(0, N, (N, random_sample_size), device=device)

    # 5. [Correction] 최종 로짓 구성 (LogQ 적용)
    logits = cos_sim / temperature
    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # 6. 로짓 수집 (Positive + Hard + Random)
    pos_logits = torch.diagonal(logits).unsqueeze(1)
    hard_neg_logits = torch.gather(logits, 1, top_k_indices)
    random_neg_logits = torch.gather(logits, 1, random_indices)
    
    # [중요] Random 샘플 중 혹시나 Positive나 Too Similar가 섞였을 경우를 대비해 아주 낮은 값으로 처리
    # (효율을 위해 완전 제외 대신 페널티 부여)
    random_mask = torch.gather(ignore_mask, 1, random_indices)
    random_neg_logits.masked_fill_(random_mask, -1e9)

    final_logits = torch.cat([pos_logits, hard_neg_logits, random_neg_logits], dim=1)
    labels = torch.zeros(N, dtype=torch.long, device=device)
    
    loss = F.cross_entropy(final_logits, labels)
    
    # 통계 계산
    with torch.no_grad():
        hard_hn_sims = torch.gather(cos_sim, 1, top_k_indices)
        avg_hn_sim = hard_hn_sims.mean().item()

    return loss, {"avg_hn_similarity": avg_hn_sim, "num_hard": num_k, "num_random": random_sample_size}


def full_batch_hard_emphasis_loss(
    user_emb, item_tower_emb, target_ids, log_q_tensor, 
    top_k_percent=0.01, hard_margin=0.2, 
    hnm_threshold=0.90, temperature=0.1, lambda_logq=1.0
):
    """
    Full-Batch HNM: 
    1) 전체 배치(N-1)를 네거티브로 사용하여 Global Structure 유지
    2) 하드 네거티브에 Margin을 추가하여 정밀도(Hard Emphasis) 강화
    """
    N = user_emb.size(0)
    device = user_emb.device
    
    # 1. 정규화 및 유사도 계산
    u_norm = F.normalize(user_emb, p=2, dim=1)
    i_batch_norm = F.normalize(item_tower_emb[target_ids], p=2, dim=1)
    cos_sim = torch.matmul(u_norm, i_batch_norm.T) 

    # 2. 마스킹 (False Negative & Too Similar)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    ignore_mask = same_item_mask | ((torch.matmul(i_batch_norm, i_batch_norm.T) > hnm_threshold) & ~diag_mask)
    
    # 3. [Mining] 하드 네거티브 위치 찾기 (Top-K)
    with torch.no_grad():
        mining_sim = cos_sim.detach().clone()
        mining_sim.masked_fill_(ignore_mask, float('-inf'))
        num_k = max(1, int((N - 1) * top_k_percent))
        # 각 행(유저)별로 하드 네거티브의 '위치(인덱스)'를 확보
        _, top_k_indices = torch.topk(mining_sim, k=num_k, dim=1)

    # 4. [Correction] 전체 로짓 구성 및 LogQ 적용
    logits = cos_sim / temperature
    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # 5. [Hard Emphasis] 하드 네거티브에 Margin 추가
    # 하드 네거티브들의 로짓에 마진을 더해, 모델이 얘네를 '실제보다 더 가깝다'고 착각하게 만듦
    # 결과적으로 더 강한 힘으로 밀어내게 됨
    emphasis_mask = torch.zeros_like(logits, dtype=torch.bool)
    emphasis_mask.scatter_(1, top_k_indices, True)
    
    # 하드 네거티브 위치에만 마진 추가 (이게 '콕 집어 패는' 핵심)
    logits = logits + (emphasis_mask.float() * (hard_margin / temperature))

    # 6. [Final Masking] 억울한 오답(False Negatives) 차단
    # 자기 자신(Positive)을 제외한 나머지 겹치는 아이템들 제거
    false_neg_mask = same_item_mask & ~diag_mask
    logits.masked_fill_(false_neg_mask, float('-inf'))

    # 7. Loss 계산 (N x N 전체 사용)
    labels = torch.arange(N, device=device)
    loss = F.cross_entropy(logits, labels)
    
    # 통계
    with torch.no_grad():
        hard_hn_sims = torch.gather(cos_sim, 1, top_k_indices)
        avg_hn_sim = hard_hn_sims.mean().item()

    return loss, {"avg_hn_similarity": avg_hn_sim, "num_hard": num_k}



def inbatch_corrected_logq_loss(
    user_emb, item_tower_emb, target_ids, user_ids, log_q_tensor, # 💡 user_ids 추가
    temperature=0.1, lambda_logq=1.0
):
    N = user_emb.size(0)
    
    # 1. 배치 내 정답 아이템 임베딩 추출
    batch_item_emb = item_tower_emb[target_ids]
    
    # 2. In-Batch Logits 계산
    logits = torch.matmul(user_emb, batch_item_emb.T)
    logits.div_(temperature)

    # 3. LogQ 편향 보정
    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # 4. 마스킹 (False Negatives 차단)
    # (A) 아이템 ID가 우연히 같은 경우
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    # (B) 💡 [추가] 동일 유저의 다른 타임스텝 타겟인 경우 (A->B 예측할 때, A->C 예측 타겟이 네거티브가 되는 것 방지)
    same_user_mask = torch.eq(user_ids.unsqueeze(1), user_ids.unsqueeze(0))
    
    # 대각선(자신의 진짜 정답)은 유지
    diag_mask = torch.eye(N, dtype=torch.bool, device=user_emb.device)
    
    # 최종적으로 억울한 오답들을 걸러내는 마스크 (같은 아이템이거나 OR 같은 유저이거나)
    false_neg_mask = (same_item_mask | same_user_mask) & ~diag_mask
    
    # -inf로 덮어씌워 네거티브 연산에서 완전히 배제
    logits.masked_fill_(false_neg_mask, float('-inf'))
    
    # 5. 최종 CrossEntropyLoss 계산
    labels = torch.arange(N, device=user_emb.device)
    return F.cross_entropy(logits, labels)





'''

import torch
import torch.nn.functional as F

def inbatch_corrected_logq_loss_with_hard_neg(
    user_emb, item_tower_emb, target_ids, user_ids, log_q_tensor,
    batch_hard_neg_ids=None, # [N, 2] 크기의 텐서
    temperature=0.1, 
    lambda_logq=1.0,         # In-batch용 강한 보정
    hard_lambda_logq=0.0     # Hard Negative용 보정
):
    N = user_emb.size(0)
    
    # -------------------------------------------------------
    # 1. In-Batch Logits & LogQ 보정
    # -------------------------------------------------------
    batch_item_emb = item_tower_emb[target_ids]
    logits = torch.matmul(user_emb, batch_item_emb.T) / temperature

    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # In-Batch 마스킹 로직 (Same Item, Same User)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    same_user_mask = torch.eq(user_ids.unsqueeze(1), user_ids.unsqueeze(0))
    diag_mask = torch.eye(N, dtype=torch.bool, device=user_emb.device)
    
    false_neg_mask = (same_item_mask | same_user_mask) & ~diag_mask
    logits.masked_fill_(false_neg_mask, float('-inf'))

    # -------------------------------------------------------
    # 2. Hard Negative Injection & 패딩 마스킹 (💡 핵심)
    # -------------------------------------------------------
    if batch_hard_neg_ids is not None:
        # batch_hard_neg_ids shape: [N, num_hard_negs]
        hard_neg_emb = item_tower_emb[batch_hard_neg_ids] # [N, num_hard_negs, dim]
        
        # [N, 1, dim] x [N, dim, num_hard_negs] -> [N, 1, num_hard_negs] -> [N, num_hard_negs]
        hn_logits = torch.bmm(user_emb.unsqueeze(1), hard_neg_emb.transpose(1, 2)).squeeze(1)
        hn_logits.div_(temperature)
        
        if hard_lambda_logq > 0.0:
            hn_log_q = log_q_tensor[batch_hard_neg_ids]
            hn_logits = hn_logits - (hn_log_q * hard_lambda_logq)
            
        # 💡 [방어 로직] 뽑힌 Hard Negative가 0번(패딩/유효하지 않음)인 경우 -inf 처리
        invalid_hn_mask = (batch_hard_neg_ids == 0)
        hn_logits.masked_fill_(invalid_hn_mask, float('-inf'))
            
        # 기존 In-Batch Logits의 우측에 Hard Negative Logits를 결합
        # 최종 logits shape: [N, N + num_hard_negs]
        logits = torch.cat([logits, hn_logits], dim=1)
        
    # -------------------------------------------------------
    # 3. CrossEntropy Loss
    # -------------------------------------------------------
    # 정답 라벨은 여전히 In-batch의 대각선(0부터 N-1)에 위치함
    labels = torch.arange(N, device=user_emb.device)
    return F.cross_entropy(logits, labels)
'''
def inbatch_corrected_logq_loss_with_hard_neg(
    user_emb, item_tower_emb, target_ids, user_ids, log_q_tensor,
    batch_hard_neg_ids=None, 
    temperature=0.1, 
    lambda_logq=1.0,         
    hard_lambda_logq=1.0,    # 💡 Force Ratio 정상화를 위해 1.0 권장
    alpha=0.7,               # 💡 사용자가 요청한 알파 가중치 유지
    return_metrics=False     
):
    N = user_emb.size(0)
    
    # 1. In-Batch Logits
    batch_item_emb = item_tower_emb[target_ids] 
    sim_matrix = torch.matmul(user_emb, batch_item_emb.T) 
    logits = sim_matrix / temperature

    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # 마스킹 (Same Item, Same User 제외)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    same_user_mask = torch.eq(user_ids.unsqueeze(1), user_ids.unsqueeze(0))
    diag_mask = torch.eye(N, dtype=torch.bool, device=user_emb.device)
    false_neg_mask = (same_item_mask | same_user_mask) & ~diag_mask
    logits.masked_fill_(false_neg_mask, float('-inf'))

    metrics = {}
    
    # 2. Hard Negative Injection
    if batch_hard_neg_ids is not None:
        # 전체 행렬에서 직접 인덱싱 (속도 중시 롤백)
        hard_neg_emb = item_tower_emb[batch_hard_neg_ids] 
        
        hn_sim_matrix = torch.bmm(user_emb.unsqueeze(1), hard_neg_emb.transpose(1, 2)).squeeze(1)
        
        # 💡 Alpha와 Temperature를 이용해 로짓 계산
        hn_logits = (hn_sim_matrix / temperature) * alpha
        
        # 💡 하드 네거티브에도 인기도 보정을 넣어줘야 Force Ratio가 살아납니다.
        if hard_lambda_logq > 0.0:
            hn_log_q = log_q_tensor[batch_hard_neg_ids]
            hn_logits = hn_logits - (hn_log_q * hard_lambda_logq)
            
        invalid_hn_mask = (batch_hard_neg_ids == 0)
        hn_logits.masked_fill_(invalid_hn_mask, float('-inf'))
        
        # 지표 계산
        if return_metrics:
            with torch.no_grad():
                combined_logits = torch.cat([logits, hn_logits], dim=1)
                probs = torch.softmax(combined_logits, dim=1)
                
                # 유저당 평균 에너지 측정
                inbatch_force = probs[:, :N][~diag_mask & ~false_neg_mask].sum().item() / N
                hn_force = probs[:, N:][~invalid_hn_mask].sum().item() / N
                
                metrics.update({
                    'sim/pos': torch.diag(sim_matrix).mean().item(),
                    'sim/hard_neg': hn_sim_matrix[~invalid_hn_mask].mean().item(),
                    'sim/inbatch_neg': sim_matrix[~diag_mask & ~false_neg_mask].mean().item(),
                    'force/ratio': hn_force / (inbatch_force + hn_force + 1e-8),
                    'hn_active_ratio': (~invalid_hn_mask).float().mean().item()
                })
            
        logits = torch.cat([logits, hn_logits], dim=1)

    loss = F.cross_entropy(logits, torch.arange(N, device=user_emb.device))
    return (loss, metrics) if return_metrics else loss


def inbatch_corrected_logq_loss_with_shared_hard_neg(
    user_emb, item_tower_emb, target_ids, user_ids, log_q_tensor,
    shared_hn_ids=None,      # [128]
    shared_hn_emb=None,      # [128, dim]
    temperature=0.1, 
    lambda_logq=1.0,         
    alpha=1.5,               
    return_metrics=False     
):
    N = user_emb.size(0)
    
    # 1. In-Batch Logits (N x N)
    batch_item_emb = item_tower_emb[target_ids] 
    sim_matrix = torch.matmul(user_emb, batch_item_emb.T) # [N, N]
    logits = sim_matrix / temperature

    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # 마스킹 (자기 자신 및 동일 유저/아이템 제외)
    diag_mask = torch.eye(N, dtype=torch.bool, device=user_emb.device)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    same_user_mask = torch.eq(user_ids.unsqueeze(1), user_ids.unsqueeze(0))
    false_neg_mask = (same_item_mask | same_user_mask) & ~diag_mask
    logits.masked_fill_(false_neg_mask, float('-inf'))

    metrics = {}
    
    # 2. Shared Hard Negative Logits (N x 128)
    if shared_hn_emb is not None:
        hn_logits = torch.matmul(user_emb, shared_hn_emb.T) # [N, 128]
        hn_logits = (hn_logits / temperature) * alpha
        
        if lambda_logq > 0.0:
            hn_log_q = log_q_tensor[shared_hn_ids]
            hn_logits = hn_logits - (hn_log_q.view(1, -1) * lambda_logq)
            
        if return_metrics:
            with torch.no_grad():
                combined_logits = torch.cat([logits, hn_logits], dim=1)
                probs = torch.softmax(combined_logits, dim=1)
                
                # 정답 확률 및 오답 에너지(Force) 계산
                pos_prob = torch.diag(probs[:, :N]).mean().item()
                inbatch_prob_sum = probs[:, :N][~diag_mask & ~false_neg_mask].sum() / N
                hn_prob_sum = probs[:, N:].sum() / N
                
                metrics.update({
                    'sim/pos': torch.diag(sim_matrix).mean().item(),
                    'sim/inbatch_neg': sim_matrix[~diag_mask & ~false_neg_mask].mean().item(),
                    'sim/hard_neg': (torch.matmul(user_emb, shared_hn_emb.T)).mean().item(),
                    'hn_active_ratio': 1.0,  # 배치 공유 방식에선 항상 모든 유저가 혜택을 보므로 1.0
                    'force/ratio': (hn_prob_sum / (inbatch_prob_sum + hn_prob_sum + 1e-8)).item(),
                    'prob/pos': pos_prob  # 모델이 정답을 얼마나 확신하는지 보는 유용한 지표
                })
            
        logits = torch.cat([logits, hn_logits], dim=1)

    # 3. Final Cross Entropy (정답 인덱스는 대각선인 0~N-1)
    labels = torch.arange(N, device=user_emb.device)
    loss = F.cross_entropy(logits, labels)
    
    return (loss, metrics) if return_metrics else loss


import torch
import torch.nn.functional as F
def inbatch_corrected_logq_loss_with_hard_neg_margin(
    user_emb, item_tower_emb, target_ids, user_ids, log_q_tensor,
    last_hard_neg_ids=None,  # [B, num_hn]
    flat_is_last=None,       # [N]
    temperature=0.1, 
    lambda_logq=1.0,         
    alpha=1,               
    margin=0.05,             # 💡 핵심: Positive Margin (정답을 더 꽉 잡게 만듦)
    return_metrics=False     
):
    N = user_emb.size(0)
    device = user_emb.device
    
    # -----------------------------------------------------------
    # 1. In-batch Logits & Positive Margin 적용
    # -----------------------------------------------------------
    batch_item_emb = item_tower_emb[target_ids] 
    sim_matrix = torch.matmul(user_emb, batch_item_emb.T) 
    
    # 💡 [핵심] Positive Margin: 정답(대각선)의 코사인 유사도를 강제로 깎아서 모델을 더 노력하게 만듦
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    sim_matrix[diag_mask] = sim_matrix[diag_mask] - margin
    
    logits = sim_matrix / temperature

    # 인기도 보정 (LogQ)
    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # 마스킹 (동일 아이템, 동일 유저 제외 - 정답 대각선은 살림)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    same_user_mask = torch.eq(user_ids.unsqueeze(1), user_ids.unsqueeze(0))
    false_neg_mask = (same_item_mask | same_user_mask) & ~diag_mask
    logits.masked_fill_(false_neg_mask, float('-inf'))

    metrics = {}
    
    # -----------------------------------------------------------
    # 2. Hard Negative Logits (마지막 스텝 B개만 연산)
    # -----------------------------------------------------------
    if last_hard_neg_ids is not None and flat_is_last is not None:
        B = last_hard_neg_ids.size(0)
        num_hn = last_hard_neg_ids.size(1)
        
        # [B, num_hn, dim]
        hard_neg_emb = item_tower_emb[last_hard_neg_ids] 
        
        # 전체 N개의 유저 임베딩 중 마지막 스텝(B개)만 추출: [B, dim]
        last_user_emb = user_emb[flat_is_last]
        
        # [B, 1, dim] x [B, dim, num_hn] -> [B, num_hn]
        hn_sim = torch.bmm(last_user_emb.unsqueeze(1), hard_neg_emb.transpose(1, 2)).squeeze(1)
        
        # 💡 오답에는 마진을 더하지 않음! 순수 코사인 유사도만 사용
        hn_logits = (hn_sim / temperature) * alpha
        
        if lambda_logq > 0.0:
            hn_log_q = log_q_tensor[last_hard_neg_ids]
            hn_logits = hn_logits - (hn_log_q * lambda_logq)
            
        invalid_hn_mask = (last_hard_neg_ids == 0)
        hn_logits.masked_fill_(invalid_hn_mask, float('-inf'))
        
        # 전체 행렬 N 크기에 맞게 채워 넣기 (마지막 스텝이 아닌 로우는 -inf)
        all_hn_logits = torch.full((N, num_hn), float('-inf'), device=device)
        all_hn_logits[flat_is_last] = hn_logits
        
        # 최종 로짓 병합 [N, N + num_hn]
        logits = torch.cat([logits, all_hn_logits], dim=1)
        # Metrics 계산 (선택 사항)
        if return_metrics:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                
                # 1. 모델 붕괴 확인용: 정답을 맞출 확률
                pos_prob = torch.diag(probs[:, :N]).mean().item()
                
                # 💡 2. [핵심] 하드 네거티브의 상대적 Gradient Power 계산
                # 하드 네거티브가 개입된 '마지막 스텝(B개)' 행만 추출
                probs_last = probs[flat_is_last] # [B, N + num_hn]
                
                # In-batch Negative들의 확률 합 (정답 및 False Negative 제외)
                # diag_mask와 false_neg_mask도 마지막 스텝 B개에 해당하는 행만 가져옵니다.
                valid_inbatch_mask = ~(diag_mask[flat_is_last] | false_neg_mask[flat_is_last])
                inbatch_prob_sum = probs_last[:, :N][valid_inbatch_mask].sum().item() / B
                
                # Hard Negative들의 확률 합
                hn_prob_sum = probs_last[:, N:].sum().item() / B
                
                metrics.update({
                    'sim/pos': (torch.diag(sim_matrix) + margin).mean().item(), # 마진 복구해서 로깅
                    'sim/hard_neg': hn_sim[~invalid_hn_mask].mean().item(),
                    'prob/pos': pos_prob,
                    
                    # 💡 하드 네거티브가 오답들 사이에서 행사하는 지분 (0.0 ~ 1.0)
                    'force/hn_power_ratio': hn_prob_sum / (inbatch_prob_sum + hn_prob_sum + 1e-8),
                    
                    # 유효하게 0번 패딩이 아닌 하드 네거티브의 비율
                    'hn_active_ratio': (~invalid_hn_mask).float().mean().item()
                })

    # -----------------------------------------------------------
    # 3. Final Loss
    # -----------------------------------------------------------
    labels = torch.arange(N, device=device)
    loss = F.cross_entropy(logits, labels)
    
    return (loss, metrics) if return_metrics else loss