import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import random
import numpy as np
import pandas as pd
import torch
import datetime
from torch.utils.data import Dataset



import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

class SASRecDataset_v3(Dataset):
    def __init__(self, processor, global_now_str="2020-09-22", max_len=30, is_train=True):
        self.processor = processor
        self.max_len = max_len
        self.is_train = is_train
        self.user_ids = processor.user_ids
        
        global_now_dt = pd.to_datetime(global_now_str)
        self.now_ordinal = global_now_dt.toordinal()
        self.now_week = global_now_dt.isocalendar().week

    def _shuffle_indices_within_session(self, indices, time_deltas_raw):
        if len(indices) <= 1: return indices
        grouped_indices = []
        current_group = [indices[0]]
        for i in range(1, len(indices)):
            if time_deltas_raw[indices[i]] == 0:
                current_group.append(indices[i])
            else:
                grouped_indices.append(current_group)
                current_group = [indices[i]]
        if current_group: grouped_indices.append(current_group)
        
        shuffled_indices = []
        for group in grouped_indices:
            if len(group) > 1 and random.random() < 0.5:
                random.shuffle(group)
            shuffled_indices.extend(group)
        return shuffled_indices

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        u_mapped_id = self.processor.user2id.get(user_id, 0)
        
        seq_raw = self.processor.u_seqs[u_mapped_id]
        time_deltas_raw = self.processor.u_deltas[u_mapped_id]
        # 1. 전처리: ID 매핑 및 타임 델타 버킷화
        seq_mapped = [self.processor.item2id.get(iid, 0) for iid in seq_raw]
        bins = np.array([0, 3, 7, 14, 30, 60, 180, 330, 395])
        time_buckets = np.digitize(time_deltas_raw, bins, right=False).tolist()
        
        # 2. 인덱스 설정 및 셔플링
        if self.is_train:
            # All-time 학습: 마지막 아이템(정답)을 제외한 인덱스들
            input_indices = list(range(len(seq_raw) - 1))
            shuffled_indices = self._shuffle_indices_within_session(input_indices, time_deltas_raw)
        else:
            shuffled_indices = list(range(len(seq_raw)))

        # 3. max_len 슬라이싱 및 패딩 계산
        shuffled_indices = shuffled_indices[-self.max_len:]
        pad_len = self.max_len - len(shuffled_indices)
        
        # 4. 시퀀스 데이터 추출 (Input, Target, Time Interval)
        input_seq = [seq_mapped[i] for i in shuffled_indices]
        input_time = [time_buckets[i] for i in shuffled_indices]
        
        if self.is_train:
            # i번째 입력의 정답은 항상 i+1번째 아이템 (Shifted Label)
            target_seq = [seq_mapped[i + 1] for i in shuffled_indices]
        else:
            target_seq = []

        # 5. 동적 피처 행렬 추출 및 Target_Now 계산
        d_buckets = self.processor.u_dyn_buckets[u_mapped_id][shuffled_indices] # [len, 3]
        d_conts = self.processor.u_dyn_conts[u_mapped_id][shuffled_indices]     # [len, 4]
        d_cats = self.processor.u_dyn_cats[u_mapped_id][shuffled_indices]       # [len, 1]
        d_time = self.processor.u_dyn_time[u_mapped_id][shuffled_indices]       # [len, 2]

        if self.is_train:
            # 훈련 시: 마지막 정답 아이템의 결제일 기준 (Leakage 방지)
            last_idx = shuffled_indices[-1] + 1
            target_now = self.processor.u_dyn_time[u_mapped_id][last_idx][0]
        else:
            target_now = self.now_ordinal

        # 상대적 거리(Recency Offset) 및 계절성(Week)
        dynamic_offsets = np.clip(target_now - d_time[:, 0], 0, 365).astype(np.int64)
        current_weeks = d_time[:, 1]

        # 6. 패딩 처리 (Left Padding)
        input_padded = [0] * pad_len + input_seq
        time_padded = [0] * pad_len + input_time
        target_padded = [0] * pad_len + target_seq if self.is_train else [0] * self.max_len
        padding_mask = [True] * pad_len + [False] * len(input_seq)

        # 7. 아이템 사이드 정보 추출 (Input Padded 기반)
        item_side_info = self.processor.i_side_arr[input_padded]
        type_ids = item_side_info[:, 0]
        color_ids = item_side_info[:, 1]
        graphic_ids = item_side_info[:, 2]
        section_ids = item_side_info[:, 3]

        # 8. 동적 피처 패딩 처리
        # 0으로 채워진 패딩 행렬 생성
        pad_b = np.zeros((pad_len, 3), dtype=np.int64)
        pad_c = np.zeros((pad_len, 4), dtype=np.float32)
        pad_cat = np.zeros((pad_len, 1), dtype=np.int64)
        pad_1d = np.zeros(pad_len, dtype=np.int64)

        d_buckets_p = np.vstack([pad_b, d_buckets]) if len(shuffled_indices) > 0 else pad_b
        d_conts_p = np.vstack([pad_c, d_conts]) if len(shuffled_indices) > 0 else pad_c
        d_cats_p = np.vstack([pad_cat, d_cats]) if len(shuffled_indices) > 0 else pad_cat
        
        offset_p = np.concatenate([pad_1d, dynamic_offsets]) if len(shuffled_indices) > 0 else pad_1d
        week_p = np.concatenate([pad_1d, current_weeks]) if len(shuffled_indices) > 0 else pad_1d

        # 9. 정적 피처 추출
        s_buckets = self.processor.u_static_buckets[u_mapped_id]
        s_cats = self.processor.u_static_cats[u_mapped_id]

        return {
            'user_ids': user_id,
            # 시퀀스 텐서
            'item_ids': torch.tensor(input_padded, dtype=torch.long),
            'target_ids': torch.tensor(target_padded, dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'time_bucket_ids': torch.tensor(time_padded, dtype=torch.long),
            
            # 아이템 사이드 정보
            'type_ids': torch.tensor(type_ids, dtype=torch.long),
            'color_ids': torch.tensor(color_ids, dtype=torch.long),
            'graphic_ids': torch.tensor(graphic_ids, dtype=torch.long),
            'section_ids': torch.tensor(section_ids, dtype=torch.long),
            
            # 동적 세션 피처
            'price_bucket': torch.tensor(d_buckets_p[:, 0], dtype=torch.long),
            'cnt_bucket': torch.tensor(d_buckets_p[:, 1], dtype=torch.long),
            'recency_bucket': torch.tensor(d_buckets_p[:, 2], dtype=torch.long),
            'cont_feats': torch.tensor(d_conts_p, dtype=torch.float32),
            'channel_ids': torch.tensor(d_cats_p[:, 0], dtype=torch.long),
            
            # 글로벌 컨텍스트 (시간 거리 및 계절)
            'recency_offset': torch.tensor(offset_p, dtype=torch.long),
            'current_week': torch.tensor(week_p, dtype=torch.long),
            
            # 정적 유저 피처
            'age_bucket': torch.tensor(s_buckets[0], dtype=torch.long),
            'club_status_ids': torch.tensor(s_cats[0], dtype=torch.long),
            'news_freq_ids': torch.tensor(s_cats[1], dtype=torch.long),
            'fn_ids': torch.tensor(s_cats[2], dtype=torch.long),
            'active_ids': torch.tensor(s_cats[3], dtype=torch.long),
        }







class SASRecDataset_v2(Dataset):
    def __init__(self, processor, global_now_str="2020-09-22", max_len=30, is_train=True):
        self.processor = processor
        self.max_len = max_len
        self.is_train = is_train
        self.user_ids = processor.user_ids
        
        global_now_dt = pd.to_datetime(global_now_str)
        self.now_ordinal = global_now_dt.toordinal()
        self.now_week = global_now_dt.isocalendar().week

    def _shuffle_within_session(self, seq_raw, time_deltas_raw):
        """동일 세션(delta=0) 아이템들을 그룹화하여 섞는 헬퍼 함수"""
        grouped_indices = []
        current_group = [0]
        
        # 1. delta=0을 기준으로 세션 그룹화
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == 0:
                current_group.append(i)
            else:
                grouped_indices.append(current_group)
                current_group = [i]
        if current_group:
            grouped_indices.append(current_group)
            
        # 2. 그룹 내 셔플 (다양성 확보를 위해 50% 확률로만 섞음)
        shuffled_indices = []
        for group in grouped_indices:
            if len(group) > 1 and random.random() < 0.5:
                random.shuffle(group)
            shuffled_indices.extend(group)
            
        # 3. 섞인 인덱스를 바탕으로 시퀀스 재구성
        shuffled_seq = [seq_raw[i] for i in shuffled_indices]
        # 시간 델타 구조는 원본 유지 (맨 앞만 >0 이고 나머진 0이므로 순서가 무의미함)
        shuffled_deltas = [time_deltas_raw[i] for i in shuffled_indices] 
        
        return shuffled_seq, shuffled_deltas

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        u_mapped_id = self.processor.user2id.get(user_id, 0)
        
        seq_raw = self.processor.seqs.loc[user_id, 'sequence_ids']
        time_deltas_raw = self.processor.seqs.loc[user_id, 'sequence_deltas']
        
        # =========================================================
        # 학습 모드일 때만 Session 단위 Shuffle 적용 (seq_row -> shuffie[seq] -> return seq)
        # =========================================================
        if self.is_train:
            seq_raw, time_deltas_raw = self._shuffle_within_session(seq_raw, time_deltas_raw)
            
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


class ContinuousFeatureTokenizer(nn.Module):
    # (기존과 완전히 동일)
    def __init__(self, num_cont_feats, embed_dim):
        super().__init__()
        self.num_cont_feats = num_cont_feats
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.Tensor(num_cont_feats, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(num_cont_feats, embed_dim))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = x.unsqueeze(-1) 
        tokenized_x = x * self.weight + self.bias
        return tokenized_x


class ContinuousFeatureTokenizer_v3(nn.Module):
    def __init__(self, num_cont_feats, embed_dim):
        super().__init__()
        self.num_cont_feats = num_cont_feats
        self.embed_dim = embed_dim
        
        # 가중치와 편향의 크기는 기존과 동일하게 유지 (피처별로 독립적인 파라미터)
        self.weight = nn.Parameter(torch.Tensor(num_cont_feats, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(num_cont_feats, embed_dim))
        
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        입력 x: (Batch, seq_len, num_cont_feats)
        출력: (Batch, seq_len, num_cont_feats, embed_dim)
        """
        # 1. 입력의 마지막 차원 뒤에 하나를 추가하여 (B, S, F, 1)로 만듦
        x = x.unsqueeze(-1) 
        
        # 2. 가중치와 편향을 (1, 1, F, E) 형태로 view를 조정하여 브로드캐스팅 준비
        # 이렇게 하면 (B, S, F, 1) * (1, 1, F, E) 연산이 수행됩니다.
        # 즉, 시퀀스의 모든 시점(S)에 대해 동일한 피처 가중치가 적용됩니다.
        weight = self.weight.view(1, 1, self.num_cont_feats, self.embed_dim)
        bias = self.bias.view(1, 1, self.num_cont_feats, self.embed_dim)
        
        # 3. 선형 투영 수행
        tokenized_x = x * weight + bias
        
        return tokenized_x

class SASRecUserTower_v2(nn.Module):
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
        num_time_buckets = 12 
        self.time_emb = nn.Embedding(num_time_buckets, self.d_model, padding_idx=0)

        # 💡 [신규] Global Context Embeddings
        # recency_offset: 0~365일 (366개) / current_week: 1~53주차 (54개)
        
        #self.global_recency_emb = nn.Embedding(367, self.d_model)
        self.recency_proj = nn.Linear(1, self.d_model) # (추가) 연속된 숫자를 벡터로!
        self.week_proj = nn.Linear(2, self.d_model)
        self.global_week_emb = nn.Embedding(54, self.d_model)
        
        # ------------------------------------------------------------------
        # [게이트 및 기타 모듈 세팅 - 기존 동일]
        # ------------------------------------------------------------------
        self.seq_gate = nn.Parameter(torch.ones(8)) 
        self.static_gate = nn.Parameter(torch.ones(13))

        self.emb_ln_item = nn.LayerNorm(self.d_model)
        self.emb_dropout_item = nn.Dropout(self.dropout_rate)
        
        self.emb_ln_feat = nn.LayerNorm(self.d_model)
        self.emb_dropout_feat = nn.Dropout(self.dropout_rate)

        self.global_ln = nn.LayerNorm(self.d_model)
        
        # Item Transformer (2층, 4헤드)
        item_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=args.nhead, dim_feedforward=self.d_model * 2,
            dropout=self.dropout_rate, activation='gelu', norm_first=True, batch_first=True
        )
        self.item_transformer = nn.TransformerEncoder(item_layer, num_layers=args.num_layers)

        # Feature Transformer (1층, 1헤드 다이어트)
        feat_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=1, dim_feedforward=self.d_model // 2,
            dropout=self.dropout_rate, activation='gelu', norm_first=True, batch_first=True
        )
        self.feature_transformer = nn.TransformerEncoder(feat_layer, num_layers=1)

        # ==================================================================
        # 2. Static Embeddings & MLP - 기존과 동일 (생략 없이 원형 유지)
        # ==================================================================
        mid_dim, low_dim = 16, 4
        self.age_emb = nn.Embedding(11, mid_dim, padding_idx=0)      
        self.price_emb = nn.Embedding(11, mid_dim, padding_idx=0)    
        self.cnt_emb = nn.Embedding(11, mid_dim, padding_idx=0)      
        self.recency_emb = nn.Embedding(11, mid_dim, padding_idx=0)  

        self.channel_emb = nn.Embedding(4, low_dim, padding_idx=0)   
        self.club_status_emb = nn.Embedding(4, low_dim, padding_idx=0) 
        self.news_freq_emb = nn.Embedding(3, low_dim, padding_idx=0)   
        self.fn_emb = nn.Embedding(3, low_dim, padding_idx=0)        
        self.active_emb = nn.Embedding(3, low_dim, padding_idx=0)    

        self.num_cont_feats = 4
        self.cont_embed_dim = 16 
        self.cont_tokenizer = ContinuousFeatureTokenizer(self.num_cont_feats, self.cont_embed_dim)

        total_static_input_dim = (mid_dim * 4) + (low_dim * 5) + (self.num_cont_feats * self.cont_embed_dim)
        
        self.static_mlp = nn.Sequential(
            nn.Linear(total_static_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.apply(self._init_weights)
        
        # first -> sigmoid gate
        nn.init.xavier_normal_(self.week_proj.weight)
        nn.init.xavier_normal_(self.recency_proj.weight)

        #
        if self.week_proj.bias is not None:
            nn.init.constant_(self.week_proj.bias, 0)
        if self.recency_proj.bias is not None:
            nn.init.constant_(self.recency_proj.bias, 0)
    
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
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    def forward(self, 
                pretrained_vecs, item_ids, time_bucket_ids, 
                type_ids, color_ids, graphic_ids, section_ids,
                age_bucket, price_bucket, cnt_bucket, recency_bucket,
                channel_ids, club_status_ids, news_freq_ids, fn_ids, active_ids,
                cont_feats, 
                recency_offset, current_week, # 💡 [신규 입력]
                padding_mask=None, training_mode=True):
        
        device = item_ids.device
        seq_len = item_ids.size(1)
        batch_size = item_ids.size(0)
        
        s_g_raw = torch.sigmoid(self.seq_gate) 
        u_g_raw = torch.sigmoid(self.static_gate)
        s_g = s_g_raw * torch.ones_like(s_g_raw)
        u_g = u_g_raw * torch.ones_like(u_g_raw)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeddings = self.pos_emb(positions)
        causal_mask = self.get_causal_mask(seq_len, device)

        # -----------------------------------------------------------
        # 💡 [핵심] Global Context 생성 (Batch, 1, d_model)
        # -----------------------------------------------------------

        
        #r_offset_vec = self.global_recency_emb(recency_offset) * s_g[6]
        recency_scaled = (recency_offset.float() / 365.0).unsqueeze(-1) # (Batch, 1)
        r_offset_vec = self.recency_proj(recency_scaled) * s_g[6] # (Batch, d_model)
        
        week_rad = (current_week.float() / 52.0) * (2 * math.pi)
        week_sin = torch.sin(week_rad).unsqueeze(-1) # (Batch, 1)
        week_cos = torch.cos(week_rad).unsqueeze(-1) # (Batch, 1)
        week_cyclical = torch.cat([week_sin, week_cos], dim=-1)
        
        w_vec = self.week_proj(week_cyclical) * s_g[7]
        
        
        #w_vec = self.global_week_emb(current_week) * s_g[7]
        
        # '오늘'의 상태를 정의하는 단일 벡터를 생성하여 시퀀스 축 차원(1)을 추가합니다.
        global_context_raw = (r_offset_vec + w_vec).unsqueeze(1)
        
        # 🔥 3. LayerNorm 통과 (날것의 분산을 정돈)
        global_context = self.global_ln(global_context_raw)

        # -----------------------------------------------------------
        # Phase 1: FDSA - Item ID Stream 
        # -----------------------------------------------------------
        item_emb = self.item_proj(pretrained_vecs) 
        item_emb += self.item_id_emb(item_ids) * s_g[0]
        item_emb += self.time_emb(time_bucket_ids) * s_g[1]
        item_emb += pos_embeddings
        
        # 🔥 Early Injection: 시퀀스 전체를 현재 날짜의 좌표계로 이동
        item_emb = item_emb + global_context
        
        item_emb = self.emb_ln_item(item_emb)
        item_emb = self.emb_dropout_item(item_emb)

        item_output = self.item_transformer(
            item_emb, mask=causal_mask, src_key_padding_mask=padding_mask
        )

        # -----------------------------------------------------------
        # Phase 2: FDSA - Feature Stream 
        # -----------------------------------------------------------
        feat_emb = self.type_emb(type_ids) * s_g[2]
        feat_emb += self.color_emb(color_ids) * s_g[3]
        feat_emb += self.graphic_emb(graphic_ids) * s_g[4]
        feat_emb += self.section_emb(section_ids) * s_g[5]
        feat_emb += pos_embeddings 
        
        # 🔥 Feature Stream에도 계절/시점 주입 (색상, 타입도 계절을 타기 때문)
        feat_emb = feat_emb + global_context
        
        feat_emb = self.emb_ln_feat(feat_emb)
        feat_emb = self.emb_dropout_feat(feat_emb)

        feat_output = self.feature_transformer(
            feat_emb, mask=causal_mask, src_key_padding_mask=padding_mask
        )

        # -----------------------------------------------------------
        # Phase 3: Static Encoding (기존 동일)
        # -----------------------------------------------------------
        emb_age = self.age_emb(age_bucket) * u_g[0]
        emb_price = self.price_emb(price_bucket) * u_g[1]
        emb_cnt = self.cnt_emb(cnt_bucket) * u_g[2]
        emb_rec = self.recency_emb(recency_bucket) * u_g[3]
        
        emb_chan = self.channel_emb(channel_ids) * u_g[4]
        emb_club = self.club_status_emb(club_status_ids) * u_g[5]
        emb_news = self.news_freq_emb(news_freq_ids) * u_g[6]
        emb_fn = self.fn_emb(fn_ids) * u_g[7]
        emb_act = self.active_emb(active_ids) * u_g[8]
        
        tokenized_x = self.cont_tokenizer(cont_feats) 
        cont_gates = u_g[9:13].view(1, 4, 1)
        tokenized_x = tokenized_x * cont_gates
        cont_tokens = F.relu(tokenized_x).view(batch_size, -1)
        
        static_input = torch.cat([
            emb_age, emb_price, emb_cnt, emb_rec,
            emb_chan, emb_club, emb_news, emb_fn, emb_act,
            cont_tokens
        ], dim=1)
        
        user_profile_vec = self.static_mlp(static_input)

        # -----------------------------------------------------------
        # Phase 5: Late Fusion (기존 동일)
        # -----------------------------------------------------------
        if training_mode:
            user_profile_expanded = user_profile_vec.unsqueeze(1).expand(-1, seq_len, -1)
            final_vec = torch.cat([item_output, feat_output, user_profile_expanded], dim=-1)
            final_vec = self.output_proj(final_vec)
            return F.normalize(final_vec, p=2, dim=-1)
        else:
            item_intent_vec = item_output[:, -1, :] 
            feat_intent_vec = feat_output[:, -1, :]
            final_vec = torch.cat([item_intent_vec, feat_intent_vec, user_profile_vec], dim=-1)
            final_vec = self.output_proj(final_vec)
            return F.normalize(final_vec, p=2, dim=-1)
        
        
        
        

class SASRecUserTower_v3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.max_len = args.max_len
        self.dropout_rate = args.dropout
        static_mlp_input_dim = 116
        # ==================================================================
        # 1. Sequence Embeddings (Dynamic: Short-term Intent)
        # ==================================================================
        self.item_proj = nn.Linear(args.pretrained_dim, self.d_model)
        self.item_id_emb = nn.Embedding(args.num_items + 1, self.d_model, padding_idx=0)
        
        self.type_emb = nn.Embedding(args.num_prod_types + 1, self.d_model, padding_idx=0)
        #self.color_emb = nn.Embedding(args.num_colors + 1, self.d_model, padding_idx=0)
        #self.graphic_emb = nn.Embedding(args.num_graphics + 1, self.d_model, padding_idx=0)
        #self.section_emb = nn.Embedding(args.num_sections + 1, self.d_model, padding_idx=0)

        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        num_time_buckets = 12 
        #self.time_emb = nn.Embedding(num_time_buckets, self.d_model, padding_idx=0)

        # 💡 [신규] Global Context Embeddings
        # recency_offset: 0~365일 (366개) / current_week: 1~53주차 (54개)
        
        #self.global_recency_emb = nn.Embedding(367, self.d_model)
        self.recency_proj = nn.Linear(1, self.d_model) # (추가) 연속된 숫자를 벡터로!
        self.week_proj = nn.Linear(2, self.d_model)
        self.global_week_emb = nn.Embedding(54, self.d_model)
        
        # ------------------------------------------------------------------
        # [게이트 및 기타 모듈 세팅 - 기존 동일]
        # ------------------------------------------------------------------
        self.seq_gate = nn.Parameter(torch.ones(8-4)) 
        self.static_gate = nn.Parameter(torch.ones(13-2))

        self.emb_ln_item = nn.LayerNorm(self.d_model)
        self.emb_dropout_item = nn.Dropout(self.dropout_rate)
        
        self.emb_ln_feat = nn.LayerNorm(self.d_model)
        self.emb_dropout_feat = nn.Dropout(self.dropout_rate)

        self.global_ln = nn.LayerNorm(self.d_model)
        
        # Item Transformer (2층, 4헤드)
        item_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=args.nhead, dim_feedforward=self.d_model * 2,
            dropout=self.dropout_rate, activation='gelu', norm_first=True, batch_first=True
        )
        self.item_transformer = nn.TransformerEncoder(item_layer, num_layers=args.num_layers)

        # Feature Transformer (1층, 1헤드 다이어트)
        feat_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=1, dim_feedforward=self.d_model // 2,
            dropout=self.dropout_rate, activation='gelu', norm_first=True, batch_first=True
        )
        self.feature_transformer = nn.TransformerEncoder(feat_layer, num_layers=1)

        # ==================================================================
        # 2. Static Embeddings & MLP - 기존과 동일 (생략 없이 원형 유지)
        # ==================================================================
        mid_dim, low_dim = 16, 4
        self.age_emb = nn.Embedding(11, mid_dim, padding_idx=0)      
        self.price_emb = nn.Embedding(11, mid_dim, padding_idx=0)    
        #self.cnt_emb = nn.Embedding(11, mid_dim, padding_idx=0)      
        #self.recency_emb = nn.Embedding(11, mid_dim, padding_idx=0)  

        self.channel_emb = nn.Embedding(4, low_dim, padding_idx=0)   
        self.club_status_emb = nn.Embedding(4, low_dim, padding_idx=0) 
        self.news_freq_emb = nn.Embedding(3, low_dim, padding_idx=0)   
        self.fn_emb = nn.Embedding(3, low_dim, padding_idx=0)        
        self.active_emb = nn.Embedding(3, low_dim, padding_idx=0)    

        self.num_cont_feats = 4
        self.cont_embed_dim = 16 
        self.cont_tokenizer = ContinuousFeatureTokenizer_v3(self.num_cont_feats, self.cont_embed_dim)

        total_static_input_dim = (mid_dim * 4) + (low_dim * 5) + (self.num_cont_feats * self.cont_embed_dim)
        
        self.static_mlp = nn.Sequential(
            nn.Linear(static_mlp_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.apply(self._init_weights)
        
        # first -> sigmoid gate
        nn.init.xavier_normal_(self.week_proj.weight)
        nn.init.xavier_normal_(self.recency_proj.weight)

        #
        if self.week_proj.bias is not None:
            nn.init.constant_(self.week_proj.bias, 0)
        if self.recency_proj.bias is not None:
            nn.init.constant_(self.recency_proj.bias, 0)
    
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
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    def forward(self, 
                pretrained_vecs, item_ids, time_bucket_ids, 
                type_ids, color_ids, graphic_ids, section_ids,
                age_bucket, price_bucket, cnt_bucket, recency_bucket,
                channel_ids, club_status_ids, news_freq_ids, fn_ids, active_ids,
                cont_feats, 
                recency_offset, current_week, # 💡 [신규 입력]
                padding_mask=None, training_mode=True):
        
        device = item_ids.device
        seq_len = item_ids.size(1)
        batch_size = item_ids.size(0)
        
        s_g_raw = torch.sigmoid(self.seq_gate) 
        u_g_raw = torch.sigmoid(self.static_gate)
        s_g = s_g_raw * torch.ones_like(s_g_raw)
        u_g = u_g_raw * torch.ones_like(u_g_raw)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeddings = self.pos_emb(positions)
        causal_mask = self.get_causal_mask(seq_len, device)

        # -----------------------------------------------------------
        # 💡 [핵심] Global Context 생성 (Batch, 1, d_model)
        # -----------------------------------------------------------

        age_bucket = age_bucket.unsqueeze(1).expand(-1, seq_len)
        club_status_ids = club_status_ids.unsqueeze(1).expand(-1, seq_len)
        news_freq_ids = news_freq_ids.unsqueeze(1).expand(-1, seq_len)
        fn_ids = fn_ids.unsqueeze(1).expand(-1, seq_len)
        active_ids = active_ids.unsqueeze(1).expand(-1, seq_len)
        
        
        # -----------------------------------------------------------
        # Phase 1: Global Context 생성
        # 💡 변경: recency_offset, current_week가 이제 (Batch, seq_len) 이므로 unsqueeze(1) 불필요!
        # -----------------------------------------------------------
        recency_scaled = (recency_offset.float() / 365.0).unsqueeze(-1) # (Batch, seq_len, 1)
        r_offset_vec = self.recency_proj(recency_scaled) * s_g[0] # (Batch, seq_len, d_model)
        
        week_rad = (current_week.float() / 52.0) * (2 * math.pi)
        week_sin = torch.sin(week_rad).unsqueeze(-1) 
        week_cos = torch.cos(week_rad).unsqueeze(-1) 
        week_cyclical = torch.cat([week_sin, week_cos], dim=-1) # (Batch, seq_len, 2)
        
        w_vec = self.week_proj(week_cyclical) * s_g[1] # (Batch, seq_len, d_model)
        
        # 기존의 .unsqueeze(1) 제거 (이미 seq_len 차원이 있음)
        global_context_raw = (r_offset_vec + w_vec) 
        global_context = self.global_ln(global_context_raw)
        
        # -----------------------------------------------------------
        # Phase 1: FDSA - Item ID Stream 
        # -----------------------------------------------------------
        item_emb = self.item_proj(pretrained_vecs) 
        item_emb += self.item_id_emb(item_ids) * s_g[2]

        item_emb += pos_embeddings
        
        # 🔥 Early Injection: 시퀀스 전체를 현재 날짜의 좌표계로 이동
        item_emb = item_emb + global_context
        
        item_emb = self.emb_ln_item(item_emb)
        item_emb = self.emb_dropout_item(item_emb)

        item_output = self.item_transformer(
            item_emb, mask=causal_mask, src_key_padding_mask=padding_mask# 💡 SDPA 엔진에게 Causal 연산임을 직접 알려줌
        )

        # -----------------------------------------------------------
        # Phase 2: FDSA - Feature Stream 
        # -----------------------------------------------------------
        feat_emb = self.type_emb(type_ids) * s_g[3]
        #item_emb += self.time_emb(time_bucket_ids) * s_g[1]
        #feat_emb += self.color_emb(color_ids) * s_g[3]
        #feat_emb += self.graphic_emb(graphic_ids) * s_g[4]
        #feat_emb += self.section_emb(section_ids) * s_g[5]
        feat_emb += pos_embeddings 
        
        # 🔥 Feature Stream에도 계절/시점 주입 (색상, 타입도 계절을 타기 때문)
        feat_emb = feat_emb + global_context
        
        feat_emb = self.emb_ln_feat(feat_emb)
        feat_emb = self.emb_dropout_feat(feat_emb)

        feat_output = self.feature_transformer(
            feat_emb, mask=causal_mask, src_key_padding_mask=padding_mask# 💡 SDPA 엔진에게 Causal 연산임을 직접 알려줌
        )

        # -----------------------------------------------------------
        # Phase 3: Static Encoding (기존 동일)
        # -----------------------------------------------------------
        emb_age = self.age_emb(age_bucket) * u_g[0]
        emb_price = self.price_emb(price_bucket) * u_g[1]
        #emb_cnt = self.cnt_emb(cnt_bucket) * u_g[2]
        #emb_rec = self.recency_emb(recency_bucket) * u_g[3]
        
        emb_chan = self.channel_emb(channel_ids) * u_g[2]
        emb_club = self.club_status_emb(club_status_ids) * u_g[3]
        emb_news = self.news_freq_emb(news_freq_ids) * u_g[4]
        emb_fn = self.fn_emb(fn_ids) * u_g[5]
        emb_act = self.active_emb(active_ids) * u_g[6]
        
        tokenized_x = self.cont_tokenizer(cont_feats) # cont_feats: (Batch, seq_len, 4)
        
        # Gate shape 조정 (Batch, seq_len, 4, 16)에 맞게 브로드캐스팅
        cont_gates = u_g[7:11].view(1, 1, 4, 1) 
        tokenized_x = tokenized_x * cont_gates
        cont_tokens = F.relu(tokenized_x).view(batch_size, seq_len, -1) # (Batch, seq_len, 64)
        
        static_input = torch.cat([
            emb_age, emb_price, #emb_cnt, emb_rec,
            emb_chan, emb_club, emb_news, emb_fn, emb_act,
            cont_tokens
        ], dim=-1) # 차원이 3D이므로 마지막 차원(dim=-1) 결합
        
        user_profile_vec = self.static_mlp(static_input) # 출력: (Batch, seq_len, d_model)

        # -----------------------------------------------------------
        # Phase 5: Late Fusion
        # 💡 변경: user_profile_vec를 확장(expand)할 필요가 없어짐
        # -----------------------------------------------------------
        if training_mode:
            # 기존 user_profile_expanded 코드 삭제, 그대로 concat
            final_vec = torch.cat([item_output, feat_output, user_profile_vec], dim=-1)
            final_vec = self.output_proj(final_vec)
            return F.normalize(final_vec, p=2, dim=-1)
        else:
            item_intent_vec = item_output[:, -1, :] 
            feat_intent_vec = feat_output[:, -1, :]
            # 마지막 스텝의 user_profile 추출
            user_intent_vec = user_profile_vec[:, -1, :]
            
            final_vec = torch.cat([item_intent_vec, feat_intent_vec, user_intent_vec], dim=-1)
            final_vec = self.output_proj(final_vec)
            return F.normalize(final_vec, p=2, dim=-1)
        
        
        
'''      
        
        

def train_user_tower_all_time(epoch, model, item_tower, log_q_tensor, dataloader, optimizer, scaler, cfg, device, hard_neg_pool_tensor,scheduler,seq_labels=None, static_labels=None):
    """단일 에포크 훈련 함수 (All Time Steps + Same-User Masking 적용)"""
    model.train()
    total_loss_accum = 0.0
    main_loss_accum = 0.0
    cl_loss_accum = 0.0
    
    # decay aux loss
    
    
    
    # 안전을 위해 labels가 None일 경우 빈 리스트로 초기화
    seq_labels = seq_labels or []
    static_labels = static_labels or []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # -------------------------------------------------------
        # 1. Data Unpacking
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

        # -------------------------------------------------------
        # 2. Forward & Loss Calculation (AMP)
        # -------------------------------------------------------
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            output_1 = model(**forward_kwargs)
            output_2 = model(**forward_kwargs)
            # =======================================================
            # 💡 (1) Main Loss (VRAM Diet & Last-Step Hard Negative)
            # =======================================================
            valid_mask = ~padding_mask 
            batch_size, seq_len = item_ids.shape
            
            # 💡 [핵심 수정] 엉터리 공식 버리기! 
            # 패딩 위치와 무관하게 '마지막 True'의 인덱스를 정확하게 추적하는 공식
            seq_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            # 유효하지 않은 곳은 -1로 밀어버리고, 가장 큰 인덱스(마지막 유효 아이템 위치)를 뽑음
            last_indices = torch.max(seq_positions.masked_fill(~valid_mask, -1), dim=1)[0]
            last_indices = last_indices.clamp(min=0) # 완전 빈 시퀀스 방어
            
            batch_range = torch.arange(batch_size, device=device)
            is_last_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            is_last_mask[batch_range, last_indices] = True
            
            # 2. 1D Flattening
            flat_output = output_1[valid_mask] 
            flat_targets = target_ids[valid_mask]
            flat_is_last = is_last_mask[valid_mask] # 💡 이제 정상적으로 768개의 True가 살아남음!
            
            batch_row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
            flat_user_ids = batch_row_indices[valid_mask] 
            
            if flat_output.size(0) > 0:
                flat_user_emb = F.normalize(flat_output, p=2, dim=1)
                last_hard_neg_ids = None # 하드 네거티브 없음!
                # 정상적으로 768명(batch_size) 전체에 대해 하드 네거티브 10개씩 샘플링
                #last_target_ids = target_ids[batch_range, last_indices] # [768]
                #num_hn = 15
                #batch_hn_pool = hard_neg_pool_tensor[last_target_ids]   # [768, 50]
                
                #rand_indices = torch.randint(0, 50, (batch_size, num_hn), device=device)
                #last_hard_neg_ids = torch.gather(batch_hn_pool, 1, rand_indices) # [768, 10]

                all_item_emb = item_tower.get_all_embeddings()
                norm_item_embeddings = F.normalize(all_item_emb, p=2, dim=1)
                
                # 이후 Loss 함수 호출은 동일하게 진행
                main_loss, b_metrics = inbatch_corrected_logq_loss_with_hard_neg_margin(
                    user_emb=flat_user_emb,
                    item_tower_emb=norm_item_embeddings,
                    target_ids=flat_targets,
                    user_ids=flat_user_ids,
                    log_q_tensor=log_q_tensor,
                    last_hard_neg_ids=last_hard_neg_ids, 
                    flat_is_last=flat_is_last,           
                    temperature=0.1, 
                    lambda_logq=cfg.lambda_logq,
                    margin=0.00, 
                    alpha=1,
                    return_metrics=True 
                )
            else:
                main_loss = torch.tensor(0.0, device=device)
            
            # =======================================================
            # (2) DuoRec Loss (여전히 Last Time Step Only 적용)
            # =======================================================
            # DuoRec은 시퀀스의 '최종 의도' 안정화에 목적이 있으므로 마지막 스텝만 사용하는 것이 맞습니다.
            last_output_1 = output_1[batch_range, last_indices]
            last_output_2 = output_2[batch_range, last_indices]
            last_targets = target_ids[batch_range, last_indices]
            
            #cl_loss = duorec_loss_refined(
            #        user_emb_1=last_output_1,
            #        user_emb_2=last_output_2,
            #        target_ids=last_targets,
            #        lambda_sup=cfg.lambda_sup
            #)

            # 최종 Loss 조합
            total_loss = main_loss #+ (cfg.lambda_cl * cl_loss)

        # -------------------------------------------------------
        # 3. Backward & Optimizer Step
        # -------------------------------------------------------
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # 💡 OneCycleLR은 매 배치마다 호출

        # 누적 및 로깅
        total_loss_accum += total_loss.item()
        main_loss_accum += main_loss.item()
        #cl_loss_accum += cl_loss.item()
        
        pbar.set_postfix({
            'Loss': f"{total_loss.item():.4f}",
            'Main': f"{main_loss.item():.4f}",
            #'CL': f"{cl_loss.item():.4f}"
        })
        
        if batch_idx % 100 == 0:
            # b_metrics 딕셔너리에서 안전하게 지표 추출
            wandb.log({
                # 1. 메인 학습 지표
                "Train/Main_Loss": main_loss.item(),
                #"Train/CL_Loss" : cl_loss.item(),
                # 2. 임베딩 유사도 (거리) 지표
                #"Sim/Pos": b_metrics.get('sim/pos', 0.0),
                #"Sim/Hard_Neg": b_metrics.get('sim/hard_neg', 0.0),
                
                # 3. 하드 네거티브 파워 및 상태 지표
                #"Force/Hard_Neg_Ratio": b_metrics.get('force/hn_power_ratio', 0.0),
                #"Sim/HN_Active_Ratio": b_metrics.get('hn_active_ratio', 0.0),
                
                # 4. 모델 붕괴 체크용 지표 (정답 확신도)
                #"Prob/Pos_Confidence": b_metrics.get('prob/pos', 0.0)
            })

    avg_loss = total_loss_accum / len(dataloader)
    avg_main = main_loss_accum / len(dataloader)
    avg_cl = cl_loss_accum / len(dataloader)

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
'''