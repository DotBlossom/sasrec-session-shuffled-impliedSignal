import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
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

        if self.is_train: # 훈련용 데이터셋일 때만 딱 한 번 로그를 보고 싶다면!
            self.verify_session_logic()
            
    def verify_session_logic(self):
        """디버깅용: 세션 분리 및 셔플링이 정상적으로 작동하는지 1명의 유저를 뽑아 콘솔에 출력합니다."""
        print("\n" + "="*75)
        print("🕵️‍♂️ [Dataset Monitor] Session Grouping & Shuffling Verification")
        print("="*75)
        
        # 시퀀스 길이가 5 이상인 유저 1명 찾기
        sample_user = None
        for uid in self.user_ids:
            u_mapped_id = self.processor.user2id.get(uid, 0)
            if len(self.processor.u_seqs[u_mapped_id]) > 5:
                sample_user = uid
                break
                
        if not sample_user:
            print("충분한 시퀀스를 가진 유저가 없습니다.")
            return
            
        u_mapped_id = self.processor.user2id[sample_user]
        seq_raw = self.processor.u_seqs[u_mapped_id]
        time_deltas_raw = self.processor.u_deltas[u_mapped_id]
        
        # 1. 세션 ID 생성 (수정된 로직)
        session_ids_raw = [1]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == time_deltas_raw[i-1]:
                session_ids_raw.append(session_ids_raw[-1])
            else:
                session_ids_raw.append(session_ids_raw[-1] + 1)
                
        # 2. 셔플링 시뮬레이션 (수정된 로직 적용, 무조건 섞이도록 강제)
        input_indices = list(range(len(seq_raw)))
        grouped_indices = []
        current_group = [input_indices[0]]
        for i in range(1, len(input_indices)):
            if time_deltas_raw[input_indices[i]] == time_deltas_raw[input_indices[i-1]]:
                current_group.append(input_indices[i])
            else:
                grouped_indices.append(current_group)
                current_group = [input_indices[i]]
        if current_group: grouped_indices.append(current_group)
        
        shuffled_indices = []
        for group in grouped_indices:
            group_copy = group.copy()
            if len(group_copy) > 1:
                random.shuffle(group_copy) # 테스트용이므로 100% 섞음
            shuffled_indices.extend(group_copy)
            
        print(f"👤 Sample User ID: {sample_user}")
        print(f"{'Orig_Idx':<9} | {'Item_ID':<11} | {'Delta (Days)':<13} | {'Session_ID':<11} | {'Shuffled_Idx':<13}")
        print("-" * 75)
        
        for i in range(len(seq_raw)):
            orig_idx = i
            shuff_idx = shuffled_indices[i] # 셔플된 결과가 위치할 인덱스
            item_id = seq_raw[shuff_idx]
            delta = time_deltas_raw[shuff_idx]
            sess_id = session_ids_raw[shuff_idx]
            
            # 같은 세션(Session ID)끼리 묶여있는지, Orig_Idx가 섞였는지 확인
            print(f"{orig_idx:<9} | {item_id:<11} | {delta:<13} | {sess_id:<11} | {shuff_idx:<13}")
        print("="*75 + "\n")
    def _shuffle_indices_within_session(self, indices, time_deltas_raw):
        if len(indices) <= 1: return indices
        grouped_indices = []
        current_group = [indices[0]]
        
        for i in range(1, len(indices)):
            idx = indices[i]
            prev_idx = indices[i-1] # 💡 추가
            
            # 💡 [핵심 버그 수정] 0인지 묻는 게 아니라, 이전 아이템과 남은 일수(Delta)가 같은지 확인!
            if time_deltas_raw[idx] == time_deltas_raw[prev_idx]:
                current_group.append(idx)
            else:
                grouped_indices.append(current_group)
                current_group = [idx]
                
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

        session_ids_raw = [1]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == time_deltas_raw[i-1]:
                session_ids_raw.append(session_ids_raw[-1]) # 델타가 같으면 같은 당일(세션)!
            else:
                session_ids_raw.append(session_ids_raw[-1] + 1) # 델타가 달라지면 새 세션 시작
        
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
        # 💡 [적용] 셔플된 순서에 맞춰 session_id도 재배열 (당일 아이템은 섞여도 같은 번호!)
        input_session = [session_ids_raw[i] for i in shuffled_indices]
        
        if self.is_train:
            target_seq = [seq_mapped[i + 1] for i in shuffled_indices]
        else:
            target_seq = []
            

        # 5. 동적 피처 행렬 추출 및 Target_Now 계산
        d_buckets = self.processor.u_dyn_buckets[u_mapped_id][shuffled_indices] # [len, 3]
        d_conts = self.processor.u_dyn_conts[u_mapped_id][shuffled_indices]     # [len, 4]
        d_cats = self.processor.u_dyn_cats[u_mapped_id][shuffled_indices]       # [len, 1]
        d_time = self.processor.u_dyn_time[u_mapped_id][shuffled_indices]       # [len, 2]

        input_dates = d_time[:, 0].tolist()
        
        if self.is_train:
            target_indices = [idx + 1 for idx in shuffled_indices]
            step_target_times = self.processor.u_dyn_time[u_mapped_id][target_indices, 0]
            # 💡 [신규] 바로 다음 타겟(정답)의 실제 주차(Week) 추출
            step_target_weeks = self.processor.u_dyn_time[u_mapped_id][target_indices, 1] 
            
            dynamic_offsets = np.clip(step_target_times - d_time[:, 0], 0, 365).astype(np.int64)
        else:
            dynamic_offsets = np.clip(self.now_ordinal - d_time[:, 0], 0, 365).astype(np.int64)
            # 💡 [신규] 추론 시에는 글로벌 타겟 주차(now_week)로 통일하여 예측 방향 강제
            step_target_weeks = np.array([self.now_week] * len(shuffled_indices))
        current_weeks = d_time[:, 1]

        # 6. 패딩 처리 (Left Padding)
        input_padded = [0] * pad_len + input_seq
        time_padded = [0] * pad_len + input_time
        target_padded = [0] * pad_len + target_seq if self.is_train else [0] * self.max_len
        padding_mask = [True] * pad_len + [False] * len(input_seq)
        session_padded = [0] * pad_len + input_session
        target_week_padded = [0] * pad_len + step_target_weeks.tolist()
        dates_padded = [0] * pad_len + input_dates
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
            'session_ids': torch.tensor(session_padded, dtype=torch.long),
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
            'target_week': torch.tensor(target_week_padded, dtype=torch.long),
            
            # 정적 유저 피처
            'age_bucket': torch.tensor(s_buckets[0], dtype=torch.long),
            'club_status_ids': torch.tensor(s_cats[0], dtype=torch.long),
            'news_freq_ids': torch.tensor(s_cats[1], dtype=torch.long),
            'fn_ids': torch.tensor(s_cats[2], dtype=torch.long),
            'active_ids': torch.tensor(s_cats[3], dtype=torch.long),

            'interaction_dates': torch.tensor(dates_padded, dtype=torch.long),
        }

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
        self.seq_gate = nn.Parameter(torch.ones(9-4)) 
        self.static_gate = nn.Parameter(torch.ones(13-2))

        self.emb_ln_item = nn.LayerNorm(self.d_model)
        self.emb_dropout_item = nn.Dropout(self.dropout_rate)
        
        self.emb_ln_feat = nn.LayerNorm(self.d_model)
        self.emb_dropout_feat = nn.Dropout(self.dropout_rate)

        self.global_ln = nn.LayerNorm(self.d_model)
        
        # Item Transformer (2층, 4헤드)
        item_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model , nhead=args.nhead , dim_feedforward=self.d_model * 2,
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
                recency_offset, current_week,target_week=None, # 💡 [신규 입력]
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
        if target_week is not None:
            # target_week: (Batch, seq_len)
            t_week_rad = (target_week.float() / 52.0) * (2 * math.pi)
            t_week_sin = torch.sin(t_week_rad).unsqueeze(-1) 
            t_week_cos = torch.cos(t_week_rad).unsqueeze(-1) 
            t_week_cyclical = torch.cat([t_week_sin, t_week_cos], dim=-1) # (Batch, seq_len, 2)
            
            # 🔥 핵심: current_week와 동일한 week_proj 가중치를 공유하여 투영!
            # 동일한 주차면 완전히 동일한 벡터 방향성을 갖게 됩니다.
            target_week_vec = self.week_proj(t_week_cyclical)* s_g[4] # (Batch, seq_len, d_model)

        # -----------------------------------------------------------
        # Phase 5: Late Fusion
        # 💡 변경: user_profile_vec를 확장(expand)할 필요가 없어짐
        # -----------------------------------------------------------
        if training_mode:
            final_vec = torch.cat([item_output, feat_output , user_profile_vec], dim=-1)
            final_vec = self.output_proj(final_vec)
            
            # 💡 [신규] 훈련 시 전체 시퀀스에 타겟 주차(계절감) 방향성 주입
            if target_week is not None:
                final_vec = final_vec + target_week_vec
                
            return F.normalize(final_vec, p=2, dim=-1)
        else:
            item_intent_vec = item_output[:, -1, :] 
            feat_intent_vec = feat_output[:, -1, :]
            user_intent_vec = user_profile_vec[:, -1, :]
            
            final_vec = torch.cat([item_intent_vec, feat_intent_vec ,user_intent_vec], dim=-1)
            final_vec = self.output_proj(final_vec)
            
            # 💡 [신규] 추론 시 마지막 스텝의 타겟 주차(계절감) 방향성 주입
            if target_week is not None:
                t_week_intent = target_week_vec[:, -1, :] # 마지막 스텝의 타겟 벡터
                final_vec = final_vec + t_week_intent
                
            return F.normalize(final_vec, p=2, dim=-1)
        
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



def inbatch_corrected_logq_loss_with_dynamic_soft_labels(
    user_emb, item_tower_emb, target_ids, user_ids, log_q_tensor,
    last_hard_neg_ids=None,  # [B, num_hn]
    flat_is_last=None,       # [N]
    temperature=0.1, 
    lambda_logq=1.0,         
    alpha=1,               
    margin=0.05,             
    return_metrics=True,
    # 💡 [신규] 동적 라벨 스무딩 파라미터
    K_peers=3, # 이 유사도 이상인 유저들만 '취향 동기화'로 인정
    max_smoothing_mass=0.2,     # Soft Positive에 나눠줄 최대 확률 총합 (내 진짜 정답은 최소 0.8 보장)
    tau_soft=0.1
):
    N = user_emb.size(0)
    device = user_emb.device
    metrics = {}
    # -----------------------------------------------------------
    # 0. Similarity Matrix 생성 (Raw Score 보존)
    # -----------------------------------------------------------
    batch_item_emb = item_tower_emb[target_ids] 
    # [N, N] 행렬: raw_sim은 마진이나 온도가 적용되지 않은 순수 유사도
    raw_sim = torch.matmul(user_emb, batch_item_emb.T) 
    
    # 💡 [메트릭 계산 1] 진짜 정답(Positive)의 유사도 평균
    if return_metrics:
        with torch.no_grad():
            metrics['sim/pos'] = torch.diag(raw_sim).mean().item()
            
    # -----------------------------------------------------------
    # 1. In-batch Logits & Positive Margin 적용 (기존 동일)
    # -----------------------------------------------------------
    sim_matrix = raw_sim.clone()
    
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    sim_matrix[diag_mask] = sim_matrix[diag_mask] - margin
    
    logits = sim_matrix / temperature

    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # 마스킹 방어막
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    same_user_mask = torch.eq(user_ids.unsqueeze(1), user_ids.unsqueeze(0))
    false_neg_mask = (same_item_mask | same_user_mask) & ~diag_mask
    logits.masked_fill_(false_neg_mask, float('-inf'))

    # (중간 Hard Negative 관련 로직 생략 - 기존 코드와 완벽히 동일하게 유지)
    # ... [기존 Hard Negative 로직 병합 코드] ...
    # 편의상 여기서는 in-batch 행렬(N x N)에 대해서만 타겟을 만든다고 가정합니다.
    # 만약 num_hn이 추가되어 logits가 [N, N + num_hn]이 되었다면, 
    # targets 행렬도 torch.zeros_like(logits)로 생성하면 됩니다.
    
    # -----------------------------------------------------------
    # 🚀 3. [핵심] 유저 인텐트 유사도 기반 Soft Positive Target 생성
    # -----------------------------------------------------------

    # 3-1. 유저 간 코사인 유사도 계산
    user_emb_norm = F.normalize(user_emb, p=2, dim=-1)
    intent_sim = torch.matmul(user_emb_norm, user_emb_norm.T)
    
    # 3-2. 안전장치: 나 자신(대각선)과, 내 세션에 있던 아이템을 고른 유저는 후보에서 제외
    invalid_peer_mask = diag_mask | false_neg_mask
    intent_sim.masked_fill_(invalid_peer_mask, float('-inf'))
    
    # 3-3. Top-K 추출: 가장 유사도 높은 K명의 점수와 위치를 뽑아냄
    # topk_sim: [N, K], topk_idx: [N, K]
    topk_sim, topk_idx = torch.topk(intent_sim, k=K_peers, dim=1)
    

    if return_metrics:
        with torch.no_grad():
            # raw_sim에서 동료들이 고른 아이템의 위치 점수를 추출
            peer_item_sims = torch.gather(raw_sim, 1, topk_idx)
            # -inf인 경우는 유효한 동료가 없는 것이므로 필터링하여 평균 계산
            valid_peer_sims = peer_item_sims[topk_sim > float('-inf')]
            if valid_peer_sims.numel() > 0:
                metrics['sim/soft_pos'] = valid_peer_sims.mean().item()
            else:
                metrics['sim/soft_pos'] = 0.0
    
    
    valid_peer_mask = topk_sim > -1e8 
    
    # 3-4. 안전한 Softmax 계산
    # 일단 0으로 꽉 찬 텐서를 준비합니다.
    peer_weights = torch.zeros_like(topk_sim)
    
    # 유효한 동료가 '최소 1명 이상' 존재하는 유저(Row)들만 찾아냅니다.
    has_valid_peers = valid_peer_mask.any(dim=1)
    
    if has_valid_peers.any():
        # 전부 -inf가 아닌, 안전한 행(Row)에 대해서만 Softmax를 계산합니다.
        safe_sims = topk_sim[has_valid_peers] / tau_soft
        peer_weights[has_valid_peers] = F.softmax(safe_sims, dim=1) * max_smoothing_mass
        
        # K명 중 정상 1명, 더미 2명이 섞여 있을 경우를 대비해, 더미의 확률을 확실하게 0으로 날려버림
        peer_weights = peer_weights * valid_peer_mask.float()
    
    
    # 3-5. 타겟 텐서(targets)에 할당
    targets = torch.zeros_like(logits)
    targets.scatter_add_(1, topk_idx, peer_weights)
    
    # 3-6. 진짜 정답 대각선 채우기 (나머지 질량)
    true_target_probs = 1.0 - targets.sum(dim=1)
    targets[diag_mask] = true_target_probs
    
    # -----------------------------------------------------------
    # 4. Final Loss: 확률 분포(Soft Targets)를 활용한 Cross Entropy
    # -----------------------------------------------------------
    # PyTorch 1.10 이상부터는 labels 인자에 1D index 대신 2D 확률 분포를 바로 넣을 수 있습니다.
    # 수식: Loss = - Σ (target_prob * log(pred_prob))
    loss = F.cross_entropy(logits, targets)
    if return_metrics:
        metrics['loss/inbatch'] = loss.item()
        # 정답에 준 확률 평균 (Confidence)
        metrics['prob/true_pos'] = true_target_probs.mean().item()
    
    return (loss, metrics) if return_metrics else loss








def create_category_mapping_tensor(json_path, processor, device):
    """
    JSON에서 product_type_name을 추출하여 아이템 모델 인덱스(1~N)에 매핑되는
    1D 카테고리 텐서를 생성합니다. (0번 인덱스는 패딩용으로 0값 유지)
    """
    with open(json_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
        
    # 💡 수정 1: FeatureProcessor_v3는 1-based index(0은 패딩)를 사용하므로 +1 크기로 생성
    num_items_total = processor.num_items + 1
    category_tensor = torch.zeros(num_items_total, dtype=torch.long)
    
    # 카테고리 ID도 0을 패딩(Unknown)으로 취급하고 1부터 시작하도록 구성
    cat_str_to_id = {}
    
    for item in data:
        # 💡 수정 2: processor 내부에서 str 타입으로 인덱싱되어 있으므로 형변환 필수
        pid = str(item['article_id'])
        
        # 💡 수정 3: processor.item_id_map 대신 processor.item2id 사용
        if pid in processor.item2id:
            idx = processor.item2id[pid]
            cat_name = item['product_type_name']
            
            if cat_name not in cat_str_to_id:
                # 카테고리 ID를 1부터 순차적으로 부여
                cat_str_to_id[cat_name] = len(cat_str_to_id) + 1
                
            category_tensor[idx] = cat_str_to_id[cat_name]
            
    print(f"📦 Category mapping created. Unique categories: {len(cat_str_to_id)}")
    return category_tensor.to(device)
def mine_category_constrained_hard_negatives(item_embs, category_tensor, k=5, device='cuda'):
    """
    에포크 시작 시 호출. 동일 카테고리 내에서 Top-K 하드 네거티브를 추출합니다.
    """
    num_items = item_embs.size(0)
    # 초기화: 기본값은 자기 자신(또는 0)으로 채우되, 이후 로직에서 덮어씌움
    hard_neg_pool = torch.zeros((num_items, k), dtype=torch.long, device=device)
    
    unique_cats = torch.unique(category_tensor)
    
    for cat in unique_cats:
        # 1. 해당 카테고리에 속한 아이템 인덱스 추출
        cat_mask = (category_tensor == cat)
        cat_indices = torch.nonzero(cat_mask).squeeze(1) # [N_cat]
        
        # 카테고리 내 아이템이 너무 적으면 마이닝 생략 (자기 자신 복제)
        if len(cat_indices) <= 1:
            hard_neg_pool[cat_indices] = cat_indices.unsqueeze(1).expand(-1, k)
            continue
            
        # 2. 카테고리 내 아이템 임베딩 추출 및 유사도 계산
        cat_embs = item_embs[cat_indices] # [N_cat, dim]
        sim_matrix = torch.matmul(cat_embs, cat_embs.T) # [N_cat, N_cat]
        
        # 3. 마스킹: 자기 자신 및 너무 똑같은 복제품(유사도 0.99 이상) 제외
        sim_matrix.fill_diagonal_(-float('inf'))
        sim_matrix.masked_fill_(sim_matrix >= 0.99, -float('inf'))
        
        # 4. Top-K 추출
        actual_k = min(k, len(cat_indices) - 1)
        _, topk_idx_local = torch.topk(sim_matrix, actual_k, dim=1)
        
        # 5. 로컬 인덱스를 전체 글로벌 아이템 인덱스로 복원
        topk_idx_global = cat_indices[topk_idx_local]
        
        # 만약 카테고리 내 아이템 수가 K개보다 적다면 부족한 만큼 패딩(첫 번째 아이템 복제)
        if actual_k < k:
            pad_idx = topk_idx_global[:, [0]].expand(-1, k - actual_k)
            topk_idx_global = torch.cat([topk_idx_global, pad_idx], dim=1)
            
        # 최종 풀에 업데이트
        hard_neg_pool[cat_indices] = topk_idx_global
        
    return hard_neg_pool        



def inbatch_corrected_logq_loss_with_hybrid_hard_neg_prev(
    user_emb, item_tower_emb, target_ids, user_ids, log_q_tensor,
    batch_hard_neg_ids=None,
    flat_history_item_ids=None,
    step_weights=None,
    temperature=0.1, 
    lambda_logq=1.0,          
    alpha=1.0,                
    margin=0.00,
    soft_penalty_weight=5.0, # 💡 추가됨: SimANS 스타일의 페널티 강도 조절 (보통 2.0 ~ 10.0 사이 튜닝)
    return_metrics=False     
):
    N = user_emb.size(0)
    device = user_emb.device
    SAFE_NEG_INF = -1e9
    
    # -----------------------------------------------------------
    # 1. In-batch Logits 계산 (기존과 동일)
    # -----------------------------------------------------------
    batch_item_emb = item_tower_emb[target_ids] 
    sim_matrix = torch.matmul(user_emb, batch_item_emb.T) 
    pos_sim = torch.diagonal(sim_matrix) # [N]
    labels = torch.arange(N, device=device)
    logits = sim_matrix / temperature

    if margin > 0.0:
        logits[labels, labels] -= (margin / temperature)

    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    same_user_mask = torch.eq(user_ids.unsqueeze(1), user_ids.unsqueeze(0))
    false_neg_mask = (same_item_mask | same_user_mask) & ~diag_mask
    logits = logits.masked_fill(false_neg_mask, SAFE_NEG_INF)
    
    metrics = {}
    
    # -----------------------------------------------------------
    # 2. Hard Negative Processing (하이브리드 로직 적용)
    # -----------------------------------------------------------
    if batch_hard_neg_ids is not None:
        hn_emb = item_tower_emb[batch_hard_neg_ids] 
        hn_sim = torch.bmm(user_emb.unsqueeze(1), hn_emb.transpose(1, 2)).squeeze(1) 
        
        # [A] 명확한 False Negative 마스킹 (구매 이력 등)
        absolute_fn_mask = torch.zeros_like(hn_sim, dtype=torch.bool, device=device)
        if flat_history_item_ids is not None:
            absolute_fn_mask = (batch_hard_neg_ids.unsqueeze(2) == flat_history_item_ids.unsqueeze(1)).any(dim=2)

        # [B] 모호한 샘플(Ambiguous Negatives)에 대한 Soft-Weighting 페널티 계산
        # threshold(0.95 * pos_sim)를 넘는 샘플은 날리지 않고, 초과한 만큼 페널티 부여
        threshold = 0.95 * pos_sim.unsqueeze(1)
        
        # threshold 초과분 계산 (ReLU를 통해 threshold 미만은 0으로 처리)
        excess_sim = torch.relu(hn_sim - threshold) 
        
        # 초과분에 가중치를 곱해 페널티 산출 (온도로 나누어 scale 맞춤)
        ambiguity_penalty = (excess_sim * soft_penalty_weight) / temperature

        if return_metrics:
            # 페널티를 받은(0.95를 넘은) 샘플의 비율
            penalized_mask = excess_sim > 0
            metrics['hn/penalized_ratio'] = penalized_mask.float().mean().item()
            metrics['sim/hn_all'] = hn_sim.mean().item() 
            metrics['sim/hn_penalized'] = hn_sim[penalized_mask].mean().item() if penalized_mask.any() else 0.0
        
        hn_logits = (hn_sim / temperature) * alpha
        
        if lambda_logq > 0.0:
            hn_log_q = log_q_tensor[batch_hard_neg_ids]
            hn_logits = hn_logits - (hn_log_q * lambda_logq)
            
        # [C] 페널티 차감 및 절대적 FN 배제
        # 1. 0.95를 넘는 모호한 샘플들의 Logit을 부드럽게 깎아냄 (Gradient 생존)
        hn_logits = hn_logits - ambiguity_penalty 
        
        # 2. 구매 이력 등 절대적 FN은 확실히 배제
        hn_logits = hn_logits.masked_fill(absolute_fn_mask, SAFE_NEG_INF)
        
        logits = torch.cat([logits, hn_logits], dim=1)

    logits = torch.clamp(logits, min=SAFE_NEG_INF, max=1e4)

    # -----------------------------------------------------------
    # 3. Loss 계산 (가중치 여부 반영)
    # -----------------------------------------------------------
    if step_weights is not None:
        step_weights = torch.as_tensor(step_weights, device=device, dtype=torch.float32)
        loss_unreduced = F.cross_entropy(logits, labels, reduction='none')
        loss = (loss_unreduced * step_weights).sum() / (step_weights.sum() + 1e-9)
        if return_metrics:
            metrics['sim/pos'] = ((pos_sim * step_weights).sum() / (step_weights.sum() + 1e-9)).item()
    else:
        loss = F.cross_entropy(logits, labels)
        if return_metrics:
            metrics['sim/pos'] = pos_sim.mean().item()

    if return_metrics:
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            if batch_hard_neg_ids is not None:
                hn_probs_sum = probs[:, N:].sum(dim=1)
                metrics['hn/influence_ratio'] = hn_probs_sum.mean().item()
                # 💡 (추가 코드) 순수 Negative들 사이에서 HN이 차지하는 비중 계산
                # 1.0(전체)에서 정답 확률(probs[:, 0])을 뺀 것이 전체 Negative 파이
                neg_probs_total = 1.0 - probs.diagonal() 
                relative_hn_ratio = hn_probs_sum / (neg_probs_total + 1e-9)
                metrics['hn/relative_influence'] = relative_hn_ratio.mean().item()
            else:
                metrics['hn/influence_ratio'] = 0.0
        return loss, metrics

    return loss


def inbatch_corrected_logq_loss_with_hybrid_hard_neg(
    user_emb, seq_item_emb, target_ids, user_ids, log_q_tensor,
    hn_item_emb=None, batch_hard_neg_ids=None,
    flat_history_item_ids=None,
    step_weights=None,
    final_idx=None, # 💡 [핵심 파라미터] 외부에서 계산된 마지막 스텝의 위치 정보
    temperature=0.1, 
    lambda_logq=1.0,          
    alpha=1.0,                
    margin=0.00,
    soft_penalty_weight=5.0,
    return_metrics=False     
):
    N = user_emb.size(0)
    device = user_emb.device
    SAFE_NEG_INF = -1e9
    
    # 가중치 텐서 준비
    if step_weights is not None:
        step_weights = torch.as_tensor(step_weights, device=device, dtype=torch.float32)
        weight_sum = step_weights.sum() + 1e-9
        sw_unsqueezed = step_weights.unsqueeze(1)
    else:
        weight_sum = None
        sw_unsqueezed = None

    # -----------------------------------------------------------
    # 1. In-batch Logits 계산 (전체 N개 스텝 대상)
    # -----------------------------------------------------------
    sim_matrix = torch.matmul(user_emb, seq_item_emb.T) 
    pos_sim = torch.diagonal(sim_matrix) # [N]
    labels = torch.arange(N, device=device)
    logits = sim_matrix / temperature

    if margin > 0.0:
        logits[labels, labels] -= (margin / temperature)

    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # In-batch 내 가짜 네거티브(False Negative) 마스킹
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    same_user_mask = torch.eq(user_ids.unsqueeze(1), user_ids.unsqueeze(0))
    false_neg_mask = (same_item_mask | same_user_mask) & ~diag_mask
    logits = logits.masked_fill(false_neg_mask, SAFE_NEG_INF)
    
    metrics = {}
    
    # -----------------------------------------------------------
    # 2. Hard Negative Processing (오직 final_idx 위치만 연산 및 조립)
    # -----------------------------------------------------------
    num_hn_to_use = 50
    
    if hn_item_emb is not None and batch_hard_neg_ids is not None and final_idx is not None and final_idx.numel() > 0:
        
        num_finals = final_idx.size(0)
        pool_multiplier = 3
        num_pool = num_hn_to_use * pool_multiplier  
        skip_top_k = 10 

        u_emb_f = user_emb[final_idx]                  # [num_finals, D]
        pos_sim_f = pos_sim[final_idx]                 # [num_finals]
        
        hn_emb_f = hn_item_emb                         
        hn_ids_f = batch_hard_neg_ids                  
        
        flat_hist_f = flat_history_item_ids[final_idx] if flat_history_item_ids is not None else None

        # [STEP 1] 기울기(Gradient) 추적 없이 후보군 필터링
        with torch.no_grad():
            hn_emb_no_grad_f = hn_emb_f.detach() 
            hn_sim_no_grad_f = torch.bmm(u_emb_f.unsqueeze(1), hn_emb_no_grad_f.transpose(1, 2)).squeeze(1)
            
            absolute_fn_mask_f = torch.zeros_like(hn_sim_no_grad_f, dtype=torch.bool, device=device)
            if flat_hist_f is not None:
                absolute_fn_mask_f = (hn_ids_f.unsqueeze(2) == flat_hist_f.unsqueeze(1)).any(dim=2)
            
            boundary_ratio = 0.80
            dynamic_boundary_f = pos_sim_f.unsqueeze(1) * boundary_ratio 
            dynamic_fn_mask_f = hn_sim_no_grad_f >= dynamic_boundary_f
            
            final_fn_mask_f = absolute_fn_mask_f | dynamic_fn_mask_f
            masked_sims_f = hn_sim_no_grad_f.masked_fill(final_fn_mask_f, -1e4)
            
            _, top_idx_all_f = torch.topk(masked_sims_f, num_pool + skip_top_k, dim=1)
            top_idx_pool_f = top_idx_all_f[:, skip_top_k:]

            rand_idx_f = torch.rand(num_finals, num_pool, device=device).argsort(dim=1)[:, :num_hn_to_use]
            top_idx_f = torch.gather(top_idx_pool_f, 1, rand_idx_f)

        # [STEP 2] 본 학습용 임베딩 조립 (initial N개)
        batch_hn_ids_final_f = torch.gather(hn_ids_f, 1, top_idx_f)
        top_idx_expanded_f = top_idx_f.unsqueeze(-1).expand(-1, -1, hn_emb_f.size(2))
        hn_emb_final_f = torch.gather(hn_emb_f, 1, top_idx_expanded_f).detach()

        # [STEP 3] 본 연산 (Gradient 흐름)
        hn_sim_f = torch.bmm(u_emb_f.unsqueeze(1), hn_emb_final_f.transpose(1, 2)).squeeze(1) 
        hn_logits_f = (hn_sim_f / temperature) * alpha
        
        if lambda_logq > 0.0:
            hn_log_q_f = log_q_tensor[batch_hn_ids_final_f] 
            hn_logits_f = hn_logits_f - (hn_log_q_f * lambda_logq)
        
        final_safety_mask_f = hn_sim_f >= (pos_sim_f.unsqueeze(1) * boundary_ratio)
        hn_logits_f = hn_logits_f.masked_fill(final_safety_mask_f, SAFE_NEG_INF)

        # =======================================================
        # 💡 [핵심 결합 로직] 전체 N 차원 캔버스에 마지막 스텝(final_idx) 퍼즐 조각 끼워넣기
        # =======================================================
        hn_logits_full = torch.full((N, num_hn_to_use), SAFE_NEG_INF, device=device)
        hn_logits_full[final_idx] = hn_logits_f
        
        logits = torch.cat([logits, hn_logits_full], dim=1)
        
        if return_metrics:
            metrics['hn/discarded_ratio'] = dynamic_fn_mask_f.float().mean().item()
            valid_hn_sim_f = hn_sim_f[~final_safety_mask_f]
            metrics['sim/hn_true_hard'] = valid_hn_sim_f.mean().item() if valid_hn_sim_f.numel() > 0 else 0.0
            
    else: 
        # HNM이 비활성화되었거나 final_idx가 없을 때 (행렬 차원 유지를 위한 더미 결합)
        hn_logits_full = torch.full((N, num_hn_to_use), SAFE_NEG_INF, device=device)
        logits = torch.cat([logits, hn_logits_full], dim=1)

    logits = torch.clamp(logits, min=SAFE_NEG_INF, max=1e4)

    # -----------------------------------------------------------
    # 3. Loss 계산 (Cross Entropy)
    # -----------------------------------------------------------
    if step_weights is not None:
        loss_unreduced = F.cross_entropy(logits, labels, reduction='none')
        loss = (loss_unreduced * step_weights).sum() / weight_sum
        if return_metrics:
            metrics['sim/pos'] = ((pos_sim * step_weights).sum() / weight_sum).item()
    else:
        loss = F.cross_entropy(logits, labels)
        if return_metrics:
            metrics['sim/pos'] = pos_sim.mean().item()

    # -----------------------------------------------------------
    # 4. Probabilities Metrics
    # -----------------------------------------------------------
    if return_metrics:
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            if hn_item_emb is not None and batch_hard_neg_ids is not None and final_idx is not None and final_idx.numel() > 0:
                hn_probs_sum = probs[:, N:].sum(dim=1) 
                neg_probs_total = 1.0 - probs.diagonal() 
                relative_hn_ratio = hn_probs_sum / (neg_probs_total + 1e-9) 
                
                if step_weights is not None:
                    metrics['hn/influence_ratio'] = ((hn_probs_sum * step_weights).sum() / weight_sum).item()
                    metrics['hn/relative_influence'] = ((relative_hn_ratio * step_weights).sum() / weight_sum).item()
                else:
                    metrics['hn/influence_ratio'] = hn_probs_sum.mean().item()
                    metrics['hn/relative_influence'] = relative_hn_ratio.mean().item()

        return loss, metrics

    return loss