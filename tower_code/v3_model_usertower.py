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