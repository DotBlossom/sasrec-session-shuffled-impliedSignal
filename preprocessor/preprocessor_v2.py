import numpy as np
import pandas as pd
import gc

import torch
import torch
import numpy as np

def dataset_peek_v3(dataset, processor):
    """동적 Offset 및 AS-OF 시퀀스 정합성 정밀 검수"""
    print("\n🔍 [Data Peek V2] Verifying Leak-Free Feature Integrity...")
    
    # 첫 번째 유저 샘플 (보통 데이터가 많은 유저를 보기 위해 dataset[0] 사용)
    sample = dataset[0]
    
    # 텐서 이동 (CPU)
    ids = sample['item_ids'].cpu().numpy()
    targets = sample['target_ids'].cpu().numpy()
    offsets = sample['recency_offset'].cpu().numpy()
    weeks = sample['current_week'].cpu().numpy()
    
    # 1. 시퀀스 Shift 및 데이터 마스킹 확인
    # 0(패딩)이 아닌 첫 데이터 위치 확인
    valid_mask = ids != 0
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 1:
        idx = valid_indices[0]
        print(f"✅ [Shift Check]")
        print(f"   - Input[t]:  {ids[idx]:>8} | Input[t+1]: {ids[idx+1]:>8}")
        print(f"   - Target[t]: {targets[idx]:>8} | Target[t+1]: {targets[idx+1]:>8}")
        if ids[idx+1] == targets[idx]:
            print("   👉 Result: Shift Logic is Correct.")
            
    # 2. 💡 동적 Recency Offset & Week 정합성 확인
    print(f"\n✅ [Time Feature Check]")
    if len(valid_indices) > 1:
        # 마지막 유효 인덱스 (Train 시점의 Target_Now 기준점 근처)
        last_idx = valid_indices[-1]
        
        print(f"   - Target Now 시점 기준 (Train Mode: {dataset.is_train})")
        print(f"   - Last Item Offset: {offsets[last_idx]} days (0에 가까울수록 정상)")
        print(f"   - Prev Item Offset: {offsets[last_idx-1]} days")
        
        # 주차 정보 (절대 계절성)
        print(f"   - Sequence Weeks:   {weeks[valid_indices][:5].tolist()} ...")
        
        # 검증: Offset은 시간 역순이어야 함 (과거일수록 숫자가 커야 함)
        if offsets[valid_indices[0]] >= offsets[valid_indices[-1]]:
            print("   👉 Result: Dynamic Offset Direction is Correct (Past has larger offset).")
        else:
            print("   👉 Result: ❌ Offset Error! Future items have larger offset.")

    # 3. 📊 AS-OF Dynamic Features (Continuous/Bucket) 확인
    print(f"\n✅ [AS-OF Sequence Check]")
    # 예시로 가격 버킷과 누적 횟수 확인
    price_b = sample['price_bucket'].cpu().numpy()
    cnt_b = sample['cnt_bucket'].cpu().numpy()
    
    # 동일 세션 내 값이 유지되는지 확인 (앞선 2~3개 샘플링)
    print(f"   - Price Buckets: {price_b[valid_indices][:5].tolist()}")
    print(f"   - Total Cnt Buckets: {cnt_b[valid_indices][:5].tolist()}")
    
    # 4. 👤 Static Metadata (Batch 단위 고정값) 확인
    print(f"\n✅ [Static Metadata Check]")
    print(f"   - Age Bucket: {sample['age_bucket'].item()}")
    print(f"   - Club Status: {sample['club_status_ids'].item()}")
    
    print("\n🚀 [Final Verdict]: If 'Last Item Offset' is near 0 and 'Past' is larger, your Leak-Free Logic is READY!")

# 실행 방법
# dataset_peek_v2(train_dataset, processor)

def make_user_features_v3(train_df, target_val_path, USER_META_PATH, USER_FEAT_VAL_PATH_PQ):
    print("\n👤 [User Stats] Calculating AS-OF Sequence Features (Zero Time Leakage)...")

    '''
    if target_val_path != None:

        print("\n🎯 [Validation User Stats] Preparing point-in-time features...")

        # 1. 평가 대상 유저 ID 추출 (정답지가 있는 유저들)
        target_val = pd.read_parquet(target_val_path)
        val_user_set = set(target_val['customer_id'].unique())
        print(f" -> Found {len(val_user_set):,} target users for validation.")

        # 2. 9/15 이전 거래 중 '평가 대상 유저'의 기록만 추출
        # (이미 full_df가 9/15 이전 데이터라면 날짜 필터는 생략 가능)
        train_df = train_df[train_df['customer_id'].isin(val_user_set)].copy()
        print(f" -> Using {len(train_df):,} transaction records for feature calculation.")
    '''
    # 💡 [핵심 추가 1] 전체 Train 데이터에서 "Warm User(최소 3개 이상 구매)" 추출
    train_user_counts = train_df.groupby('customer_id').size()
    warm_train_users = set(train_user_counts[train_user_counts >= 3].index)

    if target_val_path != None:
        print("\n🎯 [Validation User Stats] Preparing point-in-time features...")

        # 1. 평가 대상 유저 ID 추출 (정답지가 있는 유저들: 최소 1개 이상 구매자)
        target_val = pd.read_parquet(target_val_path)
        val_user_set = set(target_val['customer_id'].unique())
        print(f" -> Found {len(val_user_set):,} users with targets in validation period.")

        # 💡 [핵심 추가 2] 교집합(Intersection): 정답도 있고, 과거 이력도 3개 이상인 완벽한 타겟만 필터링
        final_valid_users = val_user_set.intersection(warm_train_users)
        print(f" -> 🛡️ [Filter] Retained {len(final_valid_users):,} strictly valid users (Target >= 1 & History >= 3).")

        # 2. 9/15 이전 거래 중 '최종 평가 대상 유저'의 기록만 추출
        train_df = train_df[train_df['customer_id'].isin(final_valid_users)].copy()
        
    else:
        # Train 모드일 경우: Warm User만 남김
        train_df = train_df[train_df['customer_id'].isin(warm_train_users)].copy()
        print(f" -> 🛡️ [Filter] Retained {len(warm_train_users):,} warm users (History >= 3) for Training.")







    # 1. 정렬: 시간의 흐름과 아이템 시퀀스 순서를 완벽히 보장
    train_df = train_df.sort_values(['customer_id', 't_dat']).copy()
    train_df['day_of_week'] = train_df['t_dat'].dt.dayofweek
    train_df['is_weekend'] = (train_df['day_of_week'] >= 5).astype(np.int8)

    # 2. 세션(일자) 단위 집계
    daily_agg = train_df.groupby(['customer_id', 't_dat']).agg(
        daily_cnt=('article_id', 'count'),
        daily_sum_price=('price', 'sum'),
        daily_last_price=('price', 'last'),
        daily_weekend_cnt=('is_weekend', 'sum'),
        daily_channel_sum=('sales_channel_id', 'sum')
    ).reset_index()

    # 재구매 비율을 위한 첫 구매 여부 계산
    train_df['is_first_buy'] = (~train_df.duplicated(subset=['customer_id', 'article_id'])).astype(int)
    daily_first = train_df.groupby(['customer_id', 't_dat'])['is_first_buy'].sum().reset_index(name='daily_unique')
    daily_agg = pd.merge(daily_agg, daily_first, on=['customer_id', 't_dat'])

    # 가격 표준편차를 위한 제곱합 계산 (분산 = E[X^2] - E[X]^2)
    train_df['price_sq'] = train_df['price'] ** 2
    daily_sq = train_df.groupby(['customer_id', 't_dat'])['price_sq'].sum().reset_index(name='daily_sum_sq')
    daily_agg = pd.merge(daily_agg, daily_sq, on=['customer_id', 't_dat'])

    # ------------------------------------------------------------------
    # 💡 3. 핵심: AS-OF (이전 세션까지의 누적) 계산 (shift(1) 적용)
    # 현재 세션의 정보는 철저히 배제하고, 직전 세션까지의 합산값만 가져옵니다.
    # ------------------------------------------------------------------
    grouped = daily_agg.groupby('customer_id')
    daily_agg['cum_cnt'] = grouped['daily_cnt'].cumsum().shift(1).fillna(0)
    daily_agg['cum_sum_price'] = grouped['daily_sum_price'].cumsum().shift(1).fillna(0)
    daily_agg['cum_unique'] = grouped['daily_unique'].cumsum().shift(1).fillna(0)
    daily_agg['cum_weekend'] = grouped['daily_weekend_cnt'].cumsum().shift(1).fillna(0)
    daily_agg['cum_channel'] = grouped['daily_channel_sum'].cumsum().shift(1).fillna(0)
    daily_agg['cum_sum_sq'] = grouped['daily_sum_sq'].cumsum().shift(1).fillna(0)
    
    daily_agg['last_price'] = grouped['daily_last_price'].shift(1).fillna(0)
    daily_agg['last_t_dat'] = grouped['t_dat'].shift(1)

    # 4. AS-OF 비율 및 파생 변수 계산
    safe_cnt = daily_agg['cum_cnt'].replace(0, 1) # 0 나누기 방지
    
    daily_agg['asof_avg_price'] = (daily_agg['cum_sum_price'] / safe_cnt).astype(np.float32)
    daily_agg['asof_repurchase_ratio'] = (1.0 - (daily_agg['cum_unique'] / safe_cnt)).astype(np.float32)
    daily_agg['asof_weekend_ratio'] = (daily_agg['cum_weekend'] / safe_cnt).astype(np.float32)
    daily_agg['asof_last_price_diff'] = (daily_agg['last_price'] - daily_agg['asof_avg_price']).astype(np.float32)
    
    # 선호 채널
    daily_agg['asof_channel_avg'] = (daily_agg['cum_channel'] / safe_cnt).astype(np.float32)
    daily_agg['asof_preferred_channel'] = np.where(daily_agg['cum_cnt'] == 0, 1, 
                                          np.where(daily_agg['asof_channel_avg'] > 1.5, 2, 1)).astype(np.int8)
    
    # Recency (최근 구매 경과일)
    daily_agg['asof_recency_days'] = (daily_agg['t_dat'] - daily_agg['last_t_dat']).dt.days.fillna(365.0).astype(np.float32)
    daily_agg.loc[daily_agg['cum_cnt'] == 0, 'asof_recency_days'] = 365.0

    # Price STD 계산
    mean_sq = daily_agg['cum_sum_sq'] / safe_cnt
    sq_mean = (daily_agg['cum_sum_price'] / safe_cnt) ** 2
    var = (mean_sq - sq_mean).clip(lower=0)
    daily_agg['asof_price_std'] = np.sqrt(var).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. AS-OF 피처를 원본 train_df에 병합 (개별 아이템 시퀀스에 맵핑)
    # ------------------------------------------------------------------
    merge_cols = ['customer_id', 't_dat', 'cum_cnt', 'asof_avg_price', 'asof_recency_days',
                  'asof_price_std', 'asof_last_price_diff', 'asof_repurchase_ratio',
                  'asof_weekend_ratio', 'asof_preferred_channel']
    train_df = pd.merge(train_df, daily_agg[merge_cols], on=['customer_id', 't_dat'], how='left')

    # 6. 전체 데이터 기반 Bucketing & Scaling
    print("   ⚙️ Bucketing & Scaling AS-OF Features...")
    train_df['asof_avg_price_bucket'] = pd.qcut(train_df['asof_avg_price'], q=10, labels=False, duplicates='drop').fillna(0).astype(np.int8) + 1
    train_df['asof_total_cnt_bucket'] = pd.qcut(train_df['cum_cnt'], q=10, labels=False, duplicates='drop').fillna(0).astype(np.int8) + 1
    train_df['asof_recency_bucket'] = pd.qcut(train_df['asof_recency_days'], q=10, labels=False, duplicates='drop').fillna(0).astype(np.int8) + 1

    cont_cols = ['asof_price_std', 'asof_last_price_diff', 'asof_repurchase_ratio', 'asof_weekend_ratio']
    for col in cont_cols:
        c_mean = train_df[col].mean()
        c_std = train_df[col].std() + 1e-9
        train_df[f'{col}_scaled'] = ((train_df[col] - c_mean) / c_std).astype(np.float32)


    sample_uid = train_df['customer_id'].iloc[0] 
    monitor_asof_logic(train_df, sample_uid)



    # ------------------------------------------------------------------
    # 💡 7. 유저별 시퀀스 리스트로 압축 (List Aggregation)
    # ------------------------------------------------------------------
    print("   ⚙️ Aggregating to Sequence Lists...")
# 💡[신규] Global Time을 유저 최종일이 아닌 "해당 트랜잭션 발생일(t_dat)" 기준으로 시퀀스화
    # global_now_dt = pd.to_datetime("2020-09-22") # 예시 기준일
    train_df['asof_t_dat_ordinal'] = train_df['t_dat'].apply(lambda x: x.toordinal()).astype(np.int32)
    train_df['asof_current_week'] = train_df['t_dat'].dt.isocalendar().week.astype(np.int8)
    # 💡 동적 피처 리스트(dynamic_cols)에 글로벌 타임도 추가하여 함께 압축
    # ------------------------------------------------------------------
    print("   ⚙️ Calculating sequence_deltas before aggregation...")

    # 1️⃣ 계산에 필요한 날짜 정수(ordinal) 생성
    train_df['days_int'] = train_df['t_dat'].apply(lambda x: x.toordinal())
    
    # 2️⃣ 유저별 마지막 구매일 찾기
    user_max_day = train_df.groupby('customer_id')['days_int'].transform('max')
    
    # 3️⃣ 실제 컬럼 생성 (이게 있어야 KeyError가 안 납니다!)
    train_df['sequence_deltas'] = (user_max_day - train_df['days_int']).astype(np.int32)
    dynamic_cols = [
        'article_id','sequence_deltas', 
        'price',     # 👈 시간 간격 시퀀스 (추가됨)
        'asof_avg_price_bucket', 'asof_total_cnt_bucket', 'asof_recency_bucket',
        'asof_price_std_scaled', 'asof_last_price_diff_scaled', 'asof_repurchase_ratio_scaled', 'asof_weekend_ratio_scaled',
        'asof_preferred_channel', 'asof_t_dat_ordinal', 'asof_current_week' # 타임 변수 추가
    ]
    # groupby.agg(list)를 통해 시퀀스 길이에 완벽히 매칭되는 피처 리스트 생성
    user_seq_df = train_df.groupby('customer_id')[dynamic_cols].agg(list).reset_index()
    user_seq_df.rename(columns={'article_id': 'sequence_ids'}, inplace=True)
    
    # 8. 정적(Static) 고객 정보 병합
    print("   👥 Loading & Merging Customer Metadata...")
    customers_df = pd.read_csv(USER_META_PATH)
    
    age_median = customers_df['age'].median()
    customers_df['age'] = customers_df['age'].fillna(age_median)
    customers_df['age_bucket'] = pd.qcut(customers_df['age'], q=10, labels=False, duplicates='drop').astype(np.int8) + 1
    
    customers_df['FN'] = customers_df['FN'].fillna(0).astype(np.int8)
    customers_df['Active'] = customers_df['Active'].fillna(0).astype(np.int8)
    
    status_map = {'ACTIVE': 1, 'PRE-CREATE': 2}
    customers_df['club_member_status'] = customers_df['club_member_status'].fillna('OTHER').astype(str).str.upper()
    customers_df['club_member_status_idx'] = customers_df['club_member_status'].map(status_map).fillna(0).astype(np.int8)
    
    news_map = {'REGULARLY': 1}
    customers_df['fashion_news_frequency'] = customers_df['fashion_news_frequency'].fillna('NONE').astype(str).str.upper()
    customers_df['fashion_news_frequency_idx'] = customers_df['fashion_news_frequency'].map(news_map).fillna(0).astype(np.int8)

    meta_cols = ['customer_id', 'age_bucket', 'FN', 'Active', 'club_member_status_idx', 'fashion_news_frequency_idx']
    
    final_df = pd.merge(user_seq_df, customers_df[meta_cols], on='customer_id', how='left')

    # 결측치 방어
    final_df['age_bucket'] = final_df['age_bucket'].fillna(0).astype(np.int8)
    for col in ['FN', 'Active', 'club_member_status_idx', 'fashion_news_frequency_idx']:
        final_df[col] = final_df[col].fillna(0).astype(np.int8)

    print("\n🔍 [Check] Generated AS-OF Sequence Features:")
    print(final_df.head(2).T.to_string())
    
    # 리스트 포맷을 완벽하게 보존하는 Parquet 저장
    final_df.to_parquet(USER_FEAT_VAL_PATH_PQ, index=False)
    print("   ✅ User AS-OF features successfully calculated and saved!")
    validate_final_lists(final_df)
    del daily_agg, customers_df, user_seq_df; gc.collect()
    return final_df

class FeatureProcessor_v3:
    def __init__(self, user_path, item_path, base_processor=None):
        print("🚀 Loading preprocessed features...")
        self.users = pd.read_parquet(user_path).drop_duplicates(subset=['customer_id']).set_index('customer_id')
        self.items = pd.read_parquet(item_path).drop_duplicates(subset=['article_id']).set_index('article_id')
        self.seqs = self.users[['sequence_ids', 'sequence_deltas']].copy() # 만약 deltas도 같이 묶으셨다면 포함

        # 인덱스 타입 강제 (String)
        self.users.index = self.users.index.astype(str)
        self.items.index = self.items.index.astype(str)
        self.seqs.index = self.seqs.index.astype(str)
        # 💡 [핵심 추가] 학습/평가에 의미가 없는 유저(데이터 1개 이하) 사전 필터링
        initial_count = len(self.seqs)
        self.seqs = self.seqs[self.seqs['sequence_ids'].apply(len) >= 2]
        filtered_count = len(self.seqs)
        
        print(f"✂️ [Filter] Removed {initial_count - filtered_count:,} users with less than 2 items.")
        # =================================================================
        # 1. ID Mappings (1-based, 0 is Padding)
        # =================================================================
        self.user_ids = self.seqs.index.tolist() # 시퀀스가 존재하는 유저만 대상
        self.user2id = {uid: i + 1 for i, uid in enumerate(self.users.index)}
        self.num_users = len(self.user_ids)
        
        
        
        
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
        num_users_total = len(self.users) + 1
        
        # 💡 [신규] 동적(Dynamic) 시퀀스 피처를 저장할 딕셔너리 
        # Numpy 배열로 변환하여 Dataset의 input_indices 슬라이싱에 완벽히 호환되도록 만듦
        self.u_dyn_buckets = {}  # Shape: (seq_len, 3)
        self.u_dyn_conts = {}    # Shape: (seq_len, 4)
        self.u_dyn_cats = {}     # Shape: (seq_len, 1) - preferred_channel
        
        # 💡 [변경] 유저 피처를 Static(1D)과 Dynamic(2D 시퀀스)으로 분리하여 저장
        self.u_static_buckets = np.zeros((num_users_total, 1), dtype=np.int64) # age
        self.u_static_cats = np.zeros((num_users_total, 4), dtype=np.int64) # club, news, fn, active
        
        # 동적 피처를 담을 딕셔너리 (유저별 시퀀스 길이가 다르므로 딕셔너리 + Numpy 배열 사용)
        self.u_dyn_buckets = {}
        self.u_dyn_conts = {}
        self.u_dyn_cats = {}
        self.u_dyn_time = {}
        self.u_seqs = {}
        self.u_deltas = {}
        for uid, row in self.users.iterrows():
            if uid not in self.user2id: continue
            uidx = self.user2id[uid]
            
            # 💡 [신규] 시퀀스 데이터를 딕셔너리에 저장
            self.u_seqs[uidx] = row['sequence_ids']
            self.u_deltas[uidx] = row['sequence_deltas']
            
            self.u_static_buckets[uidx] = [row['age_bucket']]
            self.u_static_cats[uidx] = [
                row['club_member_status_idx'], row['fashion_news_frequency_idx'], 
                row['FN'], row['Active']
            ]
            
            # 💡 [최적화] dtype을 int8, int32, float16으로 낮춰서 RAM 폭파 방지
            self.u_dyn_buckets[uidx] = np.column_stack([
                row['asof_avg_price_bucket'], row['asof_total_cnt_bucket'], row['asof_recency_bucket']
            ]).astype(np.int8)
            
            self.u_dyn_conts[uidx] = np.column_stack([
                row['asof_price_std_scaled'], row['asof_last_price_diff_scaled'],
                row['asof_repurchase_ratio_scaled'], row['asof_weekend_ratio_scaled']
            ]).astype(np.float16)
            
            self.u_dyn_cats[uidx] = np.array(row['asof_preferred_channel'], dtype=np.int8).reshape(-1, 1)
            
            self.u_dyn_time[uidx] = np.column_stack([
                row['asof_t_dat_ordinal'], row['asof_current_week']
            ]).astype(np.int32)
            
        self.i_side_arr = np.zeros((self.num_items + 1, 4), dtype=np.int16)
        for iid, row in self.items.iterrows():
            if iid not in self.item2id: continue
            idx = self.item2id[iid]
            self.i_side_arr[idx] = [
                row.get('type_id', 0), row.get('color_id', 0), 
                row.get('graphic_id', 0), row.get('section_id', 0)
            ]
        # 💡 [핵심 추가] 데이터프레임을 삭제하기 전에 확률 값만 미리 추출해서 저장합니다.
        print("📊 Extracting item probabilities for Log-Q correction...")
        # reindex를 통해 item2id 순서와 동일하게 정렬된 넘파이 배열을 만듭니다.
        self.item_raw_probs = self.items['raw_probability'].reindex(self.item_ids).values

        # -----------------------------------------------------------
        # 🧹 이제 안심하고 RAM을 잡아먹는 주범들을 삭제합니다.
        # -----------------------------------------------------------
        del self.users
        del self.items
        del self.seqs
        import gc
        gc.collect()
        print("   🧹 Memory cleaned: All heavy DataFrames deleted!")

    def get_logq_probs(self, device):
        """Negative Sampling이나 Loss 보정을 위한 아이템 등장 확률 Log 반환"""
        raw_probs = self.item_raw_probs 
        
        eps = 1e-6
        # 넘파이 배열이므로 그대로 연산 가능
        sorted_probs = np.nan_to_num(raw_probs, nan=0.0) + eps
        sorted_probs /= sorted_probs.sum()
        
        log_q_values = np.log(sorted_probs).astype(np.float32)
        
        full_log_q = np.zeros(self.num_items + 1, dtype=np.float32)
        full_log_q[1:] = log_q_values 
        full_log_q[0] = -20.0 # Padding Index
    
        return torch.tensor(full_log_q, dtype=torch.float32).to(device)

def monitor_asof_logic(df, sample_user_id):
    """특정 유저의 원본 거래와 계산된 AS-OF 피처를 비교하여 누수 여부 확인"""
    user_sample = df[df['customer_id'] == sample_user_id].sort_values('t_dat')
    
    print(f"\n📊 [Monitoring] User: {sample_user_id}")
    cols_to_show = ['t_dat', 'article_id', 'price', 'cum_cnt', 'asof_avg_price', 'asof_recency_days']
    
    # 상위 10개 행 출력: 첫 세션의 통계량이 0(또는 기본값)인지 확인
    print(user_sample[cols_to_show].head(10).to_string())
    
    # 검증 포인트 1: 첫 번째 행의 cum_cnt는 무조건 0이어야 함 (AS-OF Shift 확인)
    first_cum_cnt = user_sample.iloc[0]['cum_cnt']
    assert first_cum_cnt == 0, f"❌ Leakage Detected! First record should have 0 cumulative count, but got {first_cum_cnt}"
    
    # 검증 포인트 2: 시퀀스 길이 일치 확인
    # 리스트로 묶인 후 시퀀스 길이가 아이템 수와 같은지 체크 (aggregation 직후 실행)
    print(f"✅ AS-OF Shift Logic Passed: First session stats are properly masked.")

# make_user_features 함수 마지막 부분에 추가
def validate_final_lists(final_df):
    print("\n🔍 [Check] Sequence Length Consistency:")
    # 랜덤 유저 1명의 리스트 길이들 비교
    sample = final_df.sample(1).iloc[0]
    lengths = {col: len(sample[col]) for col in final_df.columns if isinstance(sample[col], list)}
    
    # 모든 리스트 피처의 길이가 동일한지 확인
    unique_lengths = set(lengths.values())
    if len(unique_lengths) == 1:
        print(f"✅ All feature sequences have consistent length: {list(unique_lengths)[0]}")
    else:
        print(f"❌ Consistency Error! Found varying lengths: {lengths}")
        
def monitor_processor_storage(processor):
    print("\n⚡ [Processor Monitoring] Fast Lookup Table Sanity Check")
    
    # 1. 메모리 사용량 대략적 파악
    num_users = len(processor.u_dyn_buckets)
    print(f" -> Total Users in Dynamic Storage: {num_users:,}")
    
    # 2. 랜덤 샘플 유저 정합성 체크
    if num_users > 0:
        sample_uid = list(processor.u_dyn_buckets.keys())[0]
        
        dyn_b = processor.u_dyn_buckets[sample_uid]
        dyn_c = processor.u_dyn_conts[sample_uid]
        dyn_t = processor.u_dyn_time[sample_uid]
        
        print(f" -> Sample User Map ID: {sample_uid}")
        print(f" -> Dynamic Buckets Shape: {dyn_b.shape} (Expect: [Seq_len, 3])")
        print(f" -> Dynamic Conts Shape: {dyn_c.shape} (Expect: [Seq_len, 4])")
        print(f" -> Dynamic Time Shape: {dyn_t.shape} (Expect: [Seq_len, 2])")
        
        # 3. 값의 범위 체크 (Scaling/Bucketing 확인)
        print(f" -> Conts Mean (Scaled): {dyn_c.mean():.4f} (Expect: Near 0 if well scaled)")
        print(f" -> Buckets Max: {dyn_b.max()} (Expect: <= 10 or 11)")

