import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)
    
    
from tqdm import tqdm
import wandb
from torch.nn.attention import SDPBackend, sdpa_kernel
import os
import pandas as pd
import torch
import torch.nn.functional as F



from tqdm import tqdm
import sys

from sheduler import EarlyStopping, get_cosine_schedule_with_warmup
from tower_code.params_config import PipelineConfig
from tower_code.v3_model_usertower import inbatch_corrected_logq_loss_with_hard_neg_margin
from tower_code.v3_utils import create_dataloaders, load_aligned_pretrained_embeddings, load_item_metadata_hashed, load_item_tower_state_dict, prepare_features, setup_environment, setup_models

from preprocessor.preprocessor_v2 import dataset_peek_v3




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
