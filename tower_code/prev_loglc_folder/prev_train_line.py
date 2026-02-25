
'''
import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def run_pipeline_opt_v2():
    """Airflow DAG나 MLflow Run에서 직접 호출하는 엔트리 포인트"""
    print("🚀 Starting User Tower Training Pipeline...")
    
    
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
    cfg.lr = 2e-3
    
    
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
    val_loader = create_dataloaders(val_processor, cfg, "2020-09-22",aligned_vecs, is_train=False)
    dataset_peek_v3(train_loader.dataset, processor)
    #cfg.num_prod_types = int(processor.i_side_arr[:, 0].max()) + 1
    #cfg.num_colors = int(processor.i_side_arr[:, 1].max()) + 1
    #cfg.num_graphics = int(processor.i_side_arr[:, 2].max()) + 1
    #cfg.num_sections = int(processor.i_side_arr[:, 3].max()) + 1
    print(f"✅ Final Model Config - Total Hash Nodes: {cfg.num_prod_types + cfg.num_colors + cfg.num_graphics + cfg.num_sections}")

    
    
        
    wandb.init(
        project="SASRec-User-Tower-causality-Optimization-v2", # 프로젝트명
        name=f"run_lr_{cfg.lr}_epoch_{cfg.epochs}_shuffle_session_correction", # 실험 이름
        config=cfg.__dict__ # 하이퍼파라미터 저장
    )
    
    
    # -----------------------------------------------------------
    # 3. Models & Optimizer Setup (초기 상태: Epoch 1용 세팅)
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, item_state_dict, log_q_tensor)
    item_tower.init_from_pretrained(aligned_vecs.to(device))
    item_tower.set_freeze_state(True)
    print(f"❄️ Epoch 1: Item Tower FROZEN! (User Tower LR: {cfg.lr})")
    
    
    optimizer = torch.optim.AdamW(user_tower.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # 💡 [신규 스케줄러 세팅] 
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * 0.1) # 전체 훈련의 10% 구간 동안 서서히 LR 상승
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # Best Model 트래킹 변수
    best_recall_100 = 0.0

    # -----------------------------------------------------------
    # 4. Training Loop
    # -----------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        if epoch == 2:
            print("\n🔥 [Dynamic Unfreeze] Epoch 2: Item Tower Joint Training 시작!")
            item_tower.set_freeze_state(False)
            item_finetune_lr = cfg.lr * 0.05 
            
            optimizer.add_param_group({
                'params': item_tower.parameters(), 
                'lr': item_finetune_lr # 이 그룹도 스케줄러의 비율 감쇠를 동일하게 적용받음
            })
            print(f"   - User Tower LR: {cfg.lr}")
            print(f"   - Item Tower LR: {item_finetune_lr} (Fine-tuning mode)")

        # ------------------- 훈련 (Train) -------------------
        avg_loss = train_user_tower_all_time(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower, # 정적 벡터 대신 모델 객체 자체를 넘김
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=None,
            scheduler=scheduler,
            seq_labels = SEQ_LABELS,
            static_labels = STATIC_LABELS,
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

        
        if current_recall_100 > best_recall_100:
            print(f"🌟 [New Best!] Recall@100 updated: {best_recall_100:.2f}% -> {current_recall_100:.2f}%")
            best_recall_100 = current_recall_100
            
            # 최고 성능 달성 시 파라미터 덮어쓰기 저장
            torch.save(user_tower.state_dict(), os.path.join(cfg.model_dir, "best_user_tower_v3_allseq_session.pth"))
            torch.save(item_tower.state_dict(), os.path.join(cfg.model_dir, "best_item_tower_v3_allseq_session.pth"))
            print("   💾 Best model weights saved to disk.")
        else:
            print(f"   - (Current Best: {best_recall_100:.2f}%)")
            
    print("\n🎉 Pipeline Execution Finished Successfully!")

def run_pipeline():
    """Airflow DAG나 MLflow Run에서 직접 호출하는 엔트리 포인트"""
    print("🚀 Starting User Tower Training Pipeline...")
    
    
    SEQ_LABELS = ['item_id', 'time', 'type', 'color', 'graphic', 'section']
    STATIC_LABELS = ['age', 'price', 'cnt', 'recency', 'channel', 'club', 'news', 'fn', 'active', 'cont']
    # 1. Config & Env
    cfg = PipelineConfig()
    device = setup_environment()
    processor, val_processor, cfg = prepare_features(cfg)
    
    # item metadata cfg
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    
    # 2. Data 가져오기
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    # ❌ full_item_embeddings = aligned_vecs.to(device) # 더 이상 사용하지 않음
    
    item_state_dict = load_item_tower_state_dict(cfg.model_dir, cfg.item_tower_pth_name, device)
    log_q_tensor = processor.get_logq_probs(device)
    
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    
    train_loader = create_dataloaders(processor, cfg, aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, aligned_vecs, is_train=False)
    dataset_peek_v3(train_loader.dataset, processor)
    
    
    
        
    wandb.init(
        project="SASRec-User-Tower-causality-Optimization-v2", # 프로젝트명
        name=f"run_lr_{cfg.lr}_epoch_{cfg.epochs}", # 실험 이름
        config=cfg.__dict__ # 하이퍼파라미터 저장
    )
    
    
    # -----------------------------------------------------------
    # 3. Models & Optimizer Setup (초기 상태: Epoch 1용 세팅)
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, item_state_dict, log_q_tensor)
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")
    
    # 💡 [핵심 반영] 아까 만든 깔끔한 메서드로 사전학습 벡터 강제 주입!
    item_tower.init_from_pretrained(aligned_vecs.to(device))
    
    
    
    # 💡 [초기화] Epoch 1에서는 User Tower만 학습하도록 Item Tower 완전 동결
    item_tower.set_freeze_state(True)
    print(f"❄️ Epoch 1: Item Tower FROZEN! (User Tower LR: {cfg.lr})")
    
    # User Tower만 포함된 Optimizer 생성
    optimizer = torch.optim.AdamW(user_tower.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # 💡 [스케줄러] Validation 지표(Recall@100)를 보고 정체 시 학습률 감소 (patience=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # Best Model 트래킹 변수
    best_recall_100 = 0.0

    # -----------------------------------------------------------
    # 4. Training Loop
    # -----------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        
        # 💡 [동적 Unfreeze] Epoch 2 진입 시 딱 한 번 실행하여 Joint Training 시작
        if epoch == 2:
            print("\n🔥 [Dynamic Unfreeze] Epoch 2: Item Tower Joint Training 시작!")
            item_tower.set_freeze_state(False)
            item_finetune_lr = cfg.lr * 0.05 # 아이템은 매우 미세하게만 조정 (User LR의 5%)
            
            # 기존 옵티마이저에 아이템 타워의 파라미터 그룹을 런타임에 동적으로 추가
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
            item_tower=item_tower, # 정적 벡터 대신 모델 객체 자체를 넘김
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=None,
            scheduler=None,
            seq_labels = SEQ_LABELS,
            static_labels = STATIC_LABELS,
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
            
            # 최고 성능 달성 시 파라미터 덮어쓰기 저장
            torch.save(user_tower.state_dict(), os.path.join(cfg.model_dir, "best_user_tower_duo_detach.pth"))
            torch.save(item_tower.state_dict(), os.path.join(cfg.model_dir, "best_item_tower_duo_detach.pth"))
            print("   💾 Best model weights saved to disk.")
        else:
            print(f"   - (Current Best: {best_recall_100:.2f}%)")
            
    print("\n🎉 Pipeline Execution Finished Successfully!")

def run_resume_pipeline(resume_epoch=6, last_best_recall=9.69):
    """저장된 모델을 불러와 Epoch 6부터 재학습을 진행하는 엔트리 포인트"""
    print(f"🚀 Resuming User Tower Training from Epoch {resume_epoch}...")
    # 모델 구조와 일치하는 이름표 정의
    
    
    
    from torch.optim.lr_scheduler import OneCycleLR
    
    
    SEQ_LABELS = ['item_id', 'time', 'type', 'color', 'graphic', 'section']
    STATIC_LABELS = ['age', 'price', 'cnt', 'recency', 'channel', 'club', 'news', 'fn', 'active', 'cont']
    
    cfg = PipelineConfig()
    device = setup_environment()
    processor, val_processor, cfg = prepare_features(cfg)
    # processor.analyze_distributions()
    HASH_SIZE = 1000 
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    # 아이템 개수도 processor에서 가져와서 정확히 매칭 (매우 중요)
    cfg.num_items = len(processor.item2id)
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    log_q_tensor = processor.get_logq_probs(device)
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    

    
    train_loader = create_dataloaders(processor, cfg, aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, aligned_vecs, is_train=False)
    
    
    # ----- Hard Negative Pool 로딩 -----
    cache_dir = r'D:\trainDataset\localprops\cache'
    ultimate_pool_save_path = os.path.join(cache_dir, 'hard_neg_pool_ultimate.npy')
    
    print(f"📦 Loading Hard Negative Pool from {ultimate_pool_save_path}...")
    hard_neg_pool_np = np.load(ultimate_pool_save_path)
    # GPU VRAM에 바로 올립니다 (슬라이싱 속도 극대화)
    hard_neg_pool_tensor = torch.tensor(hard_neg_pool_np, dtype=torch.long, device=device)
    print(f"✅ Hard Negative Pool loaded! Shape: {hard_neg_pool_tensor.shape}")
    # ---------------------------------------------------------
    
    
        
    wandb.init(
        project="SASRec-User-Tower-causality-Optimization", # 프로젝트명
        name=f"run_lr_{cfg.lr}_epoch_{cfg.epochs}_hmn", # 실험 이름
        config=cfg.__dict__ # 하이퍼파라미터 저장
    )
    
    # 2. 모델 생성
    # item_state_dict는 초기화용이므로 비워두거나 기본 로드 후 가중치를 덮어씌웁니다.
    user_tower, item_tower = setup_models(cfg, device, {}, log_q_tensor)
    
    # 3. [핵심] 가중치 불러오기 (Best 모델 로드)
    print("📂 Loading best weights for Resume...")
    user_weight_path = os.path.join(cfg.model_dir, "best_user_tower_c.pth")
    item_weight_path = os.path.join(cfg.model_dir, "best_item_tower_c.pth")

    if os.path.exists(user_weight_path) and os.path.exists(item_weight_path):
        # torch.load는 파일만 읽고, strict 옵션은 load_state_dict에 줍니다.
        user_state_dict = torch.load(user_weight_path, map_location=device)
        user_tower.load_state_dict(user_state_dict, strict=False) 
        
        item_state_dict = torch.load(item_weight_path, map_location=device)
        item_tower.load_state_dict(item_state_dict, strict=False)
        
        print("✅ Successfully loaded best weights from disk (Feature Gates Initialized).")
        # 4. Optimizer & Scheduler 설정
        # 재학습 시에는 Item Tower를 바로 학습 가능 상태로 둡니다.
        item_tower.set_freeze_state(False)
    
    
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    # 두 타워의 파라미터를 처음부터 나누어 관리
    user_lr = 5e-4 
    item_lr = user_lr * 0.05
    
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': user_lr},
        {'params': item_tower.parameters(), 'lr': item_lr}
    ], weight_decay=cfg.weight_decay)

    # 전체 스텝 계산
    total_epochs = resume_epoch + 9 # 예: 5에포크 더 학습
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * (total_epochs - resume_epoch + 1)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[user_lr, item_lr], # 💡 각 그룹의 Peak LR 지정
        total_steps=total_steps,
        pct_start=0.1,             # 전체의 10% 구간 동안 Warmup
        anneal_strategy='cos',
        div_factor=10,             # 시작 LR = max_lr / 10
        final_div_factor=100       # 종료 LR = max_lr / 100
    )
    
    best_recall_100 = last_best_recall # 9.69% 부터 시작
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")

    # 5. Training Loop (Epoch 6 ~ 10 등)
    total_epochs = resume_epoch + 9 # 예: 5에포크 더 학습
    decay_epochs = 0
    for epoch in range(resume_epoch, total_epochs + 1):
        
        avg_loss = train_user_tower_all_time(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower, # 정적 벡터 대신 모델 객체 자체를 넘김
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=hard_neg_pool_tensor,
            scheduler=scheduler,
            seq_labels = SEQ_LABELS,
            static_labels = STATIC_LABELS,
        )
        
        #decay_epochs += 1
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
        scheduler.step(current_recall_100)
        
        if current_recall_100 > best_recall_100:
            print(f"🌟 [New Best!] Recall@100 updated: {best_recall_100:.2f}% -> {current_recall_100:.2f}%")
            best_recall_100 = current_recall_100
            torch.save(user_tower.state_dict(), os.path.join(cfg.model_dir, "best_user_tower_c_detach.pth"))
            torch.save(item_tower.state_dict(), os.path.join(cfg.model_dir, "best_item_tower_c_detach.pth"))
            print("💾 Best model weights updated.")
        else:
            print(f" - (Current Best: {best_recall_100:.2f}%)")

    print("\n🎉 Resume Training Finished!")

'''