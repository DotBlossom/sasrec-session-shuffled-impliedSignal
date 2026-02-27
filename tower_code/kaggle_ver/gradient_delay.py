# 하기전에 파일 꺼내오자..


# 1
cfg.epochs = 50   
# 2    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01) 
    early_stopping = EarlyStopping(patience=9, mode='max')
# 3
        if (epoch >= 8) and (epoch - 8) % 3 == 0:
            # 💡 [신규] 에포크 시작 시 Pseudo-Online Category-Constrained HN 마이닝
            print(f"\n🔍 [Epoch {epoch} Start] Mining Category-Constrained Hard Negatives (K={HN_K})...")
            item_tower.eval() # 💥 중요: 마이닝 시에는 반드시 eval() 모드, no_grad()
            with torch.no_grad():
                all_item_embs = item_tower.get_all_embeddings()
                norm_item_embs = F.normalize(all_item_embs, p=2, dim=1)
                
                # 카테고리 내에서 Top-20 추출 (쌍둥이는 배제됨)
                epoch_hn_pool = mine_category_constrained_hard_negatives(
                    norm_item_embs, item_category_ids, k=HN_K, device=device
                )
                
                # 메모리 정리
                del all_item_embs, norm_item_embs
                torch.cuda.empty_cache()
                
            item_tower.train() # 마이닝 종료 후 훈련 모드 복귀

        elif epoch >= 8:
            print(f"\n♻️ [Epoch {epoch} Start] Using Cached Hard Negative Pool (No Mining Overheads)")
        
        else: 
            epoch_hn_pool = None
            
            
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
            hard_neg_pool_tensor=epoch_hn_pool, # 💡 방금 뽑은 최신 HN 풀 전달
            scheduler=scheduler, 
            seq_labels=SEQ_LABELS,
            static_labels=STATIC_LABELS,
        )
        # 4 pool 텐서 냅두셈;;
        # 메모리 반환
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        
        
        
        
        # 5 일므분리
        
         # 💡 기존 베이스라인을 덮어쓰지 않도록 파일명 변경 (hn_finetuned 명시)
            save_user_pth = os.path.join(cfg.model_dir, "best_user_tower_hn_finetuned_delay.pth")
            save_item_pth = os.path.join(cfg.model_dir, "best_item_tower_hn_finetuned_delay.pth")
            