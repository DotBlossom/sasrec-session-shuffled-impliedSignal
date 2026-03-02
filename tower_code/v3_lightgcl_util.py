import os
import torch
import torch.nn.functional as F

def load_aligned_lightgcl_user_embeddings(processor, lightgcl_checkpoint_path, lightgcl_cache_dir, embed_dim, device):
    """
    LightGCL 모델에서 학습된 User Embedding을 추출하여, 
    현재 SASRec Dataset의 user_id 인덱스에 맞게 정렬(Alignment)된 텐서를 반환합니다.
    """
    print(f"\n🔄 [Phase 3-1.5] Aligning LightGCL User Embeddings for KD...")
    
    # 1. SASRec 유저 수에 맞춘 빈 텐서 생성 (0번은 패딩)
    num_users = processor.num_users + 1 
    aligned_user_embs = torch.randn(num_users, embed_dim) * 0.01
    aligned_user_embs[0] = 0.0 # Padding
    
    try:
        # 2. LightGCL의 원본 ID 맵핑 로드 (문자열 ID -> LightGCL Index)
        map_path = os.path.join(lightgcl_cache_dir, "id_maps_train.pt")
        maps_cache = torch.load(map_path, map_location='cpu')
        lightgcl_user2id = maps_cache['user2id'] # { "user_string_id": 0, ... }
        
        # 3. LightGCL 체크포인트 로드
        checkpoint = torch.load(lightgcl_checkpoint_path, map_location='cpu')
        model_state = checkpoint['model_state_dict']
        
        # 모델 구현에 따라 변수명이 다를 수 있음 (일반적으로 E_u, user_emb 등)
        lightgcl_weights = model_state.get('E_u.weight', model_state.get('user_embedding.weight'))
        
        if lightgcl_weights is None:
            raise ValueError("Could not find user embedding weights in LightGCL checkpoint.")

        # 4. SASRec ID 체계로 정렬 (processor.user_ids 리스트가 있다고 가정)
        matched = 0
        # processor.user_ids는 현재 파이프라인의 string user id 리스트 (인덱스 + 1 = 텐서 인덱스)
        for i, current_id_str in enumerate(processor.user_ids):
            if current_id_str in lightgcl_user2id:
                lightgcl_idx = lightgcl_user2id[current_id_str]
                aligned_user_embs[i + 1] = lightgcl_weights[lightgcl_idx]
                matched += 1
                
        print(f"✅ LightGCL User Alignment Matched: {matched}/{len(processor.user_ids)}")
        
    except Exception as e:
        print(f"⚠️ [Warning] Failed to load LightGCL User Embeddings: {e}. Using random init for alignment.")
        
    # GPU로 올리고 그래디언트 차단 (Teacher 역할이므로 업데이트 안 됨)
    return aligned_user_embs.to(device).detach()