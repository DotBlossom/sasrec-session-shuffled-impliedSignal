import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


# ================================================================
# 1. Delta Tracker: User Tower 학습 전/후 스냅샷 관리
# ================================================================
class ItemEmbeddingDeltaTracker:
    """
    item_matrix의 Before/After를 추적하여
    SimCSE Encoder 재학습용 Delta 신호를 생성
    """
    def __init__(self, item_tower: nn.Module, min_delta_norm: float = 1e-4):
        self.item_tower = item_tower
        self.min_delta_norm = min_delta_norm  # 의미없는 미세 변화 필터링
        self.snapshot: torch.Tensor | None = None

    def take_snapshot(self):
        """User Tower 학습 전 호출"""
        with torch.no_grad():
            self.snapshot = self.item_tower.item_matrix.weight.data.clone().cpu()
        print(f"📸 Snapshot taken. Shape: {self.snapshot.shape}")

    def compute_delta(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        User Tower 학습 후 호출
        
        Returns:
            active_ids:    변화가 유의미한 아이템 ID 목록 (N,)
            delta_targets: 해당 아이템의 목표 벡터 = old + Δ  (N, D)
        """
        assert self.snapshot is not None, "take_snapshot() 먼저 호출 필요"

        with torch.no_grad():
            new_weights = self.item_tower.item_matrix.weight.data.clone().cpu()
            delta = new_weights - self.snapshot  # (num_items+1, D)

            # Δ의 L2 norm으로 유의미하게 변한 아이템만 필터링
            delta_norms = delta.norm(dim=-1)  # (num_items+1,)
            active_mask = delta_norms > self.min_delta_norm
            active_mask[0] = False  # padding_idx 제외

            active_ids = active_mask.nonzero(as_tuple=True)[0]  # (N,)
            delta_targets = new_weights[active_ids]  # (N, D) = old + Δ

        n_active = len(active_ids)
        n_total = active_mask.shape[0] - 1
        print(f"📊 Delta computed: {n_active}/{n_total} items changed "
              f"(mean Δ norm: {delta_norms[active_mask].mean():.4f})")

        return active_ids, delta_targets

    def get_delta_stats(self) -> dict:
        """Δ 분포 통계 (모니터링용)"""
        with torch.no_grad():
            new_weights = self.item_tower.item_matrix.weight.data.clone().cpu()
            delta = new_weights - self.snapshot
            norms = delta.norm(dim=-1)[1:]  # padding 제외
        return {
            "mean": norms.mean().item(),
            "max": norms.max().item(),
            "p95": norms.quantile(0.95).item(),
            "changed_ratio": (norms > self.min_delta_norm).float().mean().item(),
        }


# ================================================================
# 2. Delta Dataset: (raw_features, target_vec) 쌍
# ================================================================
class DeltaDistillDataset(Dataset):
    """
    active_ids에 해당하는 raw_features와 delta_targets를 묶은 Dataset
    
    raw_feature_store: {item_id(int) -> dict of tensors}
        예: {
            42: {
                'std': tensor(...),
                're_ids': tensor(...),
                're_mask': tensor(...),
                'txt_ids': tensor(...),
                'txt_mask': tensor(...),
            }
        }
    """
    def __init__(
        self,
        active_ids: torch.Tensor,
        delta_targets: torch.Tensor,
        raw_feature_store: dict,
    ):
        # raw_feature_store에 있는 아이템만 사용
        valid_mask = torch.tensor([
            int(i.item()) in raw_feature_store for i in active_ids
        ])
        self.ids = active_ids[valid_mask]
        self.targets = delta_targets[valid_mask]
        self.store = raw_feature_store

        skipped = (~valid_mask).sum().item()
        if skipped:
            print(f"⚠️ {skipped} items skipped (no raw features in store)")
        print(f"✅ DeltaDistillDataset: {len(self.ids)} items ready")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item_id = int(self.ids[idx].item())
        raw = self.store[item_id]
        return {
            "std":      raw["std"],
            "re_ids":   raw["re_ids"],
            "re_mask":  raw["re_mask"],
            "txt_ids":  raw["txt_ids"],
            "txt_mask": raw["txt_mask"],
        }, self.targets[idx]


# ================================================================
# 3. Delta Distillation Trainer
# ================================================================
class DeltaDistillationTrainer:
    """
    User Tower 학습으로 생긴 Δ를 SimCSE Encoder에 역으로 주입
    """
    def __init__(
        self,
        simcse_encoder: nn.Module,
        device: str,
        lr: float = 5e-6,
        cosine_weight: float = 0.5,   # Cosine loss 비중
        mse_weight: float = 0.5,      # MSE loss 비중
    ):
        self.encoder = simcse_encoder.to(device)
        self.device = device
        self.cosine_w = cosine_weight
        self.mse_w = mse_weight

        self.optimizer = AdamW(self.encoder.parameters(), lr=lr)

    def _compute_loss(
        self,
        pred: torch.Tensor,   # (B, D) encoder 출력
        target: torch.Tensor, # (B, D) delta_target
    ) -> dict:
        """
        방향(Cosine) + 크기(MSE) 동시 최적화
        방향: "어디로 가야 하나"
        크기: "얼마나 가야 하나"
        """
        # L1: Cosine similarity loss (방향)
        cos_loss = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()

        # L2: MSE loss (절대 위치)
        mse_loss = F.mse_loss(pred, target)

        total = self.cosine_w * cos_loss + self.mse_w * mse_loss
        return {"total": total, "cosine": cos_loss, "mse": mse_loss}

    def train_one_round(
        self,
        dataset: DeltaDistillDataset,
        batch_size: int = 64,
        epochs: int = 3,
    ):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.encoder.train()
        print(f"\n🔁 Delta Distillation: {epochs} epochs, {len(loader)} steps/epoch")

        for epoch in range(epochs):
            total_loss, total_cos, total_mse = 0., 0., 0.
            n_steps = 0

            for raw_batch, target_vecs in loader:
                target_vecs = target_vecs.to(self.device)

                std      = raw_batch["std"].to(self.device)
                re_ids   = raw_batch["re_ids"].to(self.device)
                re_mask  = raw_batch["re_mask"].to(self.device)
                txt_ids  = raw_batch["txt_ids"].to(self.device)
                txt_mask = raw_batch["txt_mask"].to(self.device)

                self.optimizer.zero_grad()

                pred = self.encoder(std, re_ids, re_mask, txt_ids, txt_mask)
                losses = self._compute_loss(pred, target_vecs)

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                self.optimizer.step()

                total_loss += losses["total"].item()
                total_cos  += losses["cosine"].item()
                total_mse  += losses["mse"].item()
                n_steps += 1

            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"Loss: {total_loss/n_steps:.4f} | "
                  f"Cos: {total_cos/n_steps:.4f} | "
                  f"MSE: {total_mse/n_steps:.4f}")

        print("✅ Delta Distillation round complete.")


# ================================================================
# 4. 전체 파이프라인 통합 (User Tower 학습 루프와 연동)
# ================================================================
def run_alternating_training(
    user_tower,
    item_tower,
    simcse_encoder,
    user_train_fn,          # 기존 User Tower 학습 함수
    raw_feature_store,      # {item_id -> raw_features dict}
    n_rounds: int = 3,      # 교대 학습 반복 횟수
    distill_epochs: int = 3,
    distill_lr: float = 5e-6,
    device: str = "cuda",
):
    """
    [Round 1]
      1. Snapshot
      2. User Tower 학습 → item_matrix 변화
      3. Delta 계산
      4. SimCSE Encoder에 Delta 역주입
    [Round 2] 위 반복 (수렴까지)
    """
    tracker = ItemEmbeddingDeltaTracker(item_tower, min_delta_norm=1e-4)
    distill_trainer = DeltaDistillationTrainer(simcse_encoder, device, lr=distill_lr)

    for round_idx in range(n_rounds):
        print(f"\n{'='*60}")
        print(f"🔄 Alternating Round {round_idx+1}/{n_rounds}")
        print(f"{'='*60}")

        # Step 1: 스냅샷
        tracker.take_snapshot()

        # Step 2: User Tower 학습
        print("\n[Step 1] Training User Tower...")
        user_train_fn(user_tower, item_tower)

        # Step 3: Delta 통계 확인
        stats = tracker.get_delta_stats()
        print(f"\n[Step 2] Delta Stats: {stats}")

        # 변화량이 너무 작으면 조기 종료
        if stats["mean"] < 1e-5:
            print("⛔ Delta too small. Early stopping.")
            break

        # Step 4: Delta → Dataset → Encoder 재학습
        active_ids, delta_targets = tracker.compute_delta()
        distill_dataset = DeltaDistillDataset(active_ids, delta_targets, raw_feature_store)

        print(f"\n[Step 3] Distilling Delta back to SimCSE Encoder...")
        distill_trainer.train_one_round(
            distill_dataset,
            batch_size=128,
            epochs=distill_epochs,
        )

        # Step 5: 업데이트된 Encoder로 item_matrix 갱신 (선택적)
        if round_idx < n_rounds - 1:
            _refresh_item_matrix_from_encoder(
                item_tower, simcse_encoder, raw_feature_store, device
            )
            print("🔃 item_matrix refreshed from updated Encoder")

    return user_tower, item_tower, simcse_encoder


def _refresh_item_matrix_from_encoder(item_tower, encoder, raw_feature_store, device):
    """Encoder 재학습 후 item_matrix를 새 벡터로 갱신"""
    encoder.eval()
    with torch.no_grad():
        for item_id, raw in raw_feature_store.items():
            vec = encoder(
                raw["std"].unsqueeze(0).to(device),
                raw["re_ids"].unsqueeze(0).to(device),
                raw["re_mask"].unsqueeze(0).to(device),
                raw["txt_ids"].unsqueeze(0).to(device),
                raw["txt_mask"].unsqueeze(0).to(device),
            ).squeeze(0)
            item_tower.item_matrix.weight.data[item_id] = vec
    encoder.train()
