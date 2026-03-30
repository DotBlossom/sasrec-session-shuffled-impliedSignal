from fastapi import Depends
from sqlalchemy import select
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup, AutoModel
from pydantic import BaseModel
from typing import Any, Dict, List
import copy
import random
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from database import ProductInferenceInput, SessionLocal, TrainingItem, get_db
from utils import vocab

import os


# --- Global Configuration ---
EMBED_DIM = 128
OUTPUT_DIM_ENCODER = 128       # Encoder(Representation) 출력
OUTPUT_DIM_PROJECTOR = 128     # Projector(SimCSE Loss용) 출력
FASHION_BERT_MODEL = "bert-base-uncased" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAD_ID = vocab.PAD_ID
UNK_ID = vocab.UNK_ID

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)




# ----------------------------------------------------------------------
# 1. Models: Encoder + Projector + Wrapper
# ----------------------------------------------------------------------

# (A) Encoder: HybridItemTower 
class SEResidualBlock(nn.Module):
    """ Squeeze-and-Excitation Residual Block """
    def __init__(self, dim, dropout=0.2, expansion_factor=4):
        super().__init__()
        
        
        # 1. Feature Transformation (SwiGLU 스타일로?)
        self.block = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.LayerNorm(dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.LayerNorm(dim),
        )        
        # 2. SE-Block (Channel Attention, SE-Net 구조 반영, Gating=Relu 파트)
        # 입력 벡터의 각 차원(feature)에 대해 중요도(0~1)를 계산
        self.se_block = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        # self.act = nn.GELU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        
        # (B) SE-Attention Path
        # MLP 출력값(out)에 중요도(weight)를 곱함
        weight = self.se_block(out)
        out = out * weight
        
        return residual + out  

class DeepResidualHead(nn.Module):

    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        
        # 차원 정의 
        mid_dim = input_dim * 2      # 256
        hidden_dim = input_dim * 4   # 512
        
        # 1. Progressive Expansion 
        self.expand_layer1 = nn.Sequential(
            nn.Linear(input_dim, mid_dim),  # step 1 128 -> 256
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.expand_layer2 = nn.Sequential(
            nn.Linear(mid_dim, hidden_dim), # step 2 256 -> 512
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 2. Deep Interaction 
        self.res_blocks = nn.Sequential(
            SEResidualBlock(hidden_dim, dropout=0.2), 
            SEResidualBlock(hidden_dim, dropout=0.2)  
        )
        
        # 3. Final Projection (Compression)s

        self.final_proj = nn.Linear(hidden_dim, output_dim)
        
        # 4. Global Skip Connection (Input Shortcut) ResNet 잔차
        self.input_skip = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # --- [Step 1] Progressive Expansion ---
        m = self.expand_layer1(x) 
        h = self.expand_layer2(m)  
        
        # --- [Step 2] Feature Interaction (SE-Attention) ---
        h = self.res_blocks(h)     
        
        # --- [Step 3] Compression ---
        main_out = self.final_proj(h) 
        
        # --- [Step 4] Global Shortcut ---
        skip_out = self.input_skip(x) 
        
        return main_out + skip_out
    

class HybridItemTower(nn.Module):
    def __init__(self,
                 std_vocab_size: int,
                 num_std_fields: int,
                 embed_dim: int = 128, 
                 output_dim: int = 128):
        super().__init__()

        # A. STD Encoder
        self.std_embedding = nn.Embedding(std_vocab_size, embed_dim, padding_idx=PAD_ID)
        self.std_field_emb = nn.Parameter(torch.randn(1, num_std_fields, embed_dim))

        # A-a. 공변량 컨트롤
        self.std_ln = nn.LayerNorm(embed_dim)
        self.re_ln = nn.LayerNorm(embed_dim)
        
        # B. Fashion-BERT Setup
        print(f"Loading {FASHION_BERT_MODEL} ...")
        self.bert_config = AutoConfig.from_pretrained(FASHION_BERT_MODEL)
        self.bert_model = AutoModel.from_pretrained(FASHION_BERT_MODEL)
        bert_dim = self.bert_config.hidden_size 

        # C. RE Encoder (Embeddings from BERT -> Project)
        self.re_proj = nn.Sequential(
            nn.Linear(bert_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.re_field_position = nn.Parameter(torch.randn(1, 9, embed_dim))

        # D. Text Encoder (Product Name)
        self.text_proj = nn.Sequential(
            nn.Linear(bert_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # E. Fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False
            )
        self.head = DeepResidualHead(input_dim=embed_dim, output_dim=output_dim)

        self._debug_logged = False
    def _debug_log(self, stage: int, title: str, tensors: Dict[str, torch.Tensor]):
        """
        [내부 함수] 스테이지별로 텐서 정보를 깔끔하게 출력합니다.
        """
        if self._debug_logged:
            return

        # 헤더 출력 (Stage 0일 때)
        if stage == 0:
            print("\n" + "="*60)
            print(f"🧩 [HybridItemTower] Forward Flow Debugging Start")
            print("="*60)

        # 스테이지 타이틀
        print(f"🔹 [Stage {stage}] {title}")

        # 텐서 정보 분석 및 출력
        if tensors:
            for name, tensor in tensors.items():
                if isinstance(tensor, torch.Tensor):
                    shape_str = str(tuple(tensor.shape))
                    
                    # 값이 실수형이면 통계(평균, 표준편차)도 출력
                    if tensor.dtype in [torch.float32, torch.float16, torch.float64]:
                        mean_val = tensor.mean().item()
                        std_val = tensor.std().item()
                        info = f"Shape: {shape_str} | Mean: {mean_val:.4f} | Std: {std_val:.4f}"
                    else:
                        info = f"Shape: {shape_str} (Type: {tensor.dtype})"
                        
                    print(f"   - {name:<15}: {info}")
                else:
                    print(f"   - {name:<15}: {tensor} (Not a Tensor)")
        
        print("-" * 40)

        # 종료 처리 (Stage 99일 때)
        if stage == 99:
            print("✅ Debugging Log Finished.")
            print("="*60 + "\n")
            self._debug_logged = True
            
    def forward(self, 
                std_input: torch.Tensor,       
                re_input_ids: torch.Tensor,    
                re_attn_mask: torch.Tensor,    
                text_input_ids: torch.Tensor,  
                text_attn_mask: torch.Tensor): 
        
        B = std_input.shape[0]


        # 1. STD
        std_emb = self.std_embedding(std_input) 
        std_emb = std_emb + self.std_field_emb  
        std_emb = self.std_ln(std_emb)
        
        self._debug_log(1, "STD Embedding", {"std_emb": std_emb})


        # 2. RE (Using Fashion-BERT Word Embeddings Only)
        flat_re_ids = re_input_ids.view(-1, re_input_ids.size(-1))
        with torch.no_grad(): 
             word_embs = self.bert_model.embeddings(input_ids=flat_re_ids) 
        
        re_feats = self.re_proj(word_embs) 
        
        # 패딩 제외 Pooling
        flat_mask = re_attn_mask.view(-1, re_attn_mask.size(-1)).unsqueeze(-1) 
        sum_re = torch.sum(re_feats * flat_mask, dim=1)
        count_re = torch.clamp(flat_mask.sum(dim=1), min=1e-9)
        re_vectors = sum_re / count_re 
        
        re_vectors = re_vectors.view(B, 9, -1) 
        re_vectors = re_vectors + self.re_field_position
        re_vectors = self.re_ln(re_vectors)
        
        # [Log] Stage 2: RE Encoding
        self._debug_log(2, "RE Process", {
            "BERT Word Emb": word_embs,
            "Pooled RE Vec": re_vectors
        })

        # 3. Text (Product Name) -> Full BERT Context
        bert_out = self.bert_model(input_ids=text_input_ids, attention_mask=text_attn_mask)
        cls_token = bert_out.last_hidden_state[:, 0, :] # [CLS]
        text_vec = self.text_proj(cls_token).unsqueeze(1) 

        self._debug_log(3, "Text Encoder", {"CLS Vector": text_vec})
        
        # 4. Fusion
    
        combined_seq = torch.cat([std_emb, re_vectors, text_vec], dim=1) 
        self._debug_log(4, "Fusion Prep", {"Combined Seq": combined_seq})
        
        context_out = self.transformer(combined_seq) 
        final_vec = context_out.mean(dim=1) 
        out = self.head(final_vec)
        self._debug_log(99, "Final Projection", {"Output": out})
        
        return F.normalize(out, p=2, dim=1)
    
# (B) Projector: OptimizedItemTower for SimCSE
class OptimizedItemTower(nn.Module):
    """
    [Optimization Tower]: Projection Head for Contrastive Learning
    Representation(128) -> Hidden(128) -> Output(128) -> Normalize
    """
    def __init__(self, input_dim=OUTPUT_DIM_ENCODER, output_dim=OUTPUT_DIM_PROJECTOR):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )
        
    def forward(self, x):
        x = self.layer(x)
        return F.normalize(x, p=2, dim=1) # L2 Normalization for Cosine Similarity

# (C) Wrapper: SimCSEModelWrapper (Input Arguments 확장)
class SimCSEModelWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, projector: nn.Module):
        super().__init__()
        self.encoder = encoder      
        self.projector = projector  

    def forward(self, std, re_ids, re_mask, txt_ids, txt_mask):
        # 1. Encoder (Representation)
        # 5개의 인자를 Encoder에 전달
        enc_out = self.encoder(std, re_ids, re_mask, txt_ids, txt_mask)
        
        # 2. Projector (SimCSE Space)
        proj_out = self.projector(enc_out)
        
        return proj_out

# ----------------------------------------------------------------------
# 2. Data Structures & Dataset (Augmentation Logic)
# ----------------------------------------------------------------------


class SimCSERecSysDataset(Dataset):
    def __init__(self, products: List[TrainingItem], dropout_prob: float):
        self.products = products
        self.dropout_prob = dropout_prob
        
        # Dropout 대상이 되는 Key 그룹 정의
        self.std_keys = vocab.get_std_field_keys() # ["product_type_name", ...]
        self.re_keys = vocab.RE_FEATURE_KEYS       # ["[CAT]", "[MAT]", ...]

    def __len__(self):
        return len(self.products)
    
    def _corrupt_data(self, item: TrainingItem) -> TrainingItem:
        # 원본 데이터 복사
        new_feature_data = copy.deepcopy(item.feature_data)
        new_name = item.product_name
        

        # 확률 설정
        KEY_DROP_PROB = self.dropout_prob - 0.1    # 키 자체를 날릴 확률 (통째로 삭제)
        VALUE_DROP_PROB = self.dropout_prob  # 리스트 내부의 값을 하나씩 날릴 확률 (부분 삭제)
        
        all_keys = list(new_feature_data.keys())
        
        for k in all_keys:
            val = new_feature_data[k]
            
            # (A) 값이 리스트인 경우 (예: [MAT]: ['Cotton', 'Poly'])
            if isinstance(val, list):
                # 1. 먼저 값을 솎아냄 (Value-level)
                # 살아남은 애들만 필터링
                surviving_values = [v for v in val if random.random() > VALUE_DROP_PROB]
                
                # 2. 만약 다 지워져서 빈 리스트가 되면 -> 키 자체를 삭제
                if not surviving_values:
                    del new_feature_data[k]
                else:
                    new_feature_data[k] = surviving_values
                    
            # (B) 값이 단일 값(문자열 등)인 경우 (예: product_type_name)
            else:
                # 그냥 키 자체를 날림 (Key-level)
                if random.random() < KEY_DROP_PROB:
                    del new_feature_data[k]

        # =======================================================
        # 2. Text Deletion (단어 구멍 뚫기)
        # =======================================================
        if new_name:
            words = new_name.split()
            # 단어가 2개 이상이면 하나를 삭제 (난이도 조절)
            if len(words) > 1: 
                if random.random() < 0.5: 
                    drop_idx = random.randint(0, len(words)-1)
                    del words[drop_idx]
                    new_name = " ".join(words)
            # 단어가 1개뿐이면 가끔 아예 삭제
            elif len(words) == 1:
                if random.random() < 0.1:
                    new_name = ""

        return TrainingItem(
            product_id=item.product_id,
            feature_data=new_feature_data,
            product_name=new_name
        )

    def _apply_dropout(self, item: TrainingItem) -> TrainingItem:
        """
        Feature Dropout 수행: 딕셔너리에서 Key를 확률적으로 제거.
        제거된 Key는 나중에 Collate 단계에서 vocab.get_std_id 호출 시 
        값이 없으므로 자동으로 PAD_ID 또는 UNK_ID가 됨.
        """
        if self.dropout_prob <= 0:
            return item

        # Deep Copy to preserve original
        new_feature_data = copy.deepcopy(item.feature_data)
        new_name = item.product_name
        # 1. STD & RE Keys Dropout
        # feature_data 안에 flattened된 형태로 있다고 가정 
        all_keys = list(new_feature_data.keys())
        
        for k in all_keys:
            # 주요 Feature Key인 경우에만 드랍아웃 시도
            if (k in self.std_keys or k in self.re_keys):
                if random.random() < self.dropout_prob:
                    del new_feature_data[k]
        
        # Text Dropout (Optional): 이름 자체를 지울지 말지 결정. 
        TEXT_DROPOUT_PROB = 0.5
        
        if random.random() < TEXT_DROPOUT_PROB:
            # 빈 문자열로 만들면 Tokenizer가 [CLS], [SEP] + Padding으로 처리
            new_name = ""
            
        return TrainingItem(
            product_id=item.product_id,
            feature_data=new_feature_data,
            product_name=new_name
        )

    def __getitem__(self, idx):
        item = self.products[idx]
        
        view1 = self._corrupt_data(item)
        view2 = self._corrupt_data(item)
        
        return view1, view2

# ----------------------------------------------------------------------
# 3. Collate Function (Tokenizer & Tensor Conversion)
# ----------------------------------------------------------------------

MAX_RE_LEN = 32  
MAX_TXT_LEN = 32
FIELD_PROMPT_MAP = {

    "[CAT]": "Clothing Category",   

    "[MAT]": "Fabric Material",     
    
    "[DET]": "Garment Detail",      
    
    "[FIT]": "Clothing Fit",        
    
    "[FNC]": "Apparel Function",    
    
    "[SPC]": "Product Specification", 
    
    "[COL]": "Garment Color",       
    
    "[CTX]": "Occasion",            
    
    "[LOC]": "Body Part"            
}
class SimCSECollator:
    """
    DataLoader에서 배치를 만들 때 토크나이징 및 텐서화를 수행하는 클래스.
    Tokenizer를 매번 로드하지 않기 위해 클래스로 감쌈.
    """
    def __init__(self, tokenizer_path=FASHION_BERT_MODEL):
        print(f"🔄 Initializing Collator with {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.std_keys = vocab.get_std_field_keys()
        self.re_keys = vocab.RE_FEATURE_KEYS
        
        # Global Config  나중에 MLops에 다 주입변수행으로
        self.max_re_len = MAX_RE_LEN
        self.max_txt_len = MAX_TXT_LEN
        

        self.sep = self.tokenizer.sep_token
        
        # data flow 초기 check flag
        self._has_logged_sample = False
    def _serialize_feature_value(self, value: Any) -> str:
        """
        리스트를 [SEP] 토큰으로 구분

        """
        if not value:
            return ""
        
        if isinstance(value, list):
            valid_items = [str(v) for v in value if v]
            if not valid_items:
                return ""
            # [SEP] 1 ...
            return f" {self.sep} ".join(valid_items)
            
        return str(value)

    def process_batch_items(self, items: List[TrainingItem], is_first_view: bool = False):
        """Raw Items -> Model Input Tensors 변환"""
        
    def process_batch_items(self, items: List[TrainingItem], is_first_view: bool = False):
        batch_std = []
        batch_re_ids = []
        batch_re_masks = []
        batch_txt = [] 

        # [Log] 첫 배치의 첫 번째 아이템의 모든 RE 필드 수집
        sample_log_re = [] 

        for idx, item in enumerate(items):
            # 1. STD
            std_ids = [vocab.get_std_id(item.feature_data.get(k, "")) for k in self.std_keys]
            batch_std.append(std_ids)

            # 2. RE 
            re_vals = [self._serialize_feature_value(item.feature_data.get(k)) for k in self.re_keys]
            
            curr_re_ids = []
            curr_re_masks = []
            for i, val in enumerate(re_vals):

                final_text = val
                if val:
                    # "[MAT]" -> "Material"
                    key_code = self.re_keys[i] 
                    prompt = FIELD_PROMPT_MAP.get(key_code, key_code) 
                    
                    # 텍스트 결합: "Material: Jersey"
                    final_text = f"{prompt}: {val}"

                enc = self.tokenizer(
                    final_text, # 👈 수정된 텍스트 입력
                    padding='max_length', 
                    truncation=True, 
                    max_length=self.max_re_len, 
                    add_special_tokens=True
                )
                curr_re_ids.append(enc['input_ids'])
                curr_re_masks.append(enc['attention_mask'])
                

                if idx == 0 and is_first_view and not self._has_logged_sample:
                    if val:
                        key_name = self.re_keys[i]
                        decoded = self.tokenizer.decode(enc['input_ids'], skip_special_tokens=False)

                        sample_log_re.append(f"      - {key_name}: '{final_text}' -> {decoded[:40]}...")

            batch_re_ids.append(curr_re_ids)    
            batch_re_masks.append(curr_re_masks) 
            batch_txt.append(item.product_name)

        # Tensor Stacking
        tensor_std = torch.tensor(batch_std, dtype=torch.long)
        tensor_re_ids = torch.tensor(batch_re_ids, dtype=torch.long)
        tensor_re_mask = torch.tensor(batch_re_masks, dtype=torch.long)

        txt_enc = self.tokenizer(
            batch_txt, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_txt_len, 
            return_tensors='pt'
        )
        
        # [Log Print]
        if not self._has_logged_sample and is_first_view:
            import sys
            msg = []
            msg.append("\n" + "="*60)
            msg.append(f"🔍 [Data Integrity Check] First Batch Sample")
            msg.append(f"   1. Product Name: '{items[0].product_name}'")
            if not items[0].product_name:
                msg.append(f"      ⚠️ WARNING: Product Name is EMPTY!")
            
            msg.append(f"   2. RE Features Found ({len(sample_log_re)} fields):")
            if sample_log_re:
                msg.extend(sample_log_re)
            else:
                msg.append("      ⚠️ NO RE FEATURES FOUND (Check Key Matching)")
            
            msg.append("="*60 + "\n")
            
            final_msg = "\n".join(msg)
            try:
                from tqdm import tqdm
                tqdm.write(final_msg)
            except ImportError:
                print(final_msg, flush=True)

            self._has_logged_sample = True

        return tensor_std, tensor_re_ids, tensor_re_mask, txt_enc['input_ids'], txt_enc['attention_mask']

    def __call__(self, batch):
        view1_list = [item[0] for item in batch]
        view2_list = [item[1] for item in batch]
        return self.process_batch_items(view1_list, is_first_view=True), self.process_batch_items(view2_list, is_first_view=False)
    
# ----------------------------------------------------------------------
# 4. Training Loop Implementation
# ----------------------------------------------------------------------
def calculate_metrics(x, y, t=2):
    """
    x: 뷰 1의 임베딩 (L2 정규화 상태)
    y: 뷰 2의 임베딩 (L2 정규화 상태, x와 같은 아이템의 다른 드롭아웃 버전)
    t: Uniformity 계산 시 가중치 (보통 2 사용)
    """
    # 1. Alignment: Positive Pair 사이의 거리 (작을수록 좋음)
    # 두 벡터가 완전히 같으면 0, 멀어질수록 커집니다.
    alignment = (x - y).norm(p=2, dim=1).pow(2).mean()

    # 2. Uniformity: 전체 벡터가 얼마나 퍼져 있는지 (작을수록 좋음 = 더 균일함)
    # 가우시안 커널을 사용하여 모든 쌍 사이의 거리를 계산합니다.
    # O(N^2) 연산이므로 메모리 보호를 위해 샘플링된 배치에서만 계산하는 것이 좋습니다.
    all_embeddings = torch.cat([x, y], dim=0)
    
    # Pairwise Squared Euclidean Distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b
    # 정규화 상태이므로 ||a||^2 = 1, ||b||^2 = 1 입니다.
    dist_sq = 2 - 2 * torch.matmul(all_embeddings, all_embeddings.T)
    
    # 지수 함수를 씌워 평균을 내고 로그를 취함
    uniformity = torch.pdist(all_embeddings, p=2).pow(2).mul(-t).exp().mean().log()

    return alignment.item(), uniformity.item()


'''

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sqlalchemy import select
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# (가정) 필요한 클래스 및 함수들이 이미 import 되어 있다고 가정합니다.
# from ... import SimCSEModelWrapper, SimCSERecSysDataset, SimCSECollator, TrainingItem, calculate_metrics

def train_simcse_from_db(    
    encoder: nn.Module,       
    projector: nn.Module,
    db_session, 
    batch_size: int,
    epochs: int,
    lr: float,
    checkpoint_path: str = None,  # [New] 체크포인트 경로 (없으면 None)
    dropout_prob: float = 0.4,    # [New] 드롭아웃 확률 (기본 0.4, 파인튜닝 시 0.1~0.2 권장)
    temperature: float = 0.08     # [New] 온도 파라미터 (기본 0.08)
):
    print(f"\n🚀 [Training Start] Configuration:")
    print(f"   - Checkpoint: {checkpoint_path if checkpoint_path else 'None (New Training)'}")
    print(f"   - Dropout: {dropout_prob}")
    print(f"   - Learning Rate: {lr}")
    print(f"   - Temperature: {temperature}")
    print(f"   - Epochs: {epochs}")

    # -------------------------------------------------------
    # 0. DB 데이터 로드 (기존 로직 유지)
    # -------------------------------------------------------
    print("🚀 Fetching data from DB...")
    stmt = select(
        ProductInferenceInput.product_id, 
        ProductInferenceInput.feature_data, 
        ProductInferenceInput.product_name 
    )
    result = db_session.execute(stmt).mappings().all()

    if not result:
        print("❌ [Error] No data found in DB.")
        return

    products_list = []
    
    for row in result:
        raw_feats = dict(row['feature_data'])
        
        if 'reinforced_feature' in raw_feats:
            re_dict = raw_feats['reinforced_feature']
            if isinstance(re_dict, dict):
                for key, val in re_dict.items():
                    if key.startswith("[") and key.endswith("]"):
                        vocab_key = key
                    else:
                        vocab_key = f"[{key}]"
                    raw_feats[vocab_key] = val
                    
        base_name = row['product_name']
        product_type = raw_feats.get('product_type_name', "").strip()
        
        # Name Tagging Logic
        if base_name:
            if product_type:
                final_name = f"{base_name} (Category: {product_type})"
            else:
                final_name = base_name
        else:
            appearance = raw_feats.get('graphical_appearance_name', "").strip()
            final_name = f"{product_type} {appearance}".strip()
            if not final_name:
                final_name = "Unknown Product"
                        
        item = TrainingItem(
            product_id=str(row['product_id']), 
            feature_data=raw_feats, 
            product_name=row['product_name'] if row['product_name'] else ""
        )
        products_list.append(item)
    print(f"✅ Loaded {len(products_list)} items.")

    # -------------------------------------------------------
    # 1. Model Setup & Checkpoint Loading
    # -------------------------------------------------------
    # 모델 래핑
    model = SimCSEModelWrapper(encoder, projector)

    # [핵심] 체크포인트 로드 로직
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f"♻️ Loading Checkpoint from: {checkpoint_path}")
            # CPU로 먼저 로드하여 매핑
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # 우리가 저장한 것은 'encoder.state_dict()'이므로, 
            # wrapper 모델의 'encoder' 부분에만 로드해야 함
            try:
                model.encoder.load_state_dict(state_dict)
                print("✅ Encoder weights loaded successfully.")
            except Exception as e:
                print(f"⚠️ Warning: Failed to load strictly. Error: {e}")
                # 혹시 키 불일치가 있다면 (유연한 로딩)
                model.encoder.load_state_dict(state_dict, strict=False)
        else:
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            return

    model = model.to(DEVICE)
    model.train()

    # AMP Scaler
    use_amp = (DEVICE == "cuda")
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("⚡ [AMP] Mixed Precision Training Enabled.")     

    # -------------------------------------------------------
    # 2. Optimization
    # -------------------------------------------------------
    bert_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 
        if "bert_model" in name:
            bert_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': lr * 0.1}, # BERT는 보통 Main LR보다 작게 설정 (10분의 1)
        {'params': other_params, 'lr': lr}
    ])
    
    # -------------------------------------------------------
    # 3. Dataset & DataLoader (Dropout 적용)
    # -------------------------------------------------------
    # [변경] 인자로 받은 dropout_prob 사용
    dataset = SimCSERecSysDataset(products_list, dropout_prob=dropout_prob)
    collator = SimCSECollator()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collator,
        drop_last=True,
        num_workers=0
    )

    # 4. Scheduler
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    from torch.nn import CrossEntropyLoss
    loss_func = CrossEntropyLoss()

    # -------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------
    print("🔥 Starting Training Loop...")
    CHECK_INTERVAL = 50  
    
    # [수정] 루프 밖에서 초기화하여 set_postfix 에러 방지
    align_val, uni_val = 0.0, 0.0

    for epoch in range(epochs):
        total_loss = 0
        step = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs_v1, inputs_v2 in progress:
            inputs_v1 = [t.to(DEVICE) for t in inputs_v1]
            inputs_v2 = [t.to(DEVICE) for t in inputs_v2]
            
            optimizer.zero_grad()
            
            with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                # Forward
                emb1 = model(*inputs_v1)
                emb2 = model(*inputs_v2)
                    
                # Loss Calculation (인자로 받은 temperature 사용)
                sim_matrix = torch.matmul(emb1, emb2.T) / temperature 
                    
                labels = torch.arange(emb1.size(0)).to(DEVICE)
                loss_1 = loss_func(sim_matrix, labels) 
                loss_2 = loss_func(sim_matrix.T, labels) 
                    
                loss = (loss_1 + loss_2) / 2
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # 🌟 [Health Check Logic]
            if step % CHECK_INTERVAL == 0:
                with torch.no_grad():
                    # calculate_metrics 함수가 있다고 가정
                    # float32로 변환하여 정확도 확보
                    cur_align, cur_uni = calculate_metrics(emb1.float(), emb2.float())
                    align_val = cur_align # 외부 변수 업데이트
                    uni_val = cur_uni     # 외부 변수 업데이트
            
            # [수정] 업데이트된 외부 변수를 사용하여 항상 표시
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "align": f"{align_val:.4f}", 
                "uni": f"{uni_val:.4f}"      
            })
                    
            total_loss += loss.item()
            step += 1

        # Epoch End Summary
        if step > 0:
            avg_loss = total_loss / step
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: No batches processed.")
            avg_loss = 0.0
        
        # 파일명에 dropout 정보 등 포함하면 관리하기 쉬움
        ckpt_name = f"encoder_ep{epoch+1:02d}_loss{avg_loss:.4f}.pth"
        if checkpoint_path: # 파인튜닝 중이었다면 표시
             ckpt_name = f"ft_encoder_ep{epoch+1:02d}_loss{avg_loss:.4f}.pth"

        save_path = os.path.join(MODEL_DIR, ckpt_name)
        
        # [중요] encoder만 저장 (기존과 동일)
        torch.save(encoder.state_dict(), save_path)
        print(f"✅ Saved Checkpoint: {ckpt_name}")
        print("-" * 50)

    print("Training Finished.")




'''



def train_simcse_from_db(    
    encoder: nn.Module,       
    projector: nn.Module,
    db_session, # DB Session 객체 주입 필요
    batch_size: int,
    epochs: int,
    lr: float,
    checkpoint_path: str = None,
):
    print("🚀 Fetching data from DB...")
    stmt = select(
        ProductInferenceInput.product_id, 
        ProductInferenceInput.feature_data, 
        ProductInferenceInput.product_name 
    )
    result = db_session.execute(stmt).mappings().all()

    if not result:
        print("❌ [Error] No data found in DB.")
        return

    # load
    products_list = []
    
    for row in result:
        # DB의 원본 데이터 (수정 불가능하므로 dict로 복사)
        raw_feats = dict(row['feature_data'])
        
        # 'reinforced_feature'가 있다면 꺼내서 처리
        if 'reinforced_feature' in raw_feats:
            re_dict = raw_feats['reinforced_feature']
            # mat -> [mat] why? tsf encoder에는 상관없고, 혹시 bert embedding에서 더 잘 쓰일까봐.
            if isinstance(re_dict, dict):
                for key, val in re_dict.items():
   
                    if key.startswith("[") and key.endswith("]"):
                        vocab_key = key
                    else:
                        vocab_key = f"[{key}]"  # "MAT" -> "[MAT]"
                    

                    raw_feats[vocab_key] = val
                    
        # name tagging 
        base_name = row['product_name']
        product_type = raw_feats.get('product_type_name', "").strip() # 예: "Underwear Tights"
        
        final_name = ""

        if base_name:
            # Case A: 이름이 있는 경우 -> "원래이름 (Category: 타입명)"
            if product_type:
                final_name = f"{base_name} (Category: {product_type})"
            else:
                final_name = base_name
        else:
            # Case B: 이름이 없는 경우 (Fallback) -> "타입명 + 외형"
            appearance = raw_feats.get('graphical_appearance_name', "").strip()
            final_name = f"{product_type} {appearance}".strip()
            
            if not final_name:
                final_name = "Unknown Product"
                
                        
        item = TrainingItem(
                product_id=str(row['product_id']), 
                feature_data=raw_feats, 
                product_name=row['product_name'] if row['product_name'] else ""
            )
        products_list.append(item)
    print(f"✅ Loaded {len(products_list)} items.")
    # ▼▼▼ 디버깅 코드 
    print("\n🔎 [DEBUG] Raw DB Data Check (First Item):")
    first_row = result[0]
    print(f"   - Keys in feature_data: {list(first_row['feature_data'].keys())}")
    print(f"   - Full content: {first_row['feature_data']}")
    print("-" * 50 + "\n")
    # ▲▲▲ 여기까지 ▲▲▲
    
    
    # 1. Model Setup
    model = SimCSEModelWrapper(encoder, projector).to(DEVICE)
        # [핵심] 체크포인트 로드 로직
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f"♻️ Loading Checkpoint from: {checkpoint_path}")
            # CPU로 먼저 로드하여 매핑
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # 우리가 저장한 것은 'encoder.state_dict()'이므로, 
            # wrapper 모델의 'encoder' 부분에만 로드해야 함
            try:
                model.encoder.load_state_dict(state_dict)
                print("✅ Encoder weights loaded successfully.")
            except Exception as e:
                print(f"⚠️ Warning: Failed to load strictly. Error: {e}")
                # 혹시 키 불일치가 있다면 (유연한 로딩)
                model.encoder.load_state_dict(state_dict, strict=False)
        else:
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            return
    model = model.to(DEVICE)
    model.train()
    # 🛠️ [AMP] Scaler 초기화 (GPU 사용 시)
    use_amp = (DEVICE == "cuda")
    scaler = GradScaler(enabled=use_amp)
    
    if use_amp:
        print("⚡ [AMP] Mixed Precision Training Enabled.")     
    bert_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 
            

        if "bert_model" in name:
            bert_params.append(param)
        else:
            other_params.append(param)
    
    
    # 2. Optimization

    optimizer = AdamW([
        {
            'params': bert_params, 
            'lr': 1e-5  
        },
        {
            'params': other_params, 
            'lr': lr
               
        }
    ])
    
    # 3. Dataset & DataLoader
    dataset = SimCSERecSysDataset(products_list, dropout_prob=0.2)
    collator = SimCSECollator() # Initialize Tokenizer once
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collator, # Use the class instance
        drop_last=True,
        num_workers=0 # win - 멀티프로세싱 시 Tokenizer 이슈 주의
    )

    # 4. Scheduler
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # Loss Function (Contrastive)
    # in-batch negatives 활용
    from torch.nn import CrossEntropyLoss
    loss_func = CrossEntropyLoss()

    print("🔥 Starting Training Loop...")
    CHECK_INTERVAL = 50  # 100 스텝마다 지표 계산 및 출력
    
    align_val, uni_val = 0.0, 0.0
    for epoch in range(epochs):
        total_loss = 0
        step = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Unpack Collator Outputs (5 tensors per view)
        for inputs_v1, inputs_v2 in progress:
            # Move to Device
            inputs_v1 = [t.to(DEVICE) for t in inputs_v1]
            inputs_v2 = [t.to(DEVICE) for t in inputs_v2]
            
            optimizer.zero_grad()
            
            
            with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    # Forward
                emb1 = model(*inputs_v1)
                emb2 = model(*inputs_v2)
                    
                    # Loss Calculation
                temperature = 0.08
                sim_matrix = torch.matmul(emb1, emb2.T) / temperature 
                    
                labels = torch.arange(emb1.size(0)).to(DEVICE)
                loss_1 = loss_func(sim_matrix, labels) 
                loss_2 = loss_func(sim_matrix.T, labels) 
                    
                loss = (loss_1 + loss_2) / 2
                
            scaler.scale(loss).backward()  # loss scaling
            scaler.step(optimizer)         # optimizer step with scaler
            scaler.update()                # update scaler factor
                    
            scheduler.step()
            
            # 🌟 [Health Check Logic 추가]
            if step % CHECK_INTERVAL == 0:
            # Metric 계산 (검증 시에는 grad 계산 제외)
                with torch.no_grad():
                    align, uni = calculate_metrics(emb1.float(), emb2.float())
            
                # tqdm 상태창 업데이트 (Loss와 함께 표시)
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "align": f"{align:.4f}",  # 0에 가까울수록 좋음
                "uni": f"{uni:.4f}"       # 낮을수록(음수값이 클수록) 잘 퍼져 있음
            })

                    
            total_loss += loss.item()
            step += 1


        if step > 0:
            avg_loss = total_loss / step
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: No batches processed.")
        
        ckpt_name = f"encoder_ep{epoch+1:02d}_loss{avg_loss:.4f}.pth"
        save_path = os.path.join(MODEL_DIR, ckpt_name)
        
        # encoder only 
        torch.save(encoder.state_dict(), save_path)
        print(f"✅ Saved Checkpoint: {ckpt_name}")
        
  
        # torch.save(projector.state_dict(), os.path.join(MODEL_DIR, f"projector_ep{epoch+1:02d}.pth"))
        
        print("-" * 50)
        # =========================================================

    print("Training Finished.")



import torch
import torch.nn.functional as F
import copy
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

class EmbeddingEvaluator:
    def __init__(self, model: nn.Module, collator, device: str):
        self.model = model.to(device)
        self.model.eval()  # 평가 모드 (Dropout 등 비활성화)
        self.collator = collator
        self.device = device
        
        # 모델은 Wrapper가 아닌 학습된 Encoder 자체를 바로 사용한다고 가정
        # 만약 Wrapper 구조라면 self.model = model.encoder 로 세팅 추천
        
    @torch.no_grad()
    def get_embeddings(self, items: List[TrainingItem], batch_size: int = 256) -> torch.Tensor:
        """아이템 리스트를 받아 전체 임베딩 텐서(N, D)를 반환합니다."""
        all_embeddings = []
        
        for i in range(0, len(items), batch_size):
            batch_items = items[i : i + batch_size]
            # collator를 통해 텐서화
            std, re_ids, re_mask, txt_ids, txt_mask = self.collator.process_batch_items(batch_items)
            
            std = std.to(self.device)
            re_ids = re_ids.to(self.device)
            re_mask = re_mask.to(self.device)
            txt_ids = txt_ids.to(self.device)
            txt_mask = txt_mask.to(self.device)
            
            # Encoder 통과 후 L2 정규화
            emb = self.model(std, re_ids, re_mask, txt_ids, txt_mask)
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu())
            
        return torch.cat(all_embeddings, dim=0)

    def generate_case_data(self, original_items: List[TrainingItem], mode: str, drop_key: str = None) -> List[TrainingItem]:
        """목적에 맞게 데이터를 변형하여 Case를 생성합니다."""
        modified_items = []
        for item in original_items:
            new_feats = copy.deepcopy(item.feature_data)
            new_name = item.product_name
            
            if mode == "case1":
                # [Case 1] 카테고리 등 기본 필드(STD)만 남기고 메타데이터(RE, Text) 전부 삭제
                for k in list(new_feats.keys()):
                    if k in vocab.RE_FEATURE_KEYS:
                        del new_feats[k]
                new_name = ""
                
            elif mode == "case2":
                # [Case 2] 원본 데이터 그대로 사용
                pass 
                
            elif mode == "ablation":
                # [Ablation] Case 2에서 특정 필드(drop_key)만 마스킹
                if drop_key in new_feats:
                    del new_feats[drop_key]
                    
            #modified_items.append(TrainingItem(item.product_id, new_feats, new_name))
            modified_items.append(TrainingItem(
                product_id=item.product_id, 
                feature_data=new_feats, 
                product_name=new_name
            ))
        return modified_items

    # ==========================================
    # 가설 1. 앵커 정렬도 (Anchor Alignment)
    # ==========================================
    def test_anchor_alignment(self, v1_emb: torch.Tensor, v2_emb: torch.Tensor):
        print("\n" + "="*50)
        print("🎯 [Hypothesis 1] Anchor Alignment (Case 1 vs Case 2)")
        print("="*50)
        
        # 동일 아이템 간의 코사인 유사도 평균 계산
        cosine_sims = F.cosine_similarity(v1_emb, v2_emb, dim=1)
        mean_sim = cosine_sims.mean().item()
        
        print(f"👉 V1 & V2 Mean Cosine Similarity: {mean_sim:.4f}")
        if mean_sim > 0.8:
            print("✅ 증명 완료: Case 2가 추가 정보로 확장되었으나, Case 1의 기존 앵커(우주)를 안정적으로 유지하고 있습니다.")
        else:
            print("⚠️ 경고: 거리가 너무 멉니다. 텍스트 정보가 기존 뼈대를 과도하게 왜곡시키고 있을 수 있습니다.")

    # ==========================================
    # 가설 2. 군집 내 분산 vs 군집 간 분산
    # ==========================================
    def test_cluster_variance(self, items: List[TrainingItem], v1_emb: torch.Tensor, v2_emb: torch.Tensor, cluster_key: str = "product_type_name"):
        print("\n" + "="*50)
        print(f"🎯 [Hypothesis 2] Intra/Inter Cluster Variance (기준: {cluster_key})")
        print("="*50)
        
        # 카테고리별 인덱스 수집
        cluster_indices = defaultdict(list)
        for idx, item in enumerate(items):
            cat = item.feature_data.get(cluster_key)
            if cat:
                cluster_indices[cat].append(idx)
                
        # 10개 이상의 아이템이 있는 의미 있는 군집만 필터링
        valid_clusters = {k: v for k, v in cluster_indices.items() if len(v) >= 10}
        print(f"분석 대상 군집 수: {len(valid_clusters)}개")
        
        def calc_cluster_stats(embeddings):
            intra_variances = []
            centroids = []
            
            for cat, indices in valid_clusters.items():
                cluster_vecs = embeddings[indices] # (N_c, D)
                centroid = cluster_vecs.mean(dim=0, keepdim=True) # (1, D)
                
                # 군집 내 분산 (중심점과의 평균 거리)
                var = F.pairwise_distance(cluster_vecs, centroid).pow(2).mean().item()
                intra_variances.append(var)
                centroids.append(centroid)
                
            centroids = torch.cat(centroids, dim=0) # (K, D)
            # 군집 간 분산 (중심점들 사이의 평균 거리)
            inter_dist = torch.pdist(centroids, p=2).pow(2).mean().item()
            
            return np.mean(intra_variances), inter_dist

        v1_intra, v1_inter = calc_cluster_stats(v1_emb)
        v2_intra, v2_inter = calc_cluster_stats(v2_emb)
        
        print(f"🔹 [Case 1] 군집 내 분산: {v1_intra:.4f} | 군집 간 거리: {v1_inter:.4f}")
        print(f"🔹 [Case 2] 군집 내 분산: {v2_intra:.4f} | 군집 간 거리: {v2_inter:.4f}")
        
        if v2_intra > v1_intra:
            print("✅ 군집 내 분산 증가: 텍스트 추가로 인해 같은 카테고리 내에서도 미세한 디테일 공간이 형성되었습니다.")
        if abs(v2_inter - v1_inter) / v1_inter < 0.2: 
            print("✅ 군집 간 거리 유지: 카테고리 간의 거시적 분류 체계가 무너지지 않고 잘 보존되었습니다.")

    # ==========================================
    # 가설 3. 메타데이터 필드별 기여도 (Ablation)
    # ==========================================
    def test_ablation_variance(self, original_items: List[TrainingItem], case2_emb: torch.Tensor):
        print("\n" + "="*50)
        print("🎯 [Hypothesis 3] Field Ablation Variance Analysis")
        print("="*50)
        
        # Case 2의 전체 분산 계산 (모든 벡터의 평균점으로부터의 분산)
        global_centroid = case2_emb.mean(dim=0, keepdim=True)
        base_variance = F.pairwise_distance(case2_emb, global_centroid).pow(2).mean().item()
        print(f"🔹 [Base Case 2] 총 분산: {base_variance:.4f}")
        print("-" * 30)
        
        # 각 필드를 하나씩 지워가며 분산 하락폭 측정
        contributions = {}
        for key in vocab.RE_FEATURE_KEYS:
            ablated_items = self.generate_case_data(original_items, mode="ablation", drop_key=key)
            ablated_emb = self.get_embeddings(ablated_items)
            
            ablated_centroid = ablated_emb.mean(dim=0, keepdim=True)
            ablated_variance = F.pairwise_distance(ablated_emb, ablated_centroid).pow(2).mean().item()
            
            variance_drop = base_variance - ablated_variance
            contributions[key] = variance_drop
            print(f"   - W/O {key:<6} : 분산 {ablated_variance:.4f} (기여도: {variance_drop:+.4f})")
            
        # 가장 기여도가 높은 필드 찾기
        best_field = max(contributions, key=contributions.get)
        print("-" * 30)
        print(f"✅ 결론: 뼈대 주변의 공간을 가장 입체적으로 부풀려준(정보 기여도가 높은) 필드는 '{best_field}' 입니다.")


# ----------------------------------------------------------------------
# 🚀 실행 스크립트 (Main Execution)
# ----------------------------------------------------------------------
def run_evaluation(encoder, dataloader, db_items, device):
    print("Initializing Evaluator...")
    evaluator = EmbeddingEvaluator(model=encoder, collator=SimCSECollator(), device=device)
    
    # 1. 테스트할 데이터 준비 (예: DB에서 불러온 처음 10000개 샘플링)
    test_items = db_items[:10000] 
    
    # 2. Case 데이터 생성
    case1_items = evaluator.generate_case_data(test_items, mode="case1")
    case2_items = evaluator.generate_case_data(test_items, mode="case2")
    
    # 3. 임베딩 일괄 추출 (추론)
    print("Extracting Case 1 Embeddings...")
    v1_embeddings = evaluator.get_embeddings(case1_items)
    
    print("Extracting Case 2 Embeddings...")
    v2_embeddings = evaluator.get_embeddings(case2_items)
    
    # 4. 가설 검증 실행
    evaluator.test_anchor_alignment(v1_embeddings, v2_embeddings)
    evaluator.test_cluster_variance(test_items, v1_embeddings, v2_embeddings, cluster_key="product_type_name")
    evaluator.test_ablation_variance(test_items, v2_embeddings)

# 실행 예시: 
# run_evaluation(model.encoder, dataloader, products_list, DEVICE)
def evaluate_trained_model(
    encoder: nn.Module, 
    db_session,          # DB 세션 객체
    checkpoint_path: str # 저장된 Encoder 가중치 경로 (예: "encoder_ep10_loss0.05.pth")
):
    DEVICE = next(encoder.parameters()).device # 모델의 디바이스 확인
    print(f"🚀 Starting Evaluation Pipeline on {DEVICE}...")

    # ==========================================
    # 1. DB에서 평가용 데이터 로드 (기존 학습 코드와 동일)
    # ==========================================
    stmt = select(
        ProductInferenceInput.product_id, 
        ProductInferenceInput.feature_data, 
        ProductInferenceInput.product_name 
    )
    result = db_session.execute(stmt).mappings().all()

    if not result:
        print("❌ [Error] No data found in DB.")
        return

    products_list = []
    for row in result:
        raw_feats = dict(row['feature_data'])
        
        if 'reinforced_feature' in raw_feats:
            re_dict = raw_feats['reinforced_feature']
            if isinstance(re_dict, dict):
                for key, val in re_dict.items():
                    vocab_key = key if (key.startswith("[") and key.endswith("]")) else f"[{key}]"
                    raw_feats[vocab_key] = val
                    
        base_name = row['product_name']
        product_type = raw_feats.get('product_type_name', "").strip()
        
        item = TrainingItem(
            product_id=str(row['product_id']), 
            feature_data=raw_feats, 
            product_name=base_name if base_name else ""
        )
        products_list.append(item)
        
    print(f"✅ Loaded {len(products_list)} items for evaluation.")

    # ==========================================
    # 2. 체크포인트(학습된 가중치) 로드
    # ==========================================
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"♻️ Loading Checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        
        try:
            encoder.load_state_dict(state_dict)
            print("✅ Encoder weights loaded successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Strict load failed. Trying non-strict. Error: {e}")
            encoder.load_state_dict(state_dict, strict=False)
    else:
        print(f"❌ [Error] Checkpoint not found at {checkpoint_path}. Exiting.")
        return

    encoder.eval() # 평가 모드 전환 (Dropout 비활성화)

    # ==========================================
    # 3. 평가 파이프라인 실행
    # ==========================================
    print("\n" + "="*50)
    print("🧠 Initializing Embedding Evaluator...")
    print("="*50)
    
    # 평가기 초기화 (Collator는 평가 시 토크나이징을 위해 필요함)
    collator = SimCSECollator()
    evaluator = EmbeddingEvaluator(model=encoder, collator=collator, device=DEVICE)
    
    # 전체 데이터가 너무 많으면 메모리 부족이 발생할 수 있으므로, 
    # 대표성을 띄는 샘플 1만 개 정도만 추출해서 평가하는 것을 권장합니다.
    test_items = products_list[:10000] 
    print(f"🧪 Using {len(test_items)} items for variance/alignment testing.")
    
    # Case 데이터 생성 (앞서 만든 EmbeddingEvaluator 내부 함수)
    print("📦 Generating Case 1 (Standard Only) Data...")
    case1_items = evaluator.generate_case_data(test_items, mode="case1")
    
    print("📦 Generating Case 2 (Full Metadata) Data...")
    case2_items = evaluator.generate_case_data(test_items, mode="case2")
    
    # 임베딩 추출
    print("🧮 Extracting Case 1 Embeddings...")
    v1_embeddings = evaluator.get_embeddings(case1_items, batch_size=128) # 메모리에 맞게 배치 조절
    
    print("🧮 Extracting Case 2 Embeddings...")
    v2_embeddings = evaluator.get_embeddings(case2_items, batch_size=128)
    
    # ------------------------------------------
    # 4. 가설 검증 결과 출력
    # ------------------------------------------
    # 가설 1: 앵커 정렬도
    evaluator.test_anchor_alignment(v1_embeddings, v2_embeddings)
    
    # 가설 2: 군집 내/간 분산
    evaluator.test_cluster_variance(test_items, v1_embeddings, v2_embeddings, cluster_key="product_type_name")
    
    # 가설 3: 필드별 기여도 (Ablation)
    evaluator.test_ablation_variance(test_items, v2_embeddings)
    
    print("\n🎉 All evaluations completed successfully!")

if __name__ == "__main__":
    # 1. 모델 아키텍처 초기화 (기존 코드 활용)
    encoder = HybridItemTower(
        std_vocab_size=vocab.get_std_vocab_size(), 
        num_std_fields=len(vocab.get_std_field_keys()),
        embed_dim=128, 
        output_dim=128
    ).to(DEVICE)

    # 2. 저장해둔 가중치 파일 경로 지정
    CHECKPOINT_FILE = os.path.join(MODEL_DIR, "encoder_ep03_loss0.8129.pth") # 실제 파일명으로 변경
    with SessionLocal() as db_session:
        # 3. 평가 전용 함수 실행!
        evaluate_trained_model(
            encoder=encoder,
            db_session=db_session, # 생성된 실제 세션 객체 주입
            checkpoint_path=CHECKPOINT_FILE
        )