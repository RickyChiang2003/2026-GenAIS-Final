import os
import sys
import json
import logging
import random
import copy
import torch
import gc

from collections import defaultdict
from typing import Optional
from tqdm import tqdm

from ..utils.log_utils import Logger
from ..utils import model_utils
from ..models import WhiteBoxModelBase, ModelBase
from .attacker_base import AttackerBase
from ..seed import SeedRandom
from ..mutation.gradient.entity_gradient import MutationEntityGradient
from ..selector import ReferenceLossSelector
from ..metrics.Evaluator.Evaluator_PrefixExactMatch import EvaluatorPrefixExactMatch
from ..datasets import JailbreakDataset, Instance


def convert_list_to_slice(pii_slice_dict):
    """将每个pii原始的列表切片转化为slice切片"""
    for pii, pii_slice_list in pii_slice_dict.items():
        pii_slice_dict[pii] = [slice(pii[0], pii[1]) for pii in pii_slice_list]
    return pii_slice_dict

def flatten_list(token_id_slices_list):
    token_id_list = []
    for token_id_slice in token_id_slices_list:
        for i in range(token_id_slice[0], token_id_slice[1]):
            token_id_list.append(i)
    return token_id_list

def slice_to_token_id(model, prompt, slice_list, replace_all=False):
    """将每个字符切片转化为模型编码后的token切片"""
    assert isinstance(model, WhiteBoxModelBase)

    idx_and_slices = list(enumerate(slice_list))
    idx_and_slices = sorted(idx_and_slices, key=lambda x: x[1])

    splited_text = []
    cur = 0
    for sl_idx, sl in idx_and_slices:
        splited_text.append((prompt[cur: sl.start], None))
        splited_text.append((prompt[sl.start: sl.stop], sl_idx))
        cur = sl.stop
    splited_text.append((prompt[cur:], None))
    splited_text = [s for s in splited_text if s[0] != '' or s[1] is not None]

    ans_input_ids = model.batch_encode(prompt, return_tensors='pt')['input_ids'].to(model.device)[:, 1:]

    ans_slices = []
    splited_text_idx = 0
    start = 0
    cur = 0
    while cur < ans_input_ids.size(1):
        text_seg = model.batch_decode(ans_input_ids[:, start: cur + 1])[0]
        if splited_text[splited_text_idx][0] == '':
            ans_slices.append((splited_text[splited_text_idx][1], slice(start, start)))
            splited_text_idx += 1
        elif splited_text[splited_text_idx][0].replace(' ', '') in text_seg.replace(' ', '') or splited_text[splited_text_idx][0].replace(', ', '').replace(' ', '') in text_seg.replace(' ', ''):
            ans_slices.append((splited_text[splited_text_idx][1], slice(start, cur + 1)))
            splited_text_idx += 1
            start = cur + 1
            cur += 1
        else:
            cur += 1
    if splited_text_idx < len(splited_text):
        ans_slices.append((splited_text[splited_text_idx][1], slice(start, cur)))

    token_id_list = [item for item in ans_slices if item[0] is not None]
    token_id_list = [(sl.start, sl.stop) for _, sl in sorted(token_id_list, key=lambda x: x[0])]
    assert len(token_id_list) == len(slice_list)

    token_id_list = flatten_list(token_id_list)
    if replace_all:
        token_id_list = flatten_list([[0, ans_input_ids.size(1)]])
    return token_id_list

class PIG(AttackerBase):
    def __init__(
            self,
            attack_model: WhiteBoxModelBase,
            target_model: ModelBase,
            eval_model: ModelBase,
            jailbreak_datasets: JailbreakDataset,
            save_path,
            dataset_name,
            jailbreak_prompt_length: int = 20,
            num_turb_sample: int = 512,
            batchsize: int = 8,
            top_k: int = 256,
            max_num_iter: int = 300,
            is_universal: bool = False
    ):
        super().__init__(attack_model, target_model, None, jailbreak_datasets)

        if batchsize is None:
            batchsize = num_turb_sample

        self.attack_model = attack_model
        self.mutator = MutationEntityGradient(
            dataset_name=dataset_name,
            attack_model=attack_model,
            num_turb_sample=num_turb_sample,
            top_k=top_k,
            is_universal=is_universal
        )
        self.selector = ReferenceLossSelector(attack_model, batch_size=batchsize, is_universal=is_universal)
        self.evaluator = EvaluatorPrefixExactMatch()
        self.max_num_iter = max_num_iter

        self.save_path = save_path[:save_path.rfind('.jsonl')]
        self.dataset_name = dataset_name

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.logger = Logger()

    def single_attack(self, instance: Instance):
        dataset = self.jailbreak_datasets
        self.jailbreak_datasets = JailbreakDataset([instance])
        self.attack()
        ans = self.jailbreak_datasets
        self.jailbreak_datasets = dataset
        return ans

    def attack(self):
        logging.info("Jailbreak started! Mode: Instance-by-Instance with RAILS Hybrid Selection")
        
        # 定義 RAILS 的 K 個候選數量
        K = 20
        
        # === 確保儲存目錄與清空舊有的檔案 ===
        # 使用 Append ('a') 模式寫入可防止中斷遺失資料，因此開始前先建好空檔案。
        for i in range(K):
            open(self.save_path + f'/final_test_shot_{i}.jsonl', 'w').close()

        # 將完整資料集轉為 List，準備逐筆處理
        original_instances = list(self.jailbreak_datasets)
        if len(original_instances) > 50:
            random.seed(42) 
            original_instances = random.sample(original_instances, 50)
            logging.info(f"Dataset too large. Randomly sampled 50 instances for evaluation.")
        
        try:
            for data_idx, base_instance in enumerate(original_instances):
                logging.info(f"========== Processing Instance {data_idx + 1}/{len(original_instances)} ==========")
                
                # --- 1. 建立這單筆資料專屬的 PII 與 Token slice ---
                one_instance_pii_token_id_dict = defaultdict(list)
                base_instance.pii_slice_dict = dict(filter(lambda x: x[1] is not None, base_instance.pii_slice_dict.items()))
                base_instance.pii_slice_dict = convert_list_to_slice(base_instance.pii_slice_dict)
                for key, value in base_instance.pii_slice_dict.items():
                    one_instance_pii_token_id_dict['pii_token_id_list'].extend(
                        slice_to_token_id(self.attack_model, base_instance.context, value)
                    )
                one_instance_pii_token_id_dict['pii_token_id_list'] = list(set(one_instance_pii_token_id_dict['pii_token_id_list']))
                all_instance_pii_token_id_dict = {base_instance['idx']: one_instance_pii_token_id_dict}

                input_ids, _, _, _ = model_utils.encode_trace(
                    self.attack_model,
                    base_instance.query,
                    f'{base_instance.context} {{query}}',
                    base_instance.reference_responses[0]
                )
                if base_instance.jailbreak_prompt is None:
                    base_instance.jailbreak_prompt = f'{base_instance.context} {{query}}'
                base_instance.token_id_length = len(input_ids[0])

                # --- 2. 針對該筆資料進行 Epoch 優化 (Phase 1) ---
                unbreaked_dataset = JailbreakDataset([base_instance])
                history_records = []
                best_loss = float('inf')
                patience = 50
                patience_counter = 0
                current_optimized_instance = base_instance 

                for epoch in tqdm(range(self.max_num_iter), desc=f"Optimizing Inst {data_idx+1}"):
                    # 突變與篩選 (此時只計算 1 筆資料的 mutants)
                    unbreaked_dataset = self.mutator(unbreaked_dataset, all_instance_pii_token_id_dict)
                    unbreaked_dataset = self.selector.select(unbreaked_dataset)
                    
                    # 紀錄最佳結果與 Loss 軌跡
                    current_optimized_instance = unbreaked_dataset[0]
                    prompt = current_optimized_instance.jailbreak_prompt.replace('{query}', current_optimized_instance.query)
                    
                    raw_loss = getattr(current_optimized_instance, '_loss', float('inf'))
                    current_loss = raw_loss.item() if hasattr(raw_loss, 'item') else raw_loss
                    
                    history_records.append((current_loss, prompt, epoch))
                    
                    # 判斷是否 Early Stop
                    if current_loss < best_loss - 1e-4:
                        best_loss = current_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        logging.info(f"Early stopping at epoch {epoch} for instance {data_idx+1} (Loss {current_loss:.4f})")
                        break

                # --- 3. RAILS 混合篩選 Hybrid Selection ---
                logging.info(f"Optimization finished for instance {data_idx+1}. Selecting RAILS candidates...")
                K_exploit = K // 2
                K_explore = K - K_exploit
                
                # 開發 (Exploitation)：選 Loss 最低的 K_exploit 個
                sorted_records = sorted(history_records, key=lambda x: x[0])
                best_exploit = sorted_records[:K_exploit]
                
                # 探索 (Exploration)：從剩餘的不同 Epoch 中均勻抽樣
                picked_prompts = set([x[1] for x in best_exploit])
                remaining_records = [x for x in history_records if x[1] not in picked_prompts]
                
                epoch_dict = defaultdict(list)
                for r in remaining_records:
                    epoch_dict[r[2]].append(r)
                
                explore_candidates = []
                available_epochs = list(epoch_dict.keys())
                random.shuffle(available_epochs)
                
                for ep in available_epochs:
                    if len(explore_candidates) >= K_explore:
                        break
                    explore_candidates.append(random.choice(epoch_dict[ep]))
                
                final_candidates = best_exploit + explore_candidates
                
                # 若候選數量不足 K，用最優解補齊避免 IndexError
                while len(final_candidates) < K and best_exploit:
                     final_candidates.append(best_exploit[0])

                # --- 4. 黑盒測試 (Phase 2) ---
                logging.info(f"Testing {len(final_candidates)} candidates on Target Model for instance {data_idx+1}...")
                if self.dataset_name == 'trustllm':
                    self.target_model.set_system_message(current_optimized_instance.system_message)
                
                for i, (cand_loss, cand_prompt, cand_epoch) in enumerate(final_candidates):
                    # API 呼叫目標黑盒模型
                    response = self.target_model.generate(cand_prompt)
                    
                    test_instance = copy.deepcopy(current_optimized_instance)
                    test_instance.jailbreak_prompt = cand_prompt
                    test_instance.target_responses = [response]
                    
                    # 使用 Append ('a') 寫入該測試檔案，斷線也不會遺失前面的結果
                    with open(self.save_path + f'/final_test_shot_{i}.jsonl', 'a') as f:
                        f.write(json.dumps(test_instance.to_dict(), ensure_ascii=False) + '\n')
                        
                    # === 黑盒測試 Early Stop 機制 (省錢神器) ===
                    is_success = False
                    for ref in test_instance.reference_responses:
                        if ref.lower() in response.lower():
                            is_success = True
                            break
                            
                    if is_success:
                        logging.info(f"Hit! Instance {data_idx+1} privacy leaked at candidate {i} (from epoch {cand_epoch}).")
                        
                        for j in range(i + 1, K):
                            with open(self.save_path + f'/final_test_shot_{j}.jsonl', 'a') as f:
                                f.write(json.dumps(test_instance.to_dict(), ensure_ascii=False) + '\n')
                        break # 直接跳脫此資料的黑盒測試，進入下一筆資料！

                # 釋放這回合的圖與記憶體
                gc.collect()
                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")

        logging.info("RAILS Transfer Attack finished!")