# PIG+RAILS

## Environment  
You need at least 48GB GPU VRAM, or you should adjust your code (no config file here).
```sh
# Basics (cuda 11.8, pip 26.0.1)
conda create -n PIG python=3.9.19
conda activate PIG
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.2 
pip install accelerate==0.25.0 datasets==2.16.1 tokenizers==0.15.0 fschat==0.2.36 protobuf==3.20.3 sentencepiece jsonlines openai>=1.0.0 numpy<2.0.0 pandas scipy anthropic google-generativeai scikit-learn nltk bitsandbytes==0.40.0
# for llama-3
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip3 install -e ".[model_worker,webui]"
```

## Usage

```sh
# execute the privacy jailbreak attack
bash run.sh
# evaluate
python eval.py
```

# Prompts

現在請你開始正式與我開始改 code 。
我的 GPU 為 48GB VRAM 的， PIG 的官方環境我都裝好了。以下附上原版與我修改好的 PIG_Eden_2024.py ，請你仔細思考有沒有問題，並告訴我怎麼設定 run.sh。

我的方法如下：
```
PIG 原本的作法是：跑完幾百個 Epoch，只要成功越獄，或者挑出 Loss 最低的那「一組」Prompt，就直接拿去攻擊黑盒模型（如 GPT-4）。這很容易「過擬合（Overfit）」在本地端那個開源模型的特定參數上。
RAILS 提出了一個概念：History-Based Candidate Selection（基於歷史的候選篩選）。低 Loss 不代表一定能成功越獄黑盒模型。
實作方式：
照常跑 PIG： 但是在跑的過程中，把每一個 Epoch 產生出來的 Prompt 及其 Loss 都記錄下來（存成一個 History Buffer）。
混合篩選（Hybrid Selection）： 跑完之後，不要只拿 Loss 最低的那一個。模仿 RAILS 的作法，挑選 $K$ 個候選者（例如 20 個）：
一半（10個）選 Loss 絕對值最低的（Exploitation / 開發）。
一半（10個）從歷史軌跡中隨機均勻抽樣不同階段的 Prompt（Exploration / 探索）。
黑盒測試： 將這 20 個 Prompt 都丟給目標黑盒模型測試，只要有一個成功撈出隱私資料，就算 Transfer 成功。
```

以下是 easyjailbreak/attacker/PIG_Eden_2024.py 的官方程式碼內容，供你參考：
```py
import os
import sys
import json
import logging

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

    # 对slice进行排序
    idx_and_slices = list(enumerate(slice_list))
    idx_and_slices = sorted(idx_and_slices, key=lambda x: x[1])

    # 切分字符串
    splited_text = []  # list<(str, int)>
    cur = 0
    for sl_idx, sl in idx_and_slices:  # sl_idx指的是sort之前的序号
        splited_text.append((prompt[cur: sl.start], None))
        splited_text.append((prompt[sl.start: sl.stop], sl_idx))
        cur = sl.stop
    splited_text.append((prompt[cur:], None))
    splited_text = [s for s in splited_text if s[0] != '' or s[1] is not None]

    # 完整input_idx，对整个句子tokenize
    ans_input_ids = model.batch_encode(prompt, return_tensors='pt')['input_ids'].to(model.device)[:, 1:]  # 1 * L

    # 查找每个字符串段落在input_ids中的区段
    ans_slices = []  # list<(int, slice)>
    splited_text_idx = 0
    start = 0
    cur = 0
    while cur < ans_input_ids.size(1):
        text_seg = model.batch_decode(ans_input_ids[:, start: cur + 1])[0]  # str
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

    # 按照顺序和传入的slice对应
    token_id_list = [item for item in ans_slices if item[0] is not None]
    # 固定头尾实体token
    # token_id_list = [(sl.start + 1, sl.stop - 1) for _, sl in sorted(token_id_list, key=lambda x: x[0])]
    # 不固定头尾实体token
    token_id_list = [(sl.start, sl.stop) for _, sl in sorted(token_id_list, key=lambda x: x[0])]
    assert len(token_id_list) == len(slice_list)

    # 将token_id_list展开
    token_id_list = flatten_list(token_id_list)
    # 同时替换实体中所有token
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
            batchsize: int = 16,
            top_k: int = 256,
            max_num_iter: int = 500,
            is_universal: bool = False
    ):
        """
        Initialize the PIG attacker.
        """

        super().__init__(attack_model, target_model, None, jailbreak_datasets)

        if batchsize is None:
            batchsize = num_turb_sample

        self.attack_model = attack_model
         # self.seeder = SeedRandom(seeds_max_length=jailbreak_prompt_length, posible_tokens=['! '])
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
        dataset = self.jailbreak_datasets  # FIXME
        self.jailbreak_datasets = JailbreakDataset([instance])
        self.attack()
        ans = self.jailbreak_datasets
        self.jailbreak_datasets = dataset
        return ans

    def attack(self):
        logging.info("Jailbreak started!")
        try:
            # if self.dataset_name == 'enron':
            #     self.jailbreak_datasets = JailbreakDataset(
            #         list(filter(lambda x: x.shotType != 'zero-shot', self.jailbreak_datasets))
            #     )

            all_instance_pii_token_id_dict = dict()
            for instance in self.jailbreak_datasets:
                one_instance_pii_token_id_dict = defaultdict(list)
                instance.pii_slice_dict = dict(filter(lambda x: x[1] is not None, instance.pii_slice_dict.items()))
                instance.pii_slice_dict = convert_list_to_slice(instance.pii_slice_dict)
                for key, value in instance.pii_slice_dict.items():
                    one_instance_pii_token_id_dict['pii_token_id_list'].extend(
                        slice_to_token_id(self.attack_model, instance.context, value)
                    )
                # 去除列表中重复元素（用于任意位置的token替换）
                one_instance_pii_token_id_dict['pii_token_id_list'] = list(set(one_instance_pii_token_id_dict['pii_token_id_list']))
                all_instance_pii_token_id_dict[instance['idx']] = one_instance_pii_token_id_dict

                input_ids, _, _, response_slice = model_utils.encode_trace(
                    self.attack_model,
                    instance.query,
                    f'{instance.context} {{query}}',
                    instance.reference_responses[0]
                )

                if instance.jailbreak_prompt is None:
                    instance.jailbreak_prompt = f'{instance.context} {{query}}'

                instance.token_id_length = len(input_ids[0])

            breaked_dataset = JailbreakDataset([])
            unbreaked_dataset = self.jailbreak_datasets
            for epoch in tqdm(range(self.max_num_iter)):
                logging.info(f"Current PIG epoch: {epoch}/{self.max_num_iter}")
                # if epoch != 0:
                unbreaked_dataset = self.mutator(unbreaked_dataset, all_instance_pii_token_id_dict)
                logging.info(f"Mutation: {len(unbreaked_dataset)} new instances generated.")
                unbreaked_dataset = self.selector.select(unbreaked_dataset)
                logging.info(f"Selection: {len(unbreaked_dataset)} instances selected.")

                for instance in unbreaked_dataset:
                    if self.dataset_name == 'trustllm':
                        self.target_model.set_system_message(instance.system_message)
                    prompt = instance.jailbreak_prompt.replace('{query}', instance.query)
                    logging.info(f'Generation: input=`{prompt}`')
                    instance.target_responses = [self.target_model.generate(prompt)]
                    logging.info(f'Generation: Output=`{instance.target_responses}`')

                self.evaluator(unbreaked_dataset)
                self.jailbreak_datasets = JailbreakDataset.merge([unbreaked_dataset, breaked_dataset])

                with open(self.save_path + f'/epoch_{epoch}.jsonl', 'w') as f:
                    for new_instance in tqdm(unbreaked_dataset):
                        line = new_instance.to_dict()
                        # if epoch == 0:
                        #     if self.dataset_name == 'enron':
                        #         line = {
                        #             'idx': line['idx'],
                        #             'query': line['query'],
                        #             'jailbreak_prompt': line['jailbreak_prompt'],
                        #             'target_responses': line['target_responses'],
                        #             'reference_responses': line['reference_responses'],
                        #             'type': line['type'],
                        #             'shotType': line['shotType'],
                        #             'ground_truth': line['ground_truth'],
                        #             'token_id_length': line['token_id_length'],
                        #             '_loss': line['_loss']
                        #         }
                        #     elif self.dataset_name == 'trustllm':
                        #         line = {
                        #             'idx': line['idx'],
                        #             'name': line['name'],
                        #             'query': line['query'],
                        #             'context': line['context'],
                        #             'jailbreak_prompt': line['jailbreak_prompt'],
                        #             'target_responses': line['target_responses'],
                        #             'reference_responses': line['reference_responses'],
                        #             'system_message': line['system_message'],
                        #             'type': line['type'],
                        #             'privacy_information': line['privacy_information'],
                        #             'ground_truth': line['ground_truth'],
                        #             'token_id_length': line['token_id_length'],
                        #             '_loss': line['_loss']
                        #         }
                        f.write(json.dumps(line, ensure_ascii=False) + '\n')

                # check
                cnt_attack_success = 0
                breaked_dataset = JailbreakDataset([])
                unbreaked_dataset = JailbreakDataset([])
                for instance in self.jailbreak_datasets:
                    if instance.eval_results[-1]:
                        cnt_attack_success += 1
                        breaked_dataset.add(instance)
                    else:
                        unbreaked_dataset.add(instance)
                logging.info(f"Successfully attacked: {cnt_attack_success}/{len(self.jailbreak_datasets)}")
                # if os.environ.get('CHECKPOINT_DIR') is not None:
                #     checkpoint_dir = os.environ.get('CHECKPOINT_DIR')
                #     self.jailbreak_datasets.save_to_jsonl(f'{checkpoint_dir}/gcg_{epoch}.jsonl')
                if cnt_attack_success == len(self.jailbreak_datasets):
                    break  # all instances is successfully attacked
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")

        self.log_results(cnt_attack_success)
        logging.info("Jailbreak finished!")
```

我修改後的版本：
```py
import os
import sys
import json
import logging
import random
import copy

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
        logging.info("Jailbreak started!")
        try:
            all_instance_pii_token_id_dict = dict()
            for instance in self.jailbreak_datasets:
                one_instance_pii_token_id_dict = defaultdict(list)
                instance.pii_slice_dict = dict(filter(lambda x: x[1] is not None, instance.pii_slice_dict.items()))
                instance.pii_slice_dict = convert_list_to_slice(instance.pii_slice_dict)
                for key, value in instance.pii_slice_dict.items():
                    one_instance_pii_token_id_dict['pii_token_id_list'].extend(
                        slice_to_token_id(self.attack_model, instance.context, value)
                    )
                one_instance_pii_token_id_dict['pii_token_id_list'] = list(set(one_instance_pii_token_id_dict['pii_token_id_list']))
                all_instance_pii_token_id_dict[instance['idx']] = one_instance_pii_token_id_dict

                input_ids, _, _, response_slice = model_utils.encode_trace(
                    self.attack_model,
                    instance.query,
                    f'{instance.context} {{query}}',
                    instance.reference_responses[0]
                )

                if instance.jailbreak_prompt is None:
                    instance.jailbreak_prompt = f'{instance.context} {{query}}'

                instance.token_id_length = len(input_ids[0])

            # RAILS 歷史紀錄與備份
            history_buffer = defaultdict(list)
            instance_map = {}

            # === Early Stop 變數 ===
            best_avg_loss = float('inf')
            patience = 50  # 若連續 50 個 Epoch Loss 沒下降則提早結束
            patience_counter = 0

            unbreaked_dataset = self.jailbreak_datasets
            for epoch in tqdm(range(self.max_num_iter)):
                logging.info(f"Current PIG epoch: {epoch}/{self.max_num_iter}")
                unbreaked_dataset = self.mutator(unbreaked_dataset, all_instance_pii_token_id_dict)
                logging.info(f"Mutation: {len(unbreaked_dataset)} new instances generated.")
                unbreaked_dataset = self.selector.select(unbreaked_dataset)
                logging.info(f"Selection: {len(unbreaked_dataset)} instances selected.")

                # 紀錄 Loss 軌跡
                current_epoch_losses = []
                for instance in unbreaked_dataset:
                    prompt = instance.jailbreak_prompt.replace('{query}', instance.query)
                    current_loss = getattr(instance, '_loss', float('inf'))
                    current_epoch_losses.append(current_loss)
                    
                    history_buffer[instance.idx].append((current_loss, prompt, epoch))
                    instance_map[instance.idx] = instance

                self.jailbreak_datasets = unbreaked_dataset

                with open(self.save_path + f'/epoch_{epoch}.jsonl', 'w') as f:
                    for new_instance in tqdm(unbreaked_dataset):
                        line = new_instance.to_dict()
                        f.write(json.dumps(line, ensure_ascii=False) + '\n')

                # === 白盒優化 Early Stop 檢查 ===
                avg_loss = sum(current_epoch_losses) / len(current_epoch_losses) if current_epoch_losses else float('inf')
                if avg_loss < best_avg_loss - 1e-4:
                    best_avg_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logging.info(f"Phase 1 Early stopping triggered! Average Loss hasn't improved for {patience} epochs.")
                    break  # 跳出 Epoch 迴圈，提早進入黑盒測試

            # 迴圈結束，執行 RAILS 混合篩選與黑盒測試
            logging.info("Optimization finished. Starting RAILS Hybrid Selection and Black-box testing...")
            K = 20
            K_exploit = K // 2
            K_explore = K - K_exploit
            
            final_test_results = defaultdict(list)

            for idx, records in history_buffer.items():
                original_instance = instance_map[idx]
                
                # 1. Exploitation (開發): Loss 最低的 K_exploit 個
                sorted_records = sorted(records, key=lambda x: x[0])
                best_exploit = sorted_records[:K_exploit]
                
                # 2. Exploration (探索): 剩餘紀錄中抽取 K_explore 個
                picked_prompts = set([x[1] for x in best_exploit])
                remaining_records = [x for x in records if x[1] not in picked_prompts]
                
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
                
                # 3. 測試黑盒模型
                logging.info(f"Testing {len(final_candidates)} candidates for instance {idx} on Target Model...")
                if self.dataset_name == 'trustllm':
                    self.target_model.set_system_message(original_instance.system_message)
                
                for cand_loss, cand_prompt, cand_epoch in final_candidates:
                    response = self.target_model.generate(cand_prompt)
                    
                    # 建立新 instance 儲存特定回覆
                    test_instance = copy.deepcopy(original_instance)
                    test_instance.jailbreak_prompt = cand_prompt
                    test_instance.target_responses = [response]
                    final_test_results[idx].append(test_instance)

                    # === [新增] Phase 2: 黑盒測試 Early Stop (Hit) ===
                    # 只要目標的隱私數據出現在 Response 裡面，就算越獄成功，停止測試剩下的候選者
                    is_success = False
                    for ref in original_instance.reference_responses:
                        if ref.lower() in response.lower():
                            is_success = True
                            break
                    
                    if is_success:
                        logging.info(f"Instance {idx} successfully transferred at candidate from epoch {cand_epoch}! Early stopping test for this instance.")
                        break

            # 寫出結果供 eval.py 解析 (使用 OR 邏輯)
            logging.info("Saving final RAILS test results...")
            for i in range(K):
                with open(self.save_path + f'/final_test_shot_{i}.jsonl', 'w') as f:
                    for idx, instances in final_test_results.items():
                        if i < len(instances):
                            f.write(json.dumps(instances[i].to_dict(), ensure_ascii=False) + '\n')

        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")

        logging.info("RAILS Transfer Attack finished!")
```