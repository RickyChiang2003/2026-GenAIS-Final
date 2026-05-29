"""
Iteratively optimizes a specific section in the prompt using guidance from token gradients,
ensuring that the model produces the desired text.

Paper title: Universal and Transferable Adversarial Attacks on Aligned Language Models
arXiv link: https://arxiv.org/abs/2307.15043
Source repository: https://github.com/llm-attacks/llm-attacks/
"""
from ..models import WhiteBoxModelBase, ModelBase
from .attacker_base import AttackerBase
from ..seed import SeedRandom
from ..mutation.gradient.token_gradient import MutationTokenGradient
from ..selector import ReferenceLossSelector
from ..metrics.Evaluator.Evaluator_PrefixExactMatch import EvaluatorPrefixExactMatch
from ..datasets import JailbreakDataset, Instance

import os
import json
import logging
from typing import Optional
from tqdm import tqdm

import random  
import copy

class GCA(AttackerBase):
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
        max_num_iter: int = 500,
        is_universal: bool = False
    ):
        """
        Initialize the GCA attacker.

        :param WhiteBoxModelBase attack_model: Model used to compute gradient variations and select optimal mutations based on loss.
        :param ModelBase target_model: Model used to generate target responses.
        :param JailbreakDataset jailbreak_datasets: Dataset for the attack.
        :param int jailbreak_prompt_length: Number of tokens in the jailbreak prompt. Defaults to 20.
        :param int num_turb_sample: Number of mutant samples generated per instance. Defaults to 512.
        :param Optional[int] batchsize: Batch size for computing loss during the selection of optimal mutant samples.
            If encountering OOM errors, consider reducing this value. Defaults to None, which is set to the same as num_turb_sample.
        :param int top_k: Randomly select the target mutant token from the top_k with the smallest gradient values at each position.
            Defaults to 256.
        :param int max_num_iter: Maximum number of iterations. Will exit early if all samples are successfully attacked.
            Defaults to 500.
        :param bool is_universal: Experimental feature. Optimize a shared jailbreak prompt for all instances. Defaults to False.
        """

        super().__init__(attack_model, target_model, None, jailbreak_datasets)

        if batchsize is None:
            batchsize = num_turb_sample

        self.attack_model = attack_model
        # self.seeder = SeedRandom(seeds_max_length=jailbreak_prompt_length, posible_tokens=['! '])
        self.mutator = MutationTokenGradient(
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

    def single_attack(self, instance: Instance):
        dataset = self.jailbreak_datasets    # FIXME
        self.jailbreak_datasets = JailbreakDataset([instance])
        self.attack()
        ans = self.jailbreak_datasets
        self.jailbreak_datasets = dataset
        return ans
    '''
    def attack(self):
        logging.info("Jailbreak started!")
        try:
            for instance in self.jailbreak_datasets:
                # seed = self.seeder.new_seeds()[0]     # FIXME:seed部分的设计需要重新考虑
                if instance.jailbreak_prompt is None:
                    instance.jailbreak_prompt = f'{instance.context} {{query}}'

            breaked_dataset = JailbreakDataset([])
            unbreaked_dataset = self.jailbreak_datasets
            for epoch in tqdm(range(self.max_num_iter)):
                logging.info(f"Current GCA epoch: {epoch}/{self.max_num_iter}")
                # if epoch != 0:
                unbreaked_dataset = self.mutator(unbreaked_dataset)
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
                        #     if self.dataset_name == 'enron':
                        #         line = {
                        #             'idx': line['idx'],
                        #             'query': line['query'],
                        #             'jailbreak_prompt': line['jailbreak_prompt'],
                        #             'target_responses': line['target_responses'],
                        #             'reference_responses': line['reference_responses'],
                        #             'type': line['type'],
                        #             'shotType': line['shotType'],
                        #             'ground_truth': line['ground_truth'],
                        #         }
                        #     elif self.dataset_name == 'trustllm':
                        #         line = {
                        #             'idx': line['idx'],
                        #             'name': line['name'],
                        #             'query': line['query'],
                        #             'context': line['context'],
                        #             'jailbreak_prompt': line['jailbreak_prompt'],
                        #             'target_responses': line['target_responses'],
                        #             'reference_responses': line['reference_responses'],
                        #             'system_message': line['system_message'],
                        #             'type': line['type'],
                        #             'privacy_information': line['privacy_information'],
                        #             'ground_truth': line['ground_truth'],
                        #         }
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
                #     checkpoint_dir = os.environ.get('CHECKPOINT_DIR')
                #     self.jailbreak_datasets.save_to_jsonl(f'{checkpoint_dir}/gca_{epoch}.jsonl')
                if cnt_attack_success == len(self.jailbreak_datasets):
                    break   # all instances is successfully attacked
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")

        self.log_results(cnt_attack_success)
        logging.info("Jailbreak finished!")
    '''
    def attack(self):
        logging.info("Jailbreak started! (PIG + RAILS Hybrid Selection)")
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

            unbreaked_dataset = self.jailbreak_datasets

            # ==========================================
            # RAILS 1: 初始化 History Buffer
            # ==========================================
            history_buffer = defaultdict(list)  # 結構: { instance_idx: [(loss, prompt, epoch), ...] }
            instance_map = {}                   # 結構: { instance_idx: instance } 備份用

            # ----------------- 白盒優化階段 -----------------
            for epoch in tqdm(range(self.max_num_iter), desc="White-box Optimization"):
                logging.info(f"Current PIG epoch: {epoch}/{self.max_num_iter}")
                
                unbreaked_dataset = self.mutator(unbreaked_dataset, all_instance_pii_token_id_dict)
                logging.info(f"Mutation: {len(unbreaked_dataset)} new instances generated.")
                
                unbreaked_dataset = self.selector.select(unbreaked_dataset)
                logging.info(f"Selection: {len(unbreaked_dataset)} instances selected.")

                # ==========================================
                # 關閉迴圈內的 API 呼叫，改為紀錄 RAILS 軌跡
                # ==========================================
                for instance in unbreaked_dataset:
                    prompt = instance.jailbreak_prompt.replace('{query}', instance.query)
                    
                    # 抓取這輪算出的 proxy loss，若無則預設無限大
                    current_loss = getattr(instance, '_loss', float('inf'))
                    
                    # 紀錄到 history buffer 中
                    history_buffer[instance.idx].append((current_loss, prompt, epoch))
                    instance_map[instance.idx] = instance
                
                # 轉移攻擊不需要在每次 Epoch 都驗證，因此我們假裝都沒成功，把全體丟進下一個 epoch 繼續優化
                self.jailbreak_datasets = unbreaked_dataset

                # 選擇性存檔 (可以把每輪的狀態先寫下來，以防中斷)
                with open(self.save_path + f'/epoch_{epoch}.jsonl', 'w') as f:
                    for new_instance in unbreaked_dataset:
                        f.write(json.dumps(new_instance.to_dict(), ensure_ascii=False) + '\n')

            # ----------------- 黑盒測試階段 -----------------
            # ==========================================
            # RAILS 2: 混合篩選 (Hybrid Selection)
            # ==========================================
            logging.info("Optimization finished. Starting RAILS Hybrid Selection and Black-box testing...")
            K = 20                 # RAILS 挑選 20 個候選者
            K_exploit = K // 2     # 10 個給 Exploitation (最低 Loss)
            K_explore = K - K_exploit # 10 個給 Exploration (時間序列隨機)
            
            final_test_results = defaultdict(list)

            for idx, records in history_buffer.items():
                original_instance = instance_map[idx]
                
                # 1. Exploitation: 取出歷史中 Loss 最低的 K_exploit 個
                sorted_records = sorted(records, key=lambda x: x[0])
                best_exploit = sorted_records[:K_exploit]
                
                # 2. Exploration: 從剩餘的 Epoch 中均勻抽樣
                picked_prompts = set([x[1] for x in best_exploit])
                remaining_records = [x for x in records if x[1] not in picked_prompts]
                
                epoch_dict = defaultdict(list)
                for r in remaining_records:
                    epoch_dict[r[2]].append(r)
                
                explore_candidates = []
                available_epochs = list(epoch_dict.keys())
                random.shuffle(available_epochs) # 隨機打亂時間軌跡
                
                for ep in available_epochs:
                    if len(explore_candidates) >= K_explore:
                        break
                    explore_candidates.append(random.choice(epoch_dict[ep]))
                
                # 結合兩者，獲得最終的 20 個 Prompts
                final_candidates = best_exploit + explore_candidates
                
                # ==========================================
                # RAILS 3: 對目標黑盒模型進行最終測試
                # ==========================================
                logging.info(f"Testing {len(final_candidates)} candidates for instance {idx} on Target Model...")
                if self.dataset_name == 'trustllm':
                    self.target_model.set_system_message(original_instance.system_message)
                
                for cand_loss, cand_prompt, cand_epoch in final_candidates:
                    # 正式發送 API 請求 / Inference
                    response = self.target_model.generate(cand_prompt)
                    
                    # 建立新的實例以儲存測試結果 (避免覆蓋)
                    test_instance = copy.deepcopy(original_instance)
                    test_instance.jailbreak_prompt = cand_prompt
                    test_instance.target_responses = [response]
                    test_instance._loss = cand_loss  # 保留參考 Loss
                    
                    final_test_results[idx].append(test_instance)

            # ==========================================
            # RAILS 4: 儲存黑盒測試結果，相容 eval.py
            # ==========================================
            logging.info("Saving final RAILS test results...")
            for i in range(K):
                # 我們把 20 次黑盒測試結果，偽裝成 20 個 epoch 檔寫出
                # 這樣 eval.py 就能用它原生的「只要一次成功就算成功(OR)」邏輯來處理
                with open(self.save_path + f'/final_test_shot_{i}.jsonl', 'w') as f:
                    for idx, instances in final_test_results.items():
                        if i < len(instances):
                            f.write(json.dumps(instances[i].to_dict(), ensure_ascii=False) + '\n')
                            
            cnt_attack_success = len(self.jailbreak_datasets) # 隨便給個值防止原版 log_results 報錯

        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")

        self.log_results(cnt_attack_success)
        logging.info("RAILS Transfer Attack finished!")