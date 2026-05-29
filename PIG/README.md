# PIG
> 與官方程式碼幾乎相同

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
由於官方的 `easyjailbreak/attacker/` 中的檔案沒寫好 batchsize 設定 (有時直接改 `attack.py` 沒卵用)，因此 OOM 時請先嘗試修改 `easyjailbreak/attacker/` 的檔案中的 batchsize 。(具體修改的檔案則取決於你在 `run.sh` 或 `attack.py` 中選擇的 jailbreak_method )
```sh
# execute the privacy jailbreak attack
bash run.sh
# evaluate
python eval.py
```