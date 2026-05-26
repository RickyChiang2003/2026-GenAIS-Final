# PIG+RAILS

## Environment  
You need at least 48GB GPU VRAM, or you should adjust your code (no config file here).
```sh
# Basics (cuda 11.8, pip 26.0.1)
conda create -n PIG python=3.9.19
conda activate PIG
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.2 
pip install accelerate==0.25.0 datasets==2.16.1 tokenizers==0.15.0 fschat==0.2.34 protobuf==3.20.3 sentencepiece jsonlines openai>=1.0.0 numpy<2.0.0 pandas scipy anthropic google-generativeai scikit-learn nltk
```

## Usage

```sh
# execute the privacy jailbreak attack
bash run.sh
# evaluate
python eval.py
```