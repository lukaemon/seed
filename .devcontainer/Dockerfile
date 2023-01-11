FROM nvcr.io/nvidia/pytorch:22.11-py3
COPY requirement.txt .
RUN apt update && apt install git-lfs nvtop -y \
    && git lfs install 
RUN pip install -U pip    
RUN pip install -r requirement.txt

# 22.12 won't work for t5 model
# RuntimeError: Failed to import transformers.models.t5.modeling_t5 because of the following error (look up to see its traceback):
# cannot import name 'LRScheduler' from 'torch.optim.lr_scheduler' (/usr/local/lib/python3.8/dist-packages/torch/optim/lr_scheduler.py)