FROM nvcr.io/nvidia/pytorch:22.12-py3
COPY requirement.txt .
RUN apt update && apt install git-lfs nvtop -y \
    && git lfs install 
RUN pip install -U pip    
RUN pip install -r requirement.txt