FROM nvcr.io/nvidia/pytorch:22.11-py3
COPY requirement.txt .
RUN apt update && apt install git-lfs -y \
    && git lfs install 
RUN pip install -U pip    
RUN pip install -r requirement.txt

# 22.12 won't work for importing UL2, pytorch version change. 