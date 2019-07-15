FROM nvidia/cuda:9.0-base-ubuntu16.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        libcudnn7=7.2.1.38-1+cuda9.0 \
        libnccl2=2.2.13-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        python3-pip \
        python3-dev \
        python3-pip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
        apt-get update && \
        apt-get install libnvinfer4=4.1.2-1+cuda9.0

RUN apt-get install -y build-essential checkinstall software-properties-common llvm cmake wget git nano nasm yasm zip unzip pkg-config && apt-get update

RUN pip3 install --upgrade \
    pip \
    setuptools

RUN pip3 install tensorflow-gpu==1.14.0 keras==2.2.4 pandas==0.23.0 numpy==1.16.4 scikit-learn==0.21.2 tqdm==4.26.0 seaborn==0.8.1 matplotlib==2.2.2 scipy==1.1.0 torch==1.0.1 catboost==0.13.1 

RUN pip3 install --pre xgboost==0.82

RUN pip3 install ipython[all]
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN jupyter notebook --generate-config
# VOLUME DUNE
# RUN jupyter notebook --no-browser --ip='0.0.0.0' --allow-root

ENTRYPOINT ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip=0.0.0.0"]

