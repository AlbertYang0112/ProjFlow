FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update; exit 0
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends vim git python3-dev python3-pip sudo libgl1-mesa-glx libsm6 libxext6 libglib2.0-0 libxrender-dev

RUN useradd -rm -d /home/flow -s /bin/bash -g root -G sudo -u 1001 flow
RUN echo 'flow:flow' | chpasswd

# For user in China
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# Otherwise
# RUN pip3 install pip -U
RUN pip3 install setuptools
RUN pip3 install numpy matplotlib scipy tqdm h5py pyproj jupyter notebook networkx opencv-python tensorflow-gpu==1.14 tensorboard sklearn 
