# This is a sample Dockerfile you can modify to deploy your own app based on face_recognition

FROM python:3.9.10

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS


# The rest of this file just runs an example script.

# If you wanted to use this Dockerfile to run your own app instead, maybe you would do this:
# COPY . /root/your_app_or_whatever
# RUN cd /root/your_app_or_whatever && \
#     pip3 install -r requirements.txt
# RUN whatever_command_you_run_to_start_your_app

COPY . /root/face_recognition
RUN cd /root/face_recognition && \
    pip3 install -r requirements.txt && \
    python3 setup.py install

WORKDIR /

RUN git clone https://github.com/FalkTannhaeuser/python-onvif-zeep.git

WORKDIR //python-onvif-zeep

RUN python setup.py install

RUN pip3 install opencv-python-headless

WORKDIR /root/face_recognition/
# CMD cd /root/face_recognition/examples && \
    # python3 web_service_example_Simplified_Chinese.py
# docker stop facetest && docker rm facetest
# docker build -t raidavid/facetest .
# docker stop facetest && docker rm facetest && \
# docker run --name facetest \
# -p 5432:5001 \
# --restart=always \
# -d \
# -it \
# raidavid/facetest

