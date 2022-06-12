FROM pytorch/pytorch
# FROM anibali/pytorch

WORKDIR /home
COPY . .
RUN apt update
RUN apt install -y git
RUN pip install -r requirements.txt
# CMD tensorboard --logdir /home/logs/lpbert --bind_all
