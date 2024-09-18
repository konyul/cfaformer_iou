# sensor_failure_detection

## 설치방법 (Linux)

directory 생성
```
mkdir $[디렉토리명}
cd $[디렉토리명}
git clone https://github.com/konyul/sensor_failure_detection.git
```
docker image 생성

```
docker pull kyparkk/meformer:python3.8_torch1.11.0_cu113
```

docker container 생성

```
docker run -it --gpus all --shm-size=512g -v ${서버 데이터 경로}:${원하는 경로} -w ${디렉토리 경로} --name ${container 명} kyparkk/meformer:python3.8_torch1.11.0_cu113 /bin/bash
```

docker container 접속

```
docker exec -it ${컨테이너명} /bin/bash
```

다음 명령어 입력

```
cd mmdetection3d
pip install -v -e.
cd ..
pip install -v -e.
```

데이터 생성

```
mkdir -p data/nuscenes
```

체크포인트 경로

Download the pretrained weight of the image backbone from [Google Drive](https://drive.google.com/drive/folders/1vNGjBxl__eF5ti-AAsNWFqJG4uBmYsPG?usp=drive_link) and move them to ckpts directory.
```shell
sensor_failure_detection
├─ ckpts
│  ├─ fcos3d_vovnet_imgbackbone-remapped.pth
│  └─ nuim_r50.pth
├─ figures
├─ projects
└─ tools
```


훈련 모델 실행 (예시)

```
tools/dist_train.sh projects/configs/moad_voxel0100_r50_800x320_cbgs.py  4

```
