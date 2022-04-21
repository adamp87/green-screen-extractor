# Person and head extraction with green screen removal


## Installation

1. Clone this repository

    `git clone https://github.com/adamp87/green-screen-extractor.git`;

2. Go into the repository

    `cd green-screen-extractor`;

3. Create conda environment and activate

    `conda create -n gse python=3.8`,

    `conda activate gse`;

4. Install dependencies

    `pip install -r requirements.txt`,

5. Download pretrained model (non-commercial)

    Link from repository by deepakcrk: [yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman)

    Download Link:  [YOLOv5m-crowd-human](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing)    

## Licensing

Source code uses YOLO5, which is licensed under GPLv3.

Pretrained model is trained on the crowdhuman dataset, which is non-commercial. [CrowdHuman](https://www.crowdhuman.org/
)