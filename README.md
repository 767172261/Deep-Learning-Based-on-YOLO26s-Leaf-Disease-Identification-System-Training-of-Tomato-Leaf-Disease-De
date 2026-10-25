YOLO26的叶片病害识别系统
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/21085750dbf44054bf4a572914aa1ec9.png)

✔包含数据集，训练好的模型，系统配置和说明文档

✔支持图像和视频检测，可识别早疫病、晚疫病、叶霉病等9种常见病害，数据集格式为yolo格式

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a01417141a054ba9bedc989cd7c16375.png)
基于 **YOLO26** 的番茄叶片病害识别系统的完整构建方案，包含数据集、训练代码、系统代码及配置说明：


### 一、数据集信息
#### 1. 类别（9种）
```
0: Early_Blight（早疫病）
1: Healthy（健康）
2: Late_Blight（晚疫病）
3: Leaf_Miner（潜叶虫）
4: Leaf_Mold（叶霉病）
5: Mosaic_V（花叶病毒）
6: Septoria（斑点霉菌病）
7: Spider_M（蜘蛛螨）
8: YLCV（黄曲叶病毒）
```

#### 2. 格式与结构
采用 **YOLO TXT格式**，数据集结构如下：
```
dataset/
├─ images/
│  ├─ train/   # 训练集图像（约70%）
│  ├─ val/     # 验证集图像（约20%）
│  └─ test/    # 测试集图像（约10%）
└─ labels/
   ├─ train/   # 训练集标注（每个图像对应.txt文件）
   ├─ val/     # 验证集标注
   └─ test/    # 测试集标注
```
*标注示例（.txt文件）*：`3 0.51 0.35 0.39 0.32`（类别ID + 归一化坐标）


### 二、模型训练代码（YOLO26）
#### 1. 环境配置
```bash
pip install ultralytics  # 需支持YOLO26的版本
```

#### 2. 训练配置文件（`data.yaml`）
```yaml
train: dataset/images/train
val: dataset/images/val
test: dataset/images/test
nc: 9
names: ['Early_Blight', 'Healthy', 'Late_Blight', 'Leaf_Miner', 'Leaf_Mold', 'Mosaic_V', 'Septoria', 'Spider_M', 'YLCV']
```

#### 3. 训练代码（`train.py`）
```python
from ultralytics import YOLO

# 加载YOLO26模型（以n版本为例，可选s/m/l/x）
model = YOLO('yolo26n.yaml')  

# 训练模型
results = model.train(
    data='data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    optimizer='musgd',  # YOLO26推荐的MuSGD优化器
    device=0  # GPU设备，无GPU则设为'cpu'
)

# 验证模型
model.val()
```


### 三、病害识别系统代码（PyQt5界面）
支持**图像/视频/摄像头检测**，核心代码如下：
```python
import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

class DiseaseDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = YOLO('tomato-leaf-disease-model.pt')  # 加载训练好的模型
        self.initUI()

    def initUI(self):
        # 界面布局（简化版）
        self.setWindowTitle("番茄叶片病害识别系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 控件：图像显示、按钮
        self.image_label = QLabel(self)
        self.image_label.setGeometry(200, 20, 800, 600)
        
        self.btn_select = QPushButton("选择图像", self)
        self.btn_select.setGeometry(20, 20, 150, 40)
        self.btn_select.clicked.connect(self.select_image)

    def select_image(self):
        # 选择图像并检测
        path, _ = QFileDialog.getOpenFileName()
        if path:
            img = cv2.imread(path)
            results = self.model(img)  # YOLO26推理
            result_img = results[0].plot()  # 绘制检测框
            
            # 显示图像
            h, w, ch = result_img.shape
            q_img = QImage(result_img.data, w, h, w*ch, QImage.Format_BGR888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(800, 600))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DiseaseDetector()
    window.show()
    sys.exit(app.exec_())
```



