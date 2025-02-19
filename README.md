

### YOLOv7 Model for Weed Detection 
This project utilizes the YOLOv7 model for weed detection. You can access the YOLOv7 model and datasets on Roboflow:  

🔗 **[YOLOv7 on Roboflow](https://universe.roboflow.com/deep-learning-assignment-ewyc5/weed-detection-d7dau/dataset/7 )   

To use YOLOv7 in this project, follow these steps:  
1. Clone the YOLOv7 repository:  
   ```bash
   git clone https://github.com/WongKinYiu/yolov7.git
   ```  
2. Install the required dependencies:  
   ```bash
   pip install -r yolov7/requirements.txt
   ```  
3. Download the trained model weights from Roboflow or train your custom model.

---
If working with a GPU, install torch with CUDA (e.g., torch==1.13.1+cu117).
You can generate your own requirements.txt by running:

pip freeze > requirements.txt
