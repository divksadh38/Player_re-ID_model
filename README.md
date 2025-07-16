## ⚽ Football Player Re-Identification using YOLOv8 + DeepSORT + TorchReID

---

## 📌 Objective

This project implements a pipeline to **detect and re-identify football players** in a video using:

- **YOLOv8** for real-time player detection  
- **DeepSORT** for multi-object tracking  
- **TorchReID** (via OSNet) to maintain consistent IDs across frames

---

## 🧠 Problem Statement

⚽ Players look similar, move fast, and get occluded.  
🏃 You want to **assign a unique ID** to each player and **track them persistently** across a full match clip.

---

## 🧱 Project Structure

├── main.py # Final working tracking script
├── best.pt # Custom YOLOv8 detection model
├── osnet_x1_0_market1501.pth # Pretrained ReID model (TorchReID)
├── 15sec_input_720p.mp4 # Input match clip
├── requirements.txt # All dependencies
├── README.md # 📄 You’re reading it
├── report.md # Optional writeup


---

## ⚙️ Set Up Instructions

### 1. 🧪 Conda Environment


### 1. ✅ Set Up the Conda Environment

Create and activate a clean Python 3.9 environment:

```bash
conda create -n player_reid_env python=3.9 -y
conda activate player_reid_env


2. 📦 Install Dependencies
Install all required packages:

bash
Copy
Edit
pip install ultralytics opencv-python deep_sort_realtime torch torchvision numpy ffmpeg-python
pip install torchreid
💡 If using GPU, install PyTorch with CUDA 12.1:

bash
Copy
Edit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
3. 🚀 Verify GPU is Working
Run this in Python to confirm CUDA:

python
Copy
Edit
import torch
print(torch.cuda.is_available())  # Should print True
4. 📁 Place the Files in the Repo Folder
Your Player_re-ID_model/ folder should contain:

bash
Copy
Edit
Player_re-ID_model/
├── main.py                          # Your tracking script
├── best.pt                          # YOLOv8 model trained on soccer data
├── osnet_x1_0_market1501.pth        # TorchReID model weights
├── 15sec_input_720p.mp4             # Test video
├── README.md                        # This file
└── requirements.txt                 # Optional
5. ▶️ Run the Tracker
From inside your virtual environment:

bash
Copy
Edit
python main.py
Press Q to quit the video preview window.

✅ Expected Behavior
YOLOv8 detects players (class 2) in each frame.

DeepSORT assigns persistent IDs using TorchReID's OSNet model.

IDs may occasionally switch on occlusion or drastic appearance change.

💡 Optional Improvements
Goal	How
Improve re-ID model	Fine-tune OSNet on SoccerNet data
Track referees or ball	Extend YOLO classes
Use jersey numbers as IDs	Add OCR like EasyOCR or Tesseract
Replace DeepSORT with ByteTrack	Use BoT-SORT or OC-SORT

