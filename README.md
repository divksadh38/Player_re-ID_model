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
```
├── main.py # Final working tracking script
├── best.pt # Custom YOLOv8 detection model
├── osnet_x1_0_market1501.pth # Pretrained ReID model (TorchReID)
├── 15sec_input_720p.mp4 # Input match clip
├── requirements.txt # All dependencies
├── README.md # 📄 You’re reading it
├── report.md # Optional writeup
```

---

## ⚙️ Set Up Instructions

### 1. 🧪 Conda Environment


### 1. ✅ Set Up the Conda Environment

Create and activate a clean Python 3.9 environment:

```bash
conda create -n player_reid_env python=3.10 -y
conda activate player_reid_env
```

2. 📦 Install Dependencies
Install all required packages:

```bash

pip install ultralytics opencv-python deep_sort_realtime torch torchvision numpy ffmpeg-python
pip install torchreid
💡 If using GPU, install PyTorch with CUDA 12.1:
```
```bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
3. 🚀 Verify GPU is Working
Run this in Python to confirm CUDA:
```
python

import torch
print(torch.cuda.is_available())  # Should print True
```

4. 📁 Place the Files in the Repo Folder
Your Player_re-ID_model/ folder should contain:

5.Download the pre=trained YOLO model:

markdown

🔗 Download the detection model (`best.pt`):(https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

bash
  Player_re-ID_model/
  ├── main.py                          # Your tracking script
  ├── best.pt                          # YOLOv8 model trained on soccer data
  ├── osnet_x1_0_market1501.pth        # TorchReID model weights
  ├── 15sec_input_720p.mp4             # Test video
  ├── README.md                        # This file
  └── requirements.txt                 # Optional
5. ▶️ Run the Tracker
From inside your virtual environment:
```
bash

python main.py
```

✅ Expected Behavior
YOLOv11 detects players (class 2) in each frame.

DeepSORT assigns persistent IDs using TorchReID's OSNet model.

IDs may occasionally switch on occlusion or drastic appearance change.

🧩 The Problems I Faced, What I Tried, and What I’ll Do Next
❌ What Went Wrong
While working on this project, I ran into a lot of frustrating issues:

StrongSORT Gave Terrible Results
The tracker didn’t work properly at all. Bounding boxes were all over the place, the IDs kept changing every second, and the entire output looked like chaos. Even though I used a pre-trained model, the tracking just wouldn’t hold.

Boxes and Crops Kept Breaking the Code
Half the time, the boxes passed to the tracker were either outside the frame or badly formatted. This caused errors in resizing the crops, and the program would just crash. It took a while to figure out what was going wrong.


🔁 What I Tried
Tried StrongSORT First
I started with StrongSORT and a model called osnet_x0_25, thinking it would be powerful. It technically ran but gave unusable results.

Thought About Training My Own Re-ID Model
I did look into training a custom player re-ID model from scratch using football footage, but I realized I didn’t have the time to do it right now.

DeepSORT with OSNet1501 (What Finally Worked)
In the end, I used DeepSORT with a pre-trained osnet_x1_0 model trained on Market1501 — and that finally gave decent results. The player IDs stuck for longer, and tracking was actually usable.

🔮 What I Plan to Do Next
Use SoccerNet for Re-ID Training
In the future, I want to use the re-ID data from SoccerNet to train a proper model that actually understands football players — especially since broadcast views are so different.

Use Jersey Numbers to Improve Tracking
I also plan to detect jersey numbers and use them to track players. That way, even if the model gets confused or a player runs across the field, I can keep their ID consistent.

Keep Improving As I Learn More
This project taught me a lot, and I know there’s still a long way to go. But now I have a working base that I can build on.

