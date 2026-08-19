# AI Waste Sorting System 🗑️

## 📝 Project Overview
Real-time waste segregation on a Raspberry Pi. A funnel-mounted camera identifies
each item as metal, paper or plastic, then a rotating drum positions the correct
bin underneath and a tray releases the item. All inference runs on-device — no
cloud, no network. 92–96% accuracy across the three classes.

## 🛠️ Tech Stack
- **AI/ML:** MediaPipe, TFLite, OpenCV
- **Vision Logic:** Custom-trained object detector for material classification
- **Hardware:** Raspberry Pi, two servos on 50 Hz PWM (GPIO 18 / 19)
- **Version Control:** Git/GitHub

## ⚙️ How It Works
1. Item drops through the funnel and settles on the holding tray
2. Camera classifies it — metal, paper or plastic
3. Rotating drum turns the matching segment under the tray
4. Tray releases; item lands in the correct bin

Detection and actuation run on separate threads, so the vision pipeline keeps
running during the ~4-second mechanical cycle.

## 📦 Model

The trained detector (`best.tflite`) is not included in this repository —
institutional project asset. Available for review on request.

To run with your own model, point `--model` at any MediaPipe-compatible
TFLite object detection file whose labels match the keys in `BIN_ANGLES`.

## 📊 Results
- 92–96% classification accuracy across classes
- ~1 item per 4–5 seconds (limited by servo travel, not inference)
- Fully on-device inference

## 🚀 Run It
```bash
pip install -r requirements.txt

# on the Pi
python ai_video_bin.py --model best.tflite

# on a laptop, from a recording
python ai_video_bin.py --model best.tflite --mock --source clip.mp4 --output demo.mp4
```

## 📂 Repository Structure
- `ai_video_bin.py` — detection, routing queue, servo control
- `requirements.txt` — dependencies
- `README.md` — documentation
