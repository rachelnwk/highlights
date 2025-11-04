# Highlight Gen.
need to install ffmpeg!

planning on having it only work with CUDA-enabled gpus

need to figure out how to delete the local files

Model: https://huggingface.co/keremberke/yolov8n-csgo-player-detection

## to first create backend and start it:
cd backend

python -m venv venv

pip install -r requirements.txt

venv\Scripts\activate

python app.py

### to run it again:
cd backend

venv\Scripts\activate

python app.py


## to start frontend:
cd frontend

npm install (if first timer)

npm run dev (or npm start)
