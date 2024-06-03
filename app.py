from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
import torch
import os
from werkzeug.utils import secure_filename
import pytesseract
import pathlib
import numpy as np
from flask_socketio import SocketIO
import yt_dlp as youtube_dl

# Chỉ định đường dẫn đến tệp thực thi của Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')
top_flag = False

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_image(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    for det in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
    cv2.imwrite(output_path, img)
    return output_path

class VideoStreaming(object):
    def __init__(self):
        super(VideoStreaming, self).__init__()
        print ("===== Video Streaming =====")
        self._preview = False
        self._flipH = False
        self._detect = False
        self._model = False
        self._confidence = 75.0

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        self._confidence = int(value)

    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)

    def show(self, url):
        print(url)
        self._preview = False
        self._flipH = False
        self._detect = False
        self._confidence = 75.0
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best",
            "forceurl": True,
        }

        if url == '0':
            cap = cv2.VideoCapture(0)
        else:
            
            ydl = youtube_dl.YoutubeDL(ydl_opts)

            info = ydl.extract_info(url, download=False)
            url = info["url"]

            cap = cv2.VideoCapture(url)

        while True:
            if self._preview:
                if stop_flag:
                    print("Process Stopped")
                    return

                grabbed, frame = cap.read()
                if not grabbed:
                    break
                if self.flipH:
                    frame = cv2.flip(frame, 1)

                if self.detect:
                    # Copy the frame
                    frame_yolo = frame.copy()

                    # Perform prediction
                    results_yolo = model(frame_yolo)

                    # Extract bounding boxes, labels, and confidence scores
                    bboxes = results_yolo.xyxy[0][:, :4].cpu().numpy()  # Bounding boxes
                    labels = results_yolo.xyxy[0][:, -1].cpu().numpy()  # Labels
                    confidences = results_yolo.xyxy[0][:, -2].cpu().numpy()  # Confidence scores

                    # Check if any objects were detected
                    if len(labels) > 0:
                        # Convert label indices to actual label names
                        label_names = [model.names[int(label)] for label in labels]

                        # Prepare the list to send via socket
                        list_labels = []
                        for bbox, label, confidence in zip(bboxes, label_names, confidences):
                            list_labels.append(label)
                            list_labels.append(f"{confidence:.2f}")

                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # Put label and confidence on the bounding box
                            label_text = f"{label} {confidence:.2f}"
                            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                            cv2.rectangle(frame_yolo, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                            cv2.putText(frame_yolo, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                        # Emit the labels and confidences
                        socketio.emit('label', list_labels)
                        
                        # Encode the modified frame for streaming
                        if isinstance(frame_yolo, np.ndarray):
                            ret, buffer = cv2.imencode('.jpg', frame_yolo)
                            frame_yolo = buffer.tobytes()
                            yield (b'--frame_yolo\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_yolo + b'\r\n')
                frame = cv2.imencode(".jpg", frame)[1].tobytes()
                yield ( 
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                )
            else:
                snap = np.zeros((
                    1000,
                    1000
                ), np.uint8)
                label = "Streaming Off"
                H, W = snap.shape
                font = cv2.FONT_HERSHEY_PLAIN
                color = (255, 255, 255)
                cv2.putText(snap, label, (W//2 - 100, H//2),
                            font, 2, color, 2)
                frame = cv2.imencode(".jpg", snap)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# check_settings()
VIDEO = VideoStreaming()


@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('hompage.html')


@app.route('/index', methods=['GET', 'POST'])
def index():
    print("index")
    global stop_flag
    stop_flag = False
    if request.method == 'POST':
        print("Index post request")
        url = request.form['url']
        print("index: ", url)
        session['url'] = url
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    url = session.get('url', None)
    print("video feed: ", url)
    if url is None:
        return redirect(url_for('homepage'))

    return Response(VIDEO.show(url), mimetype='multipart/x-mixed-replace; boundary=frame')

# * Button requests
@app.route("/request_preview_switch")
def request_preview_switch():
    VIDEO.preview = not VIDEO.preview
    print("*"*10, VIDEO.preview)
    return "nothing"

@app.route("/request_flipH_switch")
def request_flipH_switch():
    VIDEO.flipH = not VIDEO.flipH
    print("*"*10, VIDEO.flipH)
    return "nothing"

@app.route("/request_run_model_switch")
def request_run_model_switch():
    VIDEO.detect = not VIDEO.detect
    print("*"*10, VIDEO.detect)
    return "nothing"

@app.route('/update_slider_value', methods=['POST'])
def update_slider_value():
    slider_value = request.form['sliderValue']
    VIDEO.confidence = slider_value
    return 'OK'

@app.route('/stop_process')
def stop_process():
    print("Process stop Request")
    global stop_flag
    stop_flag = True
    return 'Process Stop Request'

@socketio.on('connect')
def test_connect():
    print('Connected')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            output_path = detect_image(filepath)
            return render_template('display.html', user_image=output_path)
    return render_template('upload.html')



if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])
    socketio.run(app, debug=True)