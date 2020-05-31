# 사용법
# python webstreaming.py

# 필요한 라이브러리를 import
from singlemotiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import datetime
import imutils
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# Flask object 생성
app = Flask("__name__")

# VideoStream을 초기화시키고 카메라 센서가 예열되도록 한다.
# vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()


@app.route("/")
def index():
    # index template 반환
    return render_template("index.html")


# 모션 감지 함수
def detect_motion(frameCount):
    # 출력 프레임과 lock, video stream 변수에 대한 전역변수로 참조
    global vs, outputFrame, lock

    # SingleMotionDetector 객체를 초기화하고, 총 프레임 수를 0으로 초기화한다.
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    # Video Stream으로부터 오는 frame을 반복해서 생성한다.
    while True:
        # Video Stream에서 다음 프레임을 읽고, 크기를 width 400으로 조정한다.
        # 그 후 frame을 grayscale로 변환 한 후 흐리게 처리한다.
        frame = vs.read()
        frame = imutils.resize(frame, width=750)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # 현재 시간을 구한 후 frame에 그린다.
        timestamp = datetime.datetime.now()
        cv2.putText(
            frame,
            timestamp.strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
        )

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frameCount:
            # SingleMotionDetector 클래스의 detect 함수로 이미지에서 움직임을 감지.
            motion = md.detect(gray)

            # frame에서 움직임이 감지되었는지 확인한다. return 같이 None이 아닐경우 감지.
            if motion is not None:
                # motion tuple을 풀고, 출력 frame에 motion 영역을 둘러싸는 사각형을 그린다.
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)

        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1

        # thread lock을 얻고, 출력 프레임을 설정 후 lock을 해제한다.
        with lock:
            outputFrame = frame.copy()


def generate():
    # 출력 프레임과 lock 변수에 대한 전역변수로 참조
    global outputFrame, lock

    # outputFrame으로부터 오는 frame을 반복한다.
    while True:
        # thread lock을 얻는다.
        with lock:
            # outputFrame이 None인지 확인하고, None이면 loop문의 반복을 건너뛴다.
            if outputFrame is None:
                continue

            # frame을 JPEG 형식으로 인코딩한다.
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # frame이 성공적으로 인코딩되었는지 확인한다.
            # flag의 여부로 확인.
            if not flag:
                continue

        # outputFrame을 byte 형식으로 반환한다.
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    # multipart/x-mixed-replace MIME 형식을 사용하여 하나의 jpg를 표시하고 다음에 다른 jpg가
    # 그것을 대치하고 계속해서 여러 개의 jpg 파일을 이와 같은 방법으로 대치하여 이것으로 간단한
    # 애니메이션을 웹에서 구현한다.
    # https://qaos.com/sections.php?op=viewarticle&artid=272
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# detect_motion을 수행할 thread를 시작한다.
# daemon thread를 true로 주어 main thread가 종료되면 즉시 종료되게 한다.
t = threading.Thread(target=detect_motion, args=(128,))
t.daemon = True
t.start()

# localhost 8000번 포트에 Flask Server 시작
app.run(host="0.0.0.0", port="8000", debug=True, threaded=True, use_reloader=False)

# VideoStream을 해제한다.
vs.stop()
