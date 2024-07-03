from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import cvzone
import math
import ssl
import requests
import json
import time


app = Flask(__name__)

# 학습된 모델 "best.pt"를 사용한다.
model = YOLO(r"best.pt")
# model = YOLO(r"C:\Dongmin\hamster_detector_v2\runs\detect\yolov8n_Hamster_detector\weights\best.pt")

now = time


def detect_objects():
    # 알림을 보내는 시간을 다시 초기화하기 위해 bool값을 사용
    Trigger = True

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        # results : 햄스터를 검출
        results = model(frame, stream=False)

        # 검출된 햄스터를 사각형으로 표시하기 위한 코드
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cvzone.cornerRect(frame, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100))  # 정확도

                # 정확도 70% 이상, Trigger가 True면 메세지 보내기
                if conf >= 70 and Trigger == True:
                    Trigger = False  # Trigger를 False로 바꿔서 메세지 보내지 않게 하기

                    # Slack의 API를 이용해서 Slack Bot 생성
                    api_url = "https://hooks.slack.com/services/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"

                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36",
                        "Content-type": "application/json",
                    }

                    # 햄스터가 나왔다는 메세지를 전송하기 위한 json 코드 작성
                    msg = {
                        "attachments": [
                            {
                                "text": "야생의 햄스터 출현!!",
                                "fallback": "🐹: 햄찌 나와쪄염~",
                                "color": "#3AA3E3",
                                "attachment_type": "default",
                                "actions": [
                                    {
                                        "text": "햄스터 보러가기",
                                        "type": "button",
                                        "url": "https://mobleqr.iptime.org:443",
                                    },
                                ],
                            }
                        ],
                    }

                    res = requests.post(api_url, headers=headers, data=json.dumps(msg))
                    res.raise_for_status()

                    # 메세지 전송 성공/실패 여부를 터미널에 표시
                    if res.status_code == 200:
                        print("Post Recieved")
                    else:
                        print("Failed to Recieve")
                    # 메세지를 전송하는동안 중복전송을 막기 위해 1초 정지
                    time.sleep(1)

                # 30분에 한번씩 Trigger를 True로 초기화 시켜서 메세지를 보낼 수 있는 상태 전환
                if now.localtime().tm_min % 30 == 0 and now.localtime().tm_sec == 0:
                    Trigger = True
                # print(Trigger, now.localtime().tm_min, now.localtime().tm_sec)

                # 햄스터를 검출하는 사각형에 정확도를 표시
                cvzone.putTextRect(
                    frame,
                    f"hamster {conf}%",
                    (max(0, x1), max(35, y1)),
                    scale=1,
                    thickness=1,
                )

        # 이미지를 jpg 형식으로 인코딩한 뒤 웹으로 전송
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# flask 서버 생성
@app.route("/")
def index():
    return render_template("test.html")


# flask 서버로 영상을 전송
@app.route("/video_feed")
def video_feed():
    return Response(
        detect_objects(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    # http -> https 전환을 위한 OpenSSL 인증서 인증
    ssl_context1 = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context1.load_cert_chain(
        certfile=r"SSL\server.crt",
        keyfile=r"SSL\server.key",
        password="1234",
    )
    # flask 서버 구동
    app.run(
        host="0.0.0.0",
        ssl_context=ssl_context1,
        debug=False,
        port=443,
        use_reloader=False,
    )
