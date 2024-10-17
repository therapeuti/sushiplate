from flask import Flask, render_template, jsonify, request
from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
import base64

app = Flask(__name__, template_folder='template')

# YOLO 모델 초기화
# model = YOLO("sushi_yolov8l_8.pt")
# model = YOLO("sushi9or11s.pt")
model = YOLO("yolo11s-1010.pt")
# model = YOLO("model_- 10 october 2024 15_07.torchscript.pt")
# model = YOLO("model_- 10 october 2024 15_07.tflite")
# model = YOLO("best_float32.tflite")


detect_enabled = False  # 탐지 활성화 여부

# 회전 초밥 접시별 가격
plate_prices = {
    'black': 10000,
    'blue': 9000,
    'green': 8000,
    'orange': 7000,
    'orange-rec': 6000,
    'orange-vivid': 5000,
    'purple': 4000,
    'red': 3000,
    'yellow': 2000,
    'yellow-rec': 1000
}

inference_results = defaultdict(int)
total_price = 0


@app.route('/')
def index():
    return render_template('index-new.html')


@app.route('/inference', methods=['POST'])
def inference():
    global detect_enabled, inference_results, total_price
    detect_enabled = True  # 탐지 활성화

    # 클라이언트에서 전달된 이미지 받기
    image_data = request.json['image']
    image = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # YOLO 추론
    results = model.predict(frame, save=False, conf=0.6, verbose=False)

    inference_results = []  # 리스트로 변경
    total_price = 0

    # 결과 처리
    if results:
        for box in results[0].boxes:
            class_id = int(box.cls)  # 객체 클래스 ID
            label = model.names[class_id]  # 클래스 id와 매핑되는 라벨명
            confidence = box.conf.item()  # 탐지 신뢰도
            coords = box.xyxy[0].cpu().numpy().tolist()  # 바운딩 박스 좌표를 리스트로 변환

            # 각 객체마다 새로운 엔트리 생성
            inference_results.append({
                "label": label,
                "confidence": confidence,
                "coords": coords,
                "price": plate_prices[label]
            })

            total_price += plate_prices[label]

    # 라벨별 집계 계산
    label_summary = defaultdict(lambda: {"count": 0, "total": 0})
    for result in inference_results:
        label = result["label"]
        label_summary[label]["count"] += 1
        label_summary[label]["total"] += result["price"]

    return jsonify({
        "results": inference_results,
        "summary": [
            {
                "label": label,
                "count": summary["count"],
                "price": plate_prices[label],
                "total": summary["total"]
            }
            for label, summary in label_summary.items()
        ],
        "total_price": total_price
    })


@app.route('/stop_inference')  # 추론 종료 버튼 누르면 실행
def stop_inference():
    global detect_enabled
    detect_enabled = False  # 탐지 비활성화
    return jsonify({"message": "Inference stopped."})


@app.route('/get_results')  # 추가된 엔드포인트
def get_results():
    global inference_results, total_price
    results = []
    for label, count in inference_results.items():
        results.append({
            "label": label,
            "count": count,
            "price": plate_prices[label],
            "total": count * plate_prices[label]
        })
    return jsonify({
        "results": results,
        "total_price": total_price
    })


@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8080, debug=True)
    port = int(os.environ.get('PORT', 8080))  # Cloud Run이 지정한 포트를 사용
    app.run(host='0.0.0.0', port=port, debug=False)