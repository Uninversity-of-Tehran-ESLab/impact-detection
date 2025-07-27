#!/bin/bash
python -m ultralytics.cfg train model=yolov10n.pt data=tennis_ball.yaml epochs=200 batch=32 imgsz=640