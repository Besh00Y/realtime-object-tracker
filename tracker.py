import cv2
import argparse
from ultralytics import YOLO
from utils import calculate_iou




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="Path to video file, or 0 for webcam"
    )
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)

    output_width = 1280
    output_height = 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        "output_tracking.mp4",
        fourcc,
        30.0,
        (output_width, output_height)
)
    if not cap.isOpened():
        print(f"Error: Could not open source: {source}")
        return

    ret, first_frame = cap.read()

    if not ret:
        print("Error: Could not read first frame.")
        return

    cv2.namedWindow("Select Object", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Object", 1280, 720)
 
    bbox = cv2.selectROI(
        "Select Object",
        first_frame,
        fromCenter=False
    )

    cv2.destroyWindow("Select Object")

    print("Selected box (x, y, w, h):", bbox)

 
    model = YOLO("yolov8n.pt")

 
    results = model.track(
        first_frame,
        persist=True
    )[0]
 
    x, y, w, h = bbox

    user_box = (
        x,
        y,
        x + w,
        y + h
    )

 
    best_iou = 0
    target_id = None

    for box in results.boxes:

        detection_box = box.xyxy[0].tolist()

        iou = calculate_iou(
            user_box,
            detection_box
        )

        if iou > best_iou:

            best_iou = iou

            if box.id is not None:
                target_id = int(box.id[0])
            else:
                target_id = None

    print("Target ID:", target_id)
    print("Best IoU:", best_iou)

 
    if target_id is None:
        print("Could not find a tracking ID for the selected object.")
        cap.release()
        cv2.destroyAllWindows()
        return


 
    while True:

        ret, frame = cap.read()

        if not ret:
            print("End of video or cannot read frame.")
            break

 
        results = model.track( frame, persist=True )[0]
 
        found = False

 
        for box in results.boxes:

            if box.id is not None and int(box.id[0]) == target_id:

                found = True 
                x1, y1, x2, y2 = box.xyxy[0].tolist() 
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
 
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

 
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id] 
                label = f"{cls_name} | ID: {target_id}" 

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                break
 
        if not found:

            cv2.putText(
                frame,
                "Target Lost",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
 
        display_width = 1280 
        scale = display_width / frame.shape[1] 
        display_height = int(frame.shape[0] * scale) 
        display_frame = cv2.resize(
            frame,
            (display_width, display_height)
        )

        out.write(display_frame) 
        cv2.imshow(
            "Real-Time Object Tracker",
            display_frame
        ) 
  
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 

    out.release()  
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()