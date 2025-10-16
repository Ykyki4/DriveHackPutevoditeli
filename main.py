import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


TARGET_WIDTH = 15
TARGET_HEIGHT = 150

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class VehicleBehavior:
    def __init__(self, tracker_id: int, fps: float):
        self.id = tracker_id
        self.fps = fps
        self.x_positions = deque(maxlen=int(fps * 2))
        self.speeds = deque(maxlen=int(fps * 2))
        self.danger_score = 0

    def update(self, x: float, speed_kmh: float):
        self.x_positions.append(x)
        self.speeds.append(speed_kmh)

        if len(self.speeds) >= 2:
            v1 = self.speeds[-2] / 3.6  # м/с
            v2 = self.speeds[-1] / 3.6  # м/с
            acceleration = v2 - v1  # м/с²

            if acceleration < -10.0:  # резкое торможение
                self.danger_score += 1
            elif acceleration > 10.0:  # резкое ускорение
                self.danger_score += 1

    def is_aggressive(self) -> bool:
        return self.danger_score >= 3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Behavior Analysis with Aggression and Tailgating Detection"
    )
    parser.add_argument(
        "--source_video_path",
        default="input.mp4",
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default="output.mp4",
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    SOURCE = np.array([[500, 220], [900, 220], [900, 1000], [-1100, 1000]])

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    model = YOLO("yolo11x.pt")

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=polygon_zone,
        color=sv.Color.GREEN,
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
    )
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    vehicle_behaviors = defaultdict(lambda: None)

    all_tracked_ids = set()
    aggressive_ids = set()
    completed_speeds = []
    previous_tracker_ids = set()

    # Порог дистанции в пикселях в трансформированном пространстве (TARGET)

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=args.iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_points = view_transformer.transform_points(points=points).astype(float)

            for tracker_id, [_, y] in zip(detections.tracker_id, transformed_points):
                coordinates[tracker_id].append(y)

            current_speeds = {}
            for tracker_id in detections.tracker_id:
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                current_speeds[tracker_id] = speed

            labels = []
            for i, tracker_id in enumerate(detections.tracker_id):
                x_center = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)[i][0]

                if vehicle_behaviors[tracker_id] is None:
                    vehicle_behaviors[tracker_id] = VehicleBehavior(tracker_id, video_info.fps)

                vehicle_behaviors[tracker_id].update(x_center, current_speeds[tracker_id])

                all_tracked_ids.add(tracker_id)
                if vehicle_behaviors[tracker_id].is_aggressive():
                    aggressive_ids.add(tracker_id)

                speed_text = f"{int(current_speeds[tracker_id])} km/h" if current_speeds[tracker_id] > 0 else ""
                danger_mark = "!" if tracker_id in aggressive_ids else ""
                labels.append(f"#{tracker_id} {speed_text}{danger_mark}")

            # --- Проверка дистанции между машинами (tailgating) ---
            active_tids = detections.tracker_id
            transformed_ys = [pt[1] for pt in transformed_points]  # только Y в трансформированной системе

            # Словарь: tid -> y
            tid_to_y = {tid: y for tid, y in zip(active_tids, transformed_ys)}

            # --- Обработка завершённых треков ---
            current_tracker_ids = set(detections.tracker_id)
            lost_trackers = previous_tracker_ids - current_tracker_ids

            for tid in lost_trackers:
                if tid in vehicle_behaviors and vehicle_behaviors[tid].speeds:
                    last_speed = vehicle_behaviors[tid].speeds[-1]
                    if last_speed > 0:
                        completed_speeds.append(last_speed)

            previous_tracker_ids = current_tracker_ids.copy()
            # ----------------------------------------------------------

            # --- Цвета ---
            box_colors = []
            label_colors = []
            for tid in detections.tracker_id:
                if tid in aggressive_ids:
                    box_colors.append(sv.Color.RED)
                    label_colors.append(sv.Color.RED)
                else:
                    box_colors.append(sv.Color.GREEN)
                    label_colors.append(sv.Color.GREEN)

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            annotated_frame = zone_annotator.annotate(scene=annotated_frame)

            # --- Статистика за всё время ---
            total_vehicles = len(all_tracked_ids)
            avg_speed = np.mean(completed_speeds) if completed_speeds else 0.0
            aggressive_percent = (len(aggressive_ids) / total_vehicles * 100) if total_vehicles > 0 else 0.0

            cv2.putText(
                annotated_frame,
                f"Vehicles: {total_vehicles} | Avg Speed: {avg_speed:.1f} km/h",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 255, 0),
                thickness=2,
            )
            cv2.putText(
                annotated_frame,
                f"Aggressive (% total): {aggressive_percent:.1f}%",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255),
                thickness=2,
            )
            # ------------------------------

            sink.write_frame(annotated_frame)
            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    # --- Финальная обработка ---
    for tid, behavior in vehicle_behaviors.items():
        all_tracked_ids.add(tid)
        if behavior and behavior.is_aggressive():
            aggressive_ids.add(tid)
        if tid not in previous_tracker_ids and behavior and behavior.speeds:
            completed_speeds.append(behavior.speeds[-1])

    total_final = len(all_tracked_ids)
    aggressive_final = len(aggressive_ids)
    avg_final = np.mean(completed_speeds) if completed_speeds else 0.0
    percent_final = (aggressive_final / total_final * 100) if total_final > 0 else 0.0

    print(f"Final Stats — Vehicles: {total_final}, Aggressive: {aggressive_final} ({percent_final:.1f}%), Avg Speed: {avg_final:.2f} km/h")