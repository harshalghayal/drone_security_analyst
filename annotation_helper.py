import cv2
import numpy as np

polygon_points = []

def draw_polygon(event, x, y, flags, param):
    global polygon_points

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN and len(polygon_points) > 2:
        print("Polygon Coordinates:", polygon_points)

# Load image
frame = cv2.imread('Data/sequences/uav0000075_00000_v/0000001.jpg')
original_frame = frame.copy()

cv2.namedWindow('Draw Polygon', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Draw Polygon', draw_polygon)

while True:
    display_frame = original_frame.copy()

    for i, point in enumerate(polygon_points):
        cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
        if i > 0:
            cv2.line(display_frame, polygon_points[i - 1], point, (255, 0, 0), 2)
    if len(polygon_points) > 2:
        cv2.line(display_frame, polygon_points[-1], polygon_points[0], (255, 0, 0), 2)

    cv2.imshow('Draw Polygon', display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
