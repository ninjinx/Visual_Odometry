import cv2
import numpy as np

class Camera:
    def __init__(self, x, y, rot=0, f=10, col=(0, 255, 0)):
        self.x = x
        self.y = y
        self.r = rot
        self.f = f
        self.color = col

    def draw(self, im):
        cv2.line(im, (int(self.x), int(self.y)), (int(self.x - 10), int(self.y + self.f)), self.color, 1)
        cv2.line(im, (int(self.x), int(self.y)), (int(self.x + 10), int(self.y + self.f)), self.color, 1)
        cv2.line(im, (int(self.x - 10), int(self.y + self.f)), (int(self.x + 10), int(self.y + self.f)), self.color, 1)

    def draw_lines(self, im, points):
        for p in points:
            px = p[0]
            py = p[1]
            cv2.line(im, (int(self.x), int(self.y)), (int(px), int(py)), self.color, 1)

h = 480
w = 640
canvas = np.zeros((h, w, 3), dtype=np.uint8)

cam1 = Camera(w/2, h/2)
cam2 = Camera(64+w/2, h/2-32, col=(0, 128, 255))

points = []
points.append((cam1.x+10, cam1.y+80))
points.append((cam1.x-10, cam1.y+180))
points.append((cam1.x+2, cam1.y+60))
points.append((cam1.x-15, cam1.y+120))

while True:
    cv2.rectangle(canvas, (0, 0), (w, h), (0, 0, 0), -1)
    cam1.draw(canvas)
    cam1.draw_lines(canvas, points)

    cam2.draw(canvas)
    cam2.draw_lines(canvas, points)

    cv2.imshow("window", canvas)

    key = cv2.waitKey(33) & 0xFF

    if key == 81:
        cam2.x -= 5

    if key == 82:
        cam2.y -= 5

    if key == 83:
        cam2.x += 5

    if key == 84:
        cam2.y += 5

    if key == ord("q"):
        break
