import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

b_plot = []
g_plot = []
r_plot = []

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture("WIN_20210712_16_25_07_Pro.mp4")
#cap = cv2.VideoCapture("GX011050.MP4")
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
    height, width, _ = image.shape

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    crop_imgults = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if crop_imgults.multi_face_landmarks:
      for face_landmarks in crop_imgults.multi_face_landmarks:
          landmark_points = []
          for i in range(0, 468):
              x = int(face_landmarks.landmark[i].x * width)
              y = int(face_landmarks.landmark[i].y * height)
              p = [x, y]
              landmark_points.append([x, y])
          forehead = np.array((landmark_points[9], landmark_points[107], landmark_points[66], landmark_points[105], landmark_points[104], landmark_points[103],
                               landmark_points[67], landmark_points[109], landmark_points[10], landmark_points[338], landmark_points[297], landmark_points[332],
                               landmark_points[333], landmark_points[334], landmark_points[296], landmark_points[336]))
          #cv2.polylines(image, [forehead], True, (0, 255, 255), 2)
          leftcheek = np.array((landmark_points[266], landmark_points[426], landmark_points[436], landmark_points[416], landmark_points[376],
                          landmark_points[352], landmark_points[347], landmark_points[330]))
          leftcheek1 = np.array((landmark_points[266], landmark_points[426], landmark_points[411],
                                 landmark_points[346], landmark_points[347], landmark_points[330]))
          #cv2.polylines(image, [leftcheek1], True, (0, 255, 255), 2)
          rightcheek = np.array((landmark_points[36], landmark_points[206], landmark_points[216], landmark_points[192], landmark_points[147],
                          landmark_points[123], landmark_points[117], landmark_points[118], landmark_points[101]))
          rightcheek1 = np.array((landmark_points[36], landmark_points[206], landmark_points[187],
                                 landmark_points[117], landmark_points[118], landmark_points[101]))
          #cv2.polylines(image, [rightcheek1], True, (0, 255, 255), 2)


          mask = np.zeros((height, width), dtype=np.uint8)
          cv2.fillPoly(mask, [forehead,leftcheek1,rightcheek1], (255))
          crop_img = cv2.bitwise_and(image, image, mask=mask)

          b, g, r = cv2.split(crop_img)
          indices_list = np.where(np.any(crop_img != [0, 0, 0], axis=-1))
          roi_pixel_img = crop_img[indices_list]
          b_pixels = roi_pixel_img[:, 0].mean()
          g_pixels = roi_pixel_img[:, 1].mean()
          r_pixels = roi_pixel_img[:, 2].mean()

          b_plot.append(b_pixels)
          g_plot.append(g_pixels)
          r_plot.append(r_pixels)

          mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    cv2.imshow('MediaPipe FaceMesh', image)
    cv2.imshow('MediaPipe Masked pixel crop', crop_img)
    if cv2.waitKey(1) & 0xFF == 27:
      break

plt.plot(b_plot, 'b', label='Blue')
plt.plot(g_plot, 'g', label='Green')
plt.plot(r_plot, 'r', label='Red')
plt.xlabel("Frames")
plt.ylabel("Pixels")
plt.legend()
plt.show()
cap.release()

