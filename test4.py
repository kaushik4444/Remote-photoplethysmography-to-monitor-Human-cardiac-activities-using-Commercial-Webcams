import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from imutils.video import VideoStream
from xlsxwriter import Workbook

plt.ion()  # Set interactive mode on
fig = plt.figure()
plt.legend()
plt.xlabel("Frames")
plt.ylabel("Pixels")

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

fr = []
blue = []
red = []
green = []

# plotting parameters
b_plot = []
g_plot = []
r_plot = []
f_plot = []

# We will be using Video-capture to get the fps value.
capture = cv2.VideoCapture(0)
fps = capture.get(cv2.CAP_PROP_FPS)
capture.release()
print(fps)

# New module: VideoStream
vs = VideoStream(src = "WIN_20210712_16_25_07_Pro.mp4", framerate=fps).start()

frame_count = 0
update = 0

is_update = False

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
  while True:
    image = vs.read()
    if image is None:
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
          cv2.polylines(image, [forehead], True, (0, 255, 255), 2)
          leftcheek = np.array((landmark_points[266], landmark_points[426], landmark_points[436], landmark_points[416], landmark_points[376],
                          landmark_points[352], landmark_points[347], landmark_points[330]))
          leftcheek1 = np.array((landmark_points[266], landmark_points[426], landmark_points[411],
                                 landmark_points[346], landmark_points[347], landmark_points[330]))
          cv2.polylines(image, [leftcheek1], True, (0, 255, 255), 2)
          rightcheek = np.array((landmark_points[36], landmark_points[206], landmark_points[216], landmark_points[192], landmark_points[147],
                          landmark_points[123], landmark_points[117], landmark_points[118], landmark_points[101]))
          rightcheek1 = np.array((landmark_points[36], landmark_points[206], landmark_points[187],
                                 landmark_points[117], landmark_points[118], landmark_points[101]))
          cv2.polylines(image, [rightcheek1], True, (0, 255, 255), 2)

          # mask the image and crop the ROI with black background
          mask = np.zeros((height, width), dtype=np.uint8)
          cv2.fillPoly(mask, [forehead,leftcheek1,rightcheek1], (255))
          crop_img = cv2.bitwise_and(image, image, mask=mask)

          # eliminate the black pixels and get mean of RGB for each frame
          b, g, r = cv2.split(crop_img)
          indices_list = np.where(np.any(crop_img != [0, 0, 0], axis=-1))
          roi_pixel_img = crop_img[indices_list]

          # Append the current frame's RGB to plotting parameters
          b_plot.append(roi_pixel_img[:, 0].mean())
          g_plot.append(roi_pixel_img[:, 1].mean())
          r_plot.append(roi_pixel_img[:, 2].mean())
          frame_count += 1
          f_plot.append(frame_count)

          mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
          cv2.imshow('MediaPipe FaceMesh', image)
          cv2.imshow('MediaPipe Masked pixel crop', crop_img)

          if frame_count % int(fps/3) == 0:
              is_update = True  # New frame has come

              # plot the RGB signals
              plt.plot(f_plot, b_plot, 'b', label='Blue')
              plt.plot(f_plot, g_plot, 'g', label='Green')
              plt.plot(f_plot, r_plot, 'r', label='Red')
              plt.pause(0.01)
              fig.savefig('rPPGplotVideomain.png', dpi=100)
              update += 1

          elif update > 2:
              if is_update:
                  if update == 3:
                      blue.extend(b_plot)
                      green.extend(g_plot)
                      red.extend(r_plot)
                      fr.extend(f_plot)
                  else:
                      blue.extend(b_plot[(len(b_plot) - int(fps/3)):(len(b_plot) - 1)])
                      green.extend(g_plot[(len(g_plot) - int(fps/3)):(len(g_plot) - 1)])
                      red.extend(r_plot[(len(r_plot) - int(fps/3)):(len(r_plot) - 1)])
                      fr.extend(f_plot[(len(f_plot) - int(fps/3)):(len(f_plot) - 1)])

                  del b_plot[0:int(fps/3)]
                  del g_plot[0:int(fps/3)]
                  del r_plot[0:int(fps/3)]
                  del f_plot[0:int(fps/3)]

                  is_update = False  # we added the new frame to our list structure

    if cv2.waitKey(1) & 0xFF == 27:
        break

  #cv2.destroyAllWindows()
  capture.release()
  vs.stop()

  book = Workbook('rPPGVideomain.xlsx')
  sheet = book.add_worksheet()

  row = 0
  col = 0

  sheet.write(row, col, 'Frames')
  sheet.write(row + 1, col, 'Blue mean')
  sheet.write(row + 2, col, 'Green mean')
  sheet.write(row + 3, col, 'Red mean')

  col += 1

  for f, b, g, r in zip(fr, blue, green, red):
      sheet.write(row, col, f)
      sheet.write(row + 1, col, b)
      sheet.write(row + 2, col, g)
      sheet.write(row + 3, col, r)
      col += 1

  book.close()