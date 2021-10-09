# Kaushik Goud Chandapet
# 1536199, Mechatronics
# University of Siegen, Germany

# Remote Photoplethysmography
# Version 1.0 (August 2021)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from imutils.video import VideoStream
import pickle
import heartpy as hp
import pandas as pd
import scipy.signal as sig
from xlsxwriter import Workbook
from scipy import signal
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

plt.ion()  # Set interactive mode on
fig = plt.figure(1)
plt.xlabel("Time(ms)")
plt.ylabel("Pixels")

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Excel parameters
time_stamp = []
blue = []
red = []
green = []

# plotting parameters
b_plot = []
g_plot = []
r_plot = []
t_plot = []

# Get source_mp4 Video, Ground truth values
source_mp4 = '01-base.mp4'
original_data = '01-base PPG.csv'
df_original_HR = pd.read_csv(original_data, index_col=None)

# Using Video-capture to get the fps value.
capture = cv2.VideoCapture(source_mp4)
# capture = cv2.VideoCapture(0)
fps = capture.get(cv2.CAP_PROP_FPS)
capture.release()

# Using Video-Stream to continuously run Webcam
# vs = VideoStream().start()

# Using Video-capture to run video file
cap = cv2.VideoCapture(source_mp4)

frame_count = 0  # frames count
time_count = 0  # time in milliseconds
update = 0  # plot update

is_update = False


# Butterworth forward-backward band-pass filter
def bandpass(signal, fs, order, fc_low, fc_hig, debug=False):
    """Butterworth forward-backward band-pass filter.

    :param signal: list of ints or floats; The vector containing the signal samples.
    :param fs: float; The sampling frequency in Hz.
    :param order: int; The order of the filter.
    :param fc_low: int or float; The lower cutoff frequency of the filter.
    :param fc_hig: int or float; The upper cutoff frequency of the filter.
    :param debug: bool, default=False; Flag to enable the debug mode that prints additional information.

    :return: list of floats; The filtered signal.
    """
    nyq = 0.5 * fs  # Calculate the Nyquist frequency.
    cut_low = fc_low / nyq  # Calculate the lower cutoff frequency (-3 dB).
    cut_hig = fc_hig / nyq  # Calculate the upper cutoff frequency (-3 dB).
    bp_b, bp_a = sig.butter(order, (cut_low, cut_hig), btype="bandpass")  # Design and apply the band-pass filter.
    bp_data = list(sig.filtfilt(bp_b, bp_a, signal))  # Apply forward-backward filter with linear phase.
    return bp_data


# Fast Fourier Transform
def fft(data, fs, scale="mag"):
    # Apply Hanning window function to the data.
    data_win = data * np.hanning(len(data))
    if scale == "mag":  # Select magnitude scale.
        mag = 2.0 * np.abs(np.fft.rfft(tuple(data_win)) / len(data_win))  # Single-sided DFT -> FFT
    elif scale == "pwr":  # Select power scale.
        mag = np.abs(np.fft.rfft(tuple(data_win))) ** 2  # Spectral power
    bin = np.fft.rfftfreq(len(data_win), d=1.0 / fs)  # Calculate bins, single-sided
    return bin, mag


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        # image = vs.read()
        success, image = cap.read()
        if image is None:
            # If loading a video, use 'break' instead of 'continue'.
            # continue
            break
        height, width, _ = image.shape
        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        processed_img = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert the RGB image to BGR.
        if processed_img.multi_face_landmarks:
            for face_landmarks in processed_img.multi_face_landmarks:
                landmark_points = []
                for i in range(0, 468):
                    x = int(face_landmarks.landmark[i].x * width)
                    y = int(face_landmarks.landmark[i].y * height)
                    p = [x, y]
                    landmark_points.append([x, y])
                # Set ROI points
                forehead = np.array((
                    landmark_points[9], landmark_points[107], landmark_points[66], landmark_points[105],
                    landmark_points[104], landmark_points[103],
                    landmark_points[67], landmark_points[109], landmark_points[10],
                    landmark_points[338], landmark_points[297], landmark_points[332],
                    landmark_points[333], landmark_points[334], landmark_points[296],
                    landmark_points[336]))
                left_cheek = np.array((landmark_points[266], landmark_points[426], landmark_points[436],
                                       landmark_points[416], landmark_points[376],
                                       landmark_points[352], landmark_points[347], landmark_points[330]))
                right_cheek = np.array((landmark_points[36], landmark_points[206], landmark_points[216],
                                        landmark_points[192], landmark_points[147],
                                        landmark_points[123], landmark_points[117], landmark_points[118],
                                        landmark_points[101]))
                # Alternate ROIs
                left_cheek1 = np.array((landmark_points[266], landmark_points[426], landmark_points[411],
                                        landmark_points[346], landmark_points[347], landmark_points[330]))
                right_cheek1 = np.array((landmark_points[36], landmark_points[206], landmark_points[187],
                                         landmark_points[117], landmark_points[118], landmark_points[101]))

                # Draw ROI's on the image
                cv2.polylines(image, [forehead], True, (0, 255, 255), 2)
                cv2.polylines(image, [left_cheek], True, (0, 255, 255), 2)
                cv2.polylines(image, [right_cheek], True, (0, 255, 255), 2)

                # mask the image and crop the ROI with black background
                mask = np.zeros((height, width), dtype=np.uint8)
                # cv2.fillPoly(mask, [forehead, left_cheek, right_cheek], (255))
                # cv2.fillPoly(mask, [left_cheek, right_cheek], (255))
                cv2.fillPoly(mask, [forehead], (255))
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
                t_plot.append(round(time_count))
                time_count += (1000 / fps)

                # Draw the face mesh on the image
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                cv2.imshow('MediaPipe FaceMesh', image)
                # cv2.imshow('MediaPipe Masked pixel crop', crop_img)

                # Plot the graph 4 times a sec (15 new records each time)
                if frame_count % 15 == 0:
                    is_update = True  # New frame has come

                    # plot the RGB signals
                    plt.plot(t_plot, b_plot, 'b', label='Blue')
                    plt.plot(t_plot, g_plot, 'g', label='Green')
                    plt.plot(t_plot, r_plot, 'r', label='Red')
                    plt.pause(0.01)
                    # Save the plot as png file
                    fig.savefig('rPPGplotVideo.png', dpi=100)
                    update += 1

                elif update > 2:
                    # After 3 plots push the reading to Excel parameters and clear plotting parameters
                    if is_update:
                        if update == 3:
                            blue.extend(b_plot)
                            green.extend(g_plot)
                            red.extend(r_plot)
                            time_stamp.extend(t_plot)
                        else:
                            blue.extend(b_plot[(len(b_plot) - 15):len(b_plot)])
                            green.extend(g_plot[(len(g_plot) - 15):len(g_plot)])
                            red.extend(r_plot[(len(r_plot) - 15):len(r_plot)])
                            time_stamp.extend(t_plot[(len(t_plot) - 15):len(t_plot)])

                        del b_plot[0:15]
                        del g_plot[0:15]
                        del r_plot[0:15]
                        del t_plot[0:15]

                        is_update = False  # we added the new frame to our list structure

        # Break using esc key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    capture.release()
    # vs.stop()

    # Hold plot and plot the filtered signals
    plt.ioff()
    # 2nd order butterworth bandpass filtering
    bp_r_plot = bandpass(red, fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz), taking 30-150 (0.5 - 2.5)
    bp_g_plot = bandpass(green, fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz), taking 30-150 (0.5 - 2.5)
    bp_b_plot = bandpass(blue, fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz), taking 30-150 (0.5 - 2.5)
    # plt.plot(time_stamp, bp_r_plot, 'r', label='BPFiltered_Red')
    plt.plot(time_stamp, bp_g_plot, 'g', label='BPFiltered_Green')
    # plt.plot(time_stamp, bp_b_plot, 'b', label='BPFiltered_Blue')
    plt.title("Raw and Filtered Signals")
    # plt.legend()
    # plt.show()

    # Calculate and display FFT
    X_fft, Y_fft = fft(bp_g_plot, fps, scale="mag")
    fig2 = plt.figure(2)
    plt.plot(X_fft, Y_fft)
    plt.title("FFT of filtered Signal")
    fig2.savefig('FFTplotVideo.png', dpi=100)
    # plt.show()

    # Welch's Periodogram
    f_set, Pxx_den = signal.welch(bp_g_plot, fps)
    fig3 = plt.figure(3)
    plt.semilogy(f_set, Pxx_den)
    plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title("Welchplot")
    fig3.savefig('WelchplotVideo.png', dpi=100)
    # plt.show()

    # Calculate Heart Rate and Plot
    working_data, measures = hp.process(bp_g_plot, fps)
    plot_object = hp.plotter(working_data, measures, show=False, title='Final_Heart Rate Signal Peak Detection')
    plot_object.savefig('bpmPlotVideo.png', dpi=100)
    peaks = [0] * len(working_data['hr'])
    for p, q in zip(working_data['peaklist'], working_data['binary_peaklist']):
        if q == 1:
            peaks[p] = 1

    # Calculate and display original FFT
    X_fft, Y_fft = fft(df_original_HR['Signal'], fps, scale="mag")
    fig4 = plt.figure(4)
    plt.plot(X_fft, Y_fft)
    plt.title("FFT of Original Signal")
    fig4.savefig('Original_FFTplotVideo.png', dpi=100)
    # plt.show()

    # Original Welch's Periodogram
    f_set, Pxx_den = signal.welch(df_original_HR['Signal'], fps)
    fig5 = plt.figure(5)
    plt.semilogy(f_set, Pxx_den)
    plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.title("Original_Welchplot")
    fig5.savefig('Original_WelchplotVideo.png', dpi=100)
    # plt.show()

    # Calculate Original Heart Rate and Plot
    hrdata = hp.get_data(original_data, column_name='Signal')
    timerdata = hp.get_data(original_data, column_name='Time')
    working_data1, measures1 = hp.process(hrdata, hp.get_samplerate_mstimer(timerdata))
    plot_object1 = hp.plotter(working_data1, measures1, show=False, title='Original_Heart Rate Signal Peak Detection')
    plot_object1.savefig('bpmPlotOriginal.png', dpi=100)
    plt.show()

    # calculate Evaluation metrics
    Accuracy = accuracy_score(df_original_HR['Peaks'][0:len(peaks)], peaks)
    Metrics = precision_recall_fscore_support(df_original_HR['Peaks'][0:len(peaks)], peaks)
    Precision, Recall, f1_score = Metrics[0][0], Metrics[1][0], Metrics[2][0]

    print('Accuracy:', Accuracy, 'Precision:', Precision, 'Recall:', Recall, 'f1_score:', f1_score)

    # Export Heart rate to Excel file
    book = Workbook('Heartrate_video.xlsx')
    sheet = book.add_worksheet()
    row = 0
    col = 0

    sheet.write(row, col, 'Time')
    sheet.write(row, col + 1, 'Signal')
    sheet.write(row, col + 2, 'Peaks')
    row += 1

    for f, b, g in zip(time_stamp, working_data['hr'], peaks):
        sheet.write(row, col, f)
        sheet.write(row, col + 1, b)
        sheet.write(row, col + 2, g)
        row += 1
    book.close()

    # Export to pickle byte stream object
    tabular_rgb = np.array([np.array(time_stamp), np.array(red), np.array(green), np.array(blue)])
    pickle.dump(tabular_rgb, open("video_save.p", "wb"))
    # _pickle_imported_obj = pickle.load(open("save.p", "rb"))

    # Export to Excel file
    book = Workbook('rPPGVideo.xlsx')
    sheet = book.add_worksheet()
    row = 0
    col = 0

    sheet.write(row, col, 'Time')
    sheet.write(row, col + 1, 'Blue mean')
    sheet.write(row, col + 2, 'Green mean')
    sheet.write(row, col + 3, 'Red mean')
    row += 1

    for f, b, g, r in zip(time_stamp, blue, green, red):
        sheet.write(row, col, f)
        sheet.write(row, col + 1, b)
        sheet.write(row, col + 2, g)
        sheet.write(row, col + 3, r)
        row += 1
    book.close()
