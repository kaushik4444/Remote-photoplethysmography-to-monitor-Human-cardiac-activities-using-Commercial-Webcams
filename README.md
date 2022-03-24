# Development of rPPG algorithm to monitor Human cardiac activities using Commercial Webcams

Remote photoplethysmography (rPPG) is a contactless method to monitor human cardiac activities by detecting the pulse-induced subtle color variations on the human skin surface using a multi-wavelength RGB camera. By measuring the variance of red, green, and blue light reflection changes from the skin, as the contrast between specular reflection and diffused reflection.

• Developed a Face detector and carried out real-time face tracking.
• Implemented adaptive skin segmentation technique to filter out Non-skin pixels from the image using a Region of Interest (ROI) mask.
• Extracted the pulse information by applying the Plane orthogonal to skin-tone technique, Butterworth Bandpass filters, and the Fast Fourier transformations.
• Achieved an overall accuracy of around 98% with an f1_score of 70%.
