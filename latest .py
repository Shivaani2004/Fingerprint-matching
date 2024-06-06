import os
import cv2
from tkinter import Tk, filedialog

# Create a Tkinter root window
root = Tk()
root.withdraw() # Hide the root window

# Ask the user to select an image file using a file dialog
image_path = filedialog.askopenfilename(title="Select Image",
                                        filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")))

# Check if a file was selected
if image_path:
    # Load the selected image
    sample = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if sample is None:
        print("Error: Failed to load the sample image.")
    else:
        best_score = 0
        best_match = None
        best_image = None
        kp1, kp2, mp = None, None, None
        counter =0

        for file in os.listdir("SOCOFing/Real")[:1000]:
            fingerprint_image = cv2.imread("SOCOFing/Real/" + file)
            if counter%10 ==0:
                  print(counter)
                  print(file)
            counter += 1
            fingerprint_image = cv2.imread("SOCOFing/Real/" + file)

            if fingerprint_image is None:
                print(f"Error: Failed to load fingerprint image '{file}'. Skipping.")
                continue
                
            sift = cv2.SIFT_create()
            keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

            # FLANN matcher
            matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(
                descriptors_1, descriptors_2, k=2
            )

            match_points = []
            for p, q in matches:
                if p.distance < 0.1 * q.distance:
                    match_points.append(p)

            keypoints = min(len(keypoints_1), len(keypoints_2))
            match_ratio = len(match_points) / keypoints * 100

            if match_ratio > best_score:
                best_score = match_ratio
                best_match = file
                best_image = fingerprint_image
                kp1, kp2, mp = keypoints_1, keypoints_2, match_points

        if best_match is not None:
            print("Best match: ", best_match)
            print("Best score: ", best_score)
            # Draw matches between sample and best match
            result = cv2.drawMatches(sample, kp1, best_image, kp2, mp, None)
            result = cv2.resize(result, None, fx=5, fy=5)
            cv2.imshow("Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No match found.")
else:
    print("Error: No image selected.")
