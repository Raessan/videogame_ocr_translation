import cv2
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
import os
from PIL import Image
from params import *
from cnn_train import SimpleCNN, letterbox
import torch
from torchvision import datasets

# Class to perform Optical Character Recognition (OCR)
class OCR:
    def __init__(self, region, threshold_binary, min_zeros_between_lines, min_nonzeros_line, min_zeros_between_characters, min_zeros_space, folder_chars, chars_dict, size_characters, folder_model=None, name_model=None):
        # Region of the screen to capture
        self.region = region
        # Threshold for binarizing the image
        self.threshold_binary = threshold_binary
        # Minimum number of consecutive zeros to consider a line break
        self.min_zeros_between_lines = min_zeros_between_lines
        # Threshold of nonzeros in the vertical histogram to consider that an isolated part does not belong to a character (such as umlaut)
        self.min_nonzeros_line = min_nonzeros_line
        # Minimum number of consecutive zeros to consider a character break
        self.min_zeros_between_characters = min_zeros_between_characters
        # Minimum number of consecutive zeros to consider a space
        self.min_zeros_space = min_zeros_space
        # Folder containing character images
        self.folder_chars = folder_chars
        # Dictionary mapping characters to their filenames
        self.chars_dict = chars_dict
        # Size to which characters are resized
        self.size_characters = size_characters
        # Reverse dictionary for character lookup
        self.reversed_chars_dict = {v: k for k, v in self.chars_dict.items()}


        # Text variable that concatenates sentences until desired
        self.text = ""
        # Boolean variable to check if the last character of the last sentence ended with hyphen
        self.last_hyphen = False

        # Load NN model
        self.folder_model = folder_model
        self.name_model = name_model

        self.model = None
        self.device = "cpu"

        # Additional plotting data for optimizing parameters and debugging
        self.img_bw = None
        self.vertical_histogram = None
        self.horizontal_histogram = None

        if os.path.isfile(os.path.join(self.folder_model, self.name_model)):
            # Count the number of classes (subfolders)
            classes = os.listdir(self.folder_chars)
            num_classes = len(classes)
            # Create the model instance
            self.model = SimpleCNN(num_classes, self.size_characters)
            # Load the saved model weights
            self.model.load_state_dict(torch.load(os.path.join(self.folder_model, self.name_model)))
            # Set the model to evaluation mode
            self.model.eval()
            # If using GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            # To obtain the labels
            # Load the dataset
            full_dataset = datasets.ImageFolder(root=folder_chars)
            class_to_idx = full_dataset.class_to_idx
            self.labels = list(class_to_idx.keys())
            # print(self.labels)

    # Function that captures a screenshot with the specified region
    def capture_screen(self):
        # Capture screenshot
        screenshot = pyautogui.screenshot(region=self.region)
        # Color to BGR
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        return screenshot
    
    # Image thresholding to get binary image
    def threshold_img(self, img):
        # Convert image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply binary thresholding
        im_bw = cv2.threshold(img_gray, self.threshold_binary, 255, cv2.THRESH_BINARY_INV)[1]
        
        return im_bw/255

    def crop_binary_image_with_border(self, image):
        # Find the coordinates of all the 1s
        rows = np.any(image, axis=1)
        cols = np.any(image, axis=0)
        
        # Find the initial bounding box
        top, bottom = np.where(rows)[0][[0, -1]]
        left, right = np.where(cols)[0][[0, -1]]
        
        # Function to check if a row or column contains at least one "1"
        def contains_one(arr):
            return np.any(arr)

        # Ensure there is at least one "1" on each side
        while top > 0 and not contains_one(image[top]):
            top -= 1
        while bottom < image.shape[0] - 1 and not contains_one(image[bottom]):
            bottom += 1
        while left > 0 and not contains_one(image[:, left]):
            left -= 1
        while right < image.shape[1] - 1 and not contains_one(image[:, right]):
            right += 1
        
        # Crop the image to the bounding box
        cropped_image = image[top:bottom+1, left:right+1]
        
        # Return image and bounding box
        return cropped_image, [left, top, right+1, bottom+1]
    
    # Function to find start-end positions of consecutive 0s in a histogram
    def find_zero_ranges(self, histogram, threshold_nonzero=0):
        zero_ranges = []
        start = None
        last_end = 0
        for i, value in enumerate(histogram):
            if value == 0:
                if start is None:
                    start = i
            else:
                if start is not None:
                    end = i - 1
                    if (start - last_end) >= threshold_nonzero:
                        zero_ranges.append((start, end))
                        last_end = end
                    start = None

        # Handle the case where the histogram ends with zeros
        if start is not None:
            zero_ranges.append((start, len(histogram) - 1))
        
        return zero_ranges
    
    # Crop lines
    def crop_lines(self, img_bw, zero_ranges_vh):
        # Select ranges where the gap between consecutive zeros is larger than the minimum threshold
        selected_ranges = [sublist for sublist in zero_ranges_vh if (sublist[1] - sublist[0]) > self.min_zeros_between_lines]
        # Sort the selected ranges by starting position 
        selected_ranges_sorted = sorted(selected_ranges, key=lambda x: x[0])
        # Vector with the images of each line
        imgs_line = []
        # Vector with the position of each line
        pos_line = []

        last_end = 0
        # Get all the images except the last one
        for (start, end) in selected_ranges_sorted:
            imgs_line.append(img_bw[last_end:start, :])
            pos_line.append([last_end, start])
            last_end = end

        # Add the last cropped image
        imgs_line.append(img_bw[last_end:, :])
        pos_line.append([last_end, img_bw.shape[0]])

        return imgs_line, pos_line
    
    # Function to segment characters from a line of text
    def segment_characters_from_line(self, line, zero_ranges_hh):
        # Vector of characters to return
        characters = []
        # Position of the spaces
        spaces = []
        # Position of each character in the image
        characters_pos = []

        counter_chars = 0
        last_end = 0
        # Get all the segmentations (except last one)
        for (start, end) in zero_ranges_hh:
            # Check if the number of zeros is greater than the threshold, to consider it a character
            if (end-start) > self.min_zeros_between_characters:
                characters.append(line[:, last_end:start])
                characters_pos.append([last_end, start])
                last_end = end
                counter_chars += 1
                # If the number of zeros is also greater than min_zeros_space, the next character is a space
                if (end-start) > self.min_zeros_space:
                    spaces.append(counter_chars)
        # Update the last character and its position in the image
        characters.append(line[:, last_end:])
        characters_pos.append([last_end, line.shape[1]])
        return characters, spaces, characters_pos

    # This function captures the image and extracts the image of each character in several lines
    # The number of sublists in the list corresponds to the number of lines
    def segment_characters(self):
        # Capture screen and threshold the image
        screen_img = self.capture_screen()
        # Threshold the image
        img_bw = self.threshold_img(screen_img)
        # Count the number of "1" pixels
        count_ones = np.sum(img_bw == 1)
        # Count the number of "0" pixels
        count_zeros = np.sum(img_bw == 0)

        # If the number of "1" pixels is greater than the number of "0" pixels, invert the image
        if count_ones > count_zeros:
            img_bw = 1 - img_bw
        # Store data for debugging
        self.img_bw = img_bw
        # kernel = np.ones((3,3),np.uint8)
        # img_bw = cv2.erode(img_bw,kernel,iterations = 1)

        # Crop the text part to exactly fit
        img_bw_cropped, box_img = self.crop_binary_image_with_border(img_bw)
        
        # Vertical histogram to extract lines
        vertical_histogram = np.sum(img_bw_cropped, axis=1)
        # Store data for debugging
        self.vertical_histogram = vertical_histogram

        # Obtain the zero ranges of the vertical histogram
        zero_ranges_vh = self.find_zero_ranges(vertical_histogram, self.min_nonzeros_line)
        
        # Crop the lines using the vector of zero ranges
        img_bw_lines, box_lines = self.crop_lines(img_bw_cropped, zero_ranges_vh)

        # Initialize the characters (image)
        characters = []
        # Initialize the coordinates of each character in the image
        characters_boxes = []
        # Initialize the position of the spaces in the image
        spaces = []

        # Loop over the lines
        for i in range(len(img_bw_lines)):

            # Crop the image so that the line exactly fits the image
            img_bw_lines[i], box_line_cropped = self.crop_binary_image_with_border(img_bw_lines[i])

            img_line = img_bw_lines[i]
            # Obtain the horizontal histogram of the current line
            horizontal_histogram = np.sum(img_line, axis=0)
            # Store data for debugging
            if i==0:
                self.horizontal_histogram = horizontal_histogram
            # Obtain the zero ranges
            zero_ranges_hh = self.find_zero_ranges(horizontal_histogram, 0)
            # Obtain the character segmentation of each line
            characters_line, spaces_line, characters_pos_line = self.segment_characters_from_line(img_line, zero_ranges_hh)

            # Loop over the characters to make them fit in the image, and define its bounding box
            characters_box_line = []
            for j in range(len(characters_line)):
                # Fit the character in the image
                characters_line[j], _ = self.crop_binary_image_with_border(characters_line[j])
                # Then get the bounding box
                characters_box_line.append([box_img[0] + characters_pos_line[j][0], 
                                            box_img[1] + box_lines[i][0],
                                            box_img[0] + characters_pos_line[j][1],
                                            box_img[1] + box_lines[i][1]])
                
            # Update the characters, spaces and bounding boxes
            characters.append(characters_line)
            spaces.append(spaces_line)
            characters_boxes.append(characters_box_line)

        return img_bw, characters, spaces, characters_boxes
    
    # This function extracts the character of the image and plots the segmentation
    def segment_characters_and_bbox_image(self):
        # Capture the image
        #img = self.capture_screen()
        # Segment the characters. The only needed variable is the characters_boxes
        img_bw, characters, spaces, characters_boxes = self.segment_characters()
        color_image = cv2.cvtColor((img_bw*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # Flatten the boxes (we don't need them separated by lines)
        flattened_boxes = [item for sublist in characters_boxes for item in sublist]
        # Iterate over the flattened vector to draw a rectangle
        for box in flattened_boxes:
            cv2.rectangle(color_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # Finally, plot the image
        return characters, spaces, characters_boxes, color_image
    
    def ranking_characters(self):
        subfolder_file_counts = []
        total_count = 0
        total_files = 0
        # Walk through the root directory
        for subdir, dirs, files in os.walk(self.folder_chars):
            # Skip the root folder itself
            if subdir == self.folder_chars:
                continue
            # Count the number of files in the subdirectory
            file_count = len(files)
            total_count += file_count
            total_files += 1
            subfolder_file_counts.append((subdir, file_count))

        # Sort the subfolders by the number of files in ascending order
        subfolder_file_counts.sort(key=lambda x: x[1])

        # Print the results
        for subdir, count in subfolder_file_counts:
            print(f"{subdir}: {count} files")
        print("Total characters in dataset: ", total_files)
        print("Total instances in dataset: ", total_count)

    # This function extracts the characters and matches them with a ground truth given by the user, to store the characters
    # in the selected folder
    def save_characters(self, ground_truth):

        # Create the folder of characters if it doesn't exist
        if not os.path.exists(self.folder_chars):
            os.makedirs(self.folder_chars)

        # Get the character
        _, characters, _, _ = self.segment_characters()
        # There is one special character in the game, which is a cursor that appears at the end. We consider that this cursor is present
        # if the user inputed the sentence with a space
        ends_with_cursor = ground_truth[-1] == " "
        # Remove the spaces from the user's sentence, only the cursor is needed if it exists
        ground_truth = ground_truth.replace(" ", "")
        if ends_with_cursor:
            ground_truth += " "

        # Flatten the characters
        flattened_characters = [item for sublist in characters for item in sublist]

        # Ensure that the length of the sentence matches that of the ground truth.
        # The cursor will be detected in the image if it exists, that is why we added the space in the end
        assert(len(ground_truth) == len(flattened_characters))

        # Iterate over all characters
        for i in range(len(ground_truth)):

            # To define the name of the char, we check if it is inside the chars dict
            if ground_truth[i] in self.chars_dict:
                # If the character is inside the char dict, use the alias given by the dictionary
                char_save_name = self.chars_dict[ground_truth[i]]
            else:
                # If not, use the same name as the ground truth
                char_save_name = ground_truth[i]
            # File to save the image character
            path_folder = os.path.join(self.folder_chars, char_save_name)

            # Only save a new character if it doesn't exist
            if not os.path.exists(path_folder):
                os.makedirs(path_folder)
                path_file = os.path.join(path_folder, char_save_name+"_0.png")
            else:
                index = len(os.listdir(path_folder))
                path_file = os.path.join(path_folder, char_save_name+"_" + str(index) + ".png")
            # Remember to multiply by 255 to make it an image
            cv2.imwrite(path_file, flattened_characters[i]*255)


    # This function loads an image of a character and returns the character that most matches the image, from all the chars stored so far    
    def get_char_from_image(self, image):
        image = letterbox(image)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        image = image.astype('float')/255.0
        # # NEW AXIS
        image = torch.from_numpy(image).to(self.device, dtype=torch.float).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image)

            _, predicted = torch.max(output.data, 1)
            output_char = self.labels[predicted.item()]

        if output_char in self.reversed_chars_dict:
            char_result = self.reversed_chars_dict[output_char]
        else:
            char_result = output_char

        return char_result
    
    # This function gets all the chars from an image
    def get_chars_from_image(self):

        # First, segment the images
        _, char_images, spaces, _ = self.segment_characters()
        
        # This is the vector of characters for all the lines
        sentence_lines = []
        # Iterate over the char images extracted
        for line in range(len(char_images)):
            # Vector of characters of one line
            sentence_line = []
            # Extract character by character
            for char_image in char_images[line]:
                sentence_line.append(self.get_char_from_image((char_image*255).astype(np.uint8)))

            # Now, we manage the spaces of that line
            space_line = spaces[line]
            for space_pos in sorted(space_line, reverse=True):
                sentence_line.insert(space_pos, ' ')

            # Finally, we append to the sentence_lines vector the chars of the current line
            sentence_lines.append(''.join(sentence_line))

        # If the text is not empty and the last character was not a hyphen, we add a space to separate with the new text
        if self.text != "" and self.last_hyphen == False:
            self.text += " "

        # Set the last hyphen to false, and only turn true if the last character of the last line is a hyphen
        self.last_hyphen = False
        # Iterate over the lines
        for i in range(len(sentence_lines)):
            # Remove any final spaces (e.g., caused by the cursor)
            sentence_lines[i] = sentence_lines[i].rstrip()
            if sentence_lines[i].endswith('-'):
                # Remove the hyphen and concatenate with the next line without space
                sentence = sentence_lines[i][:-1]
                if i == len(sentence_lines)-1:
                    self.last_hyphen = True
            else:
                # Add the line with a space
                sentence = sentence_lines[i] + " "
            # Add only if the text is not repeated
            if sentence not in self.text:
                self.text += sentence
        # Delete spaces at the end
        self.text = self.text.rstrip()

    # This function resets the text variable
    def reset_text(self):
        self.text = ""
        self.last_hyphen = False

    def plot_region(self):
        self.segment_characters()
        cv2.imshow("image", self.img_bw)
        cv2.waitKey(0)

    def plot_vertical_histogram(self):
        self.segment_characters()
        plt.barh(range(len(self.vertical_histogram)), self.vertical_histogram, color='black')
        plt.show()
        
    def plot_horizontal_histogram(self):
        self.segment_characters()
        plt.bar(range(len(self.horizontal_histogram)), self.horizontal_histogram, color='black')
        plt.show()

    def plot_segmentation(self):
        _, _, _, img_characters_segmented = ocr_object.segment_characters_and_bbox_image()
        cv2.imshow("image", img_characters_segmented)
        cv2.waitKey(0)


if __name__ == "__main__":
    # Create object
    ocr_object = OCR(region, threshold_binary, min_zeros_between_lines, min_nonzeros_line, min_zeros_between_characters, min_zeros_space, folder_chars, chars_dict, size_characters, save_cnn_folder, save_cnn_file)
   
    # These lines are for debug or to help define the parameters
    #ocr_object.plot_region()
    #ocr_object.plot_vertical_histogram()
    #ocr_object.plot_horizontal_histogram()
    #ocr_object.plot_segmentation()
    
    # This line is to save ground truth (the text must correspond the image)
    #ocr_object.save_characters("RAFA gewinnt $448! ")
    ocr_object.ranking_characters()
