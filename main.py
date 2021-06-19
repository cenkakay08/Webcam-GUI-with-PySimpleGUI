# Import the user interface library.
import PySimpleGUI as sg

# Import cv2 to image process.
import cv2

# Global variable for strenght of Blur effect.
blurStrength = 53
# Global variable for drawing line across the face.
isFaceLineDrawActivated = False
# Global variable for the dropdown list value.
gloabalDropDownStringValue = "None"
# Global emoji image variable.
emoji = None

# Haar cascade xml file for face detection.
cascPath = "haarcascade_frontalface_default.xml"
# Inserted to cv2 classifier.
faceCascade = cv2.CascadeClassifier(cascPath)

# Function for set isFaceLineDrawActivated variable value.
def setIsFaceLineDrawActivated(boolean):
    global isFaceLineDrawActivated
    isFaceLineDrawActivated = boolean


# Function for set blurStrength variable value.
def setBlurStrength(val):
    global blurStrength
    blurStrength = val


# Function for set emoji variable value.
def setEffect(string):
    global emoji
    emoji = cv2.imread(string + ".png")


# Main function for detect face and manipulate the image.
def detectFace(cap):
    # We get the frame from frame and validation text.
    ret, frame = cap.read()

    # If return value is normal.
    if ret == True:
        # Get gray image of frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Run the cascade classifier to detect faces.
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # For loop for each face.
        for (x, y, w, h) in faces:
            if gloabalDropDownStringValue != "None":
                if gloabalDropDownStringValue == "Blur":
                    # Apply the GaussianBlur according to face location and dimensions.
                    frame[y : y + h, x : x + w] = cv2.GaussianBlur(
                        frame[y : y + h, x : x + w], (51, 51), blurStrength
                    )
                else:
                    # Resize the emoji image according to face dimensions.
                    emojiResized = cv2.resize(emoji, (w, h))
                    # Define the location of image part that will change. ROI = rectangular region of interest.
                    roi = frame[y : w + y, x : h + x]
                    # Now create a mask of emoji and create its inverse mask also
                    emojiResizedgray = cv2.cvtColor(emojiResized, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(
                        emojiResizedgray, 1, 255, cv2.THRESH_BINARY
                    )
                    mask_inv = cv2.bitwise_not(mask)

                    # Now black-out the area of logo in ROI
                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                    # Take only region of emoji from emoji image.
                    img2_fg = cv2.bitwise_and(emojiResized, emojiResized, mask=mask)

                    # Put emoji in ROI and modify the main image.
                    dst = cv2.add(img1_bg, img2_fg)
                    frame[y : w + y, x : h + x] = dst

            if isFaceLineDrawActivated:
                # Draw a renctangle around face.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Format the frame as png because our GUI only support PNG.
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        # Return frame after all process.
        return imgbytes


# Define the window layout.
layout = [
    # Image as video.
    [sg.Image(key="_IMAGE_")],
    [  # Text
        sg.Text(text="Select the Effect:"),
        # Dropdown list.
        sg.Combo(
            ["Mask", "Sunglasses", "Hugg", "OMG", "Love", "Blur", "None"],
            enable_events=True,
            key="dropdown",
        ),
    ],
    [  # Text
        sg.Text(text="Strength of Blur Effect:"),
        # Slider for blur strenght.
        sg.Slider(
            range=(1, 20),
            default_value=blurStrength,
            size=(20, 15),
            orientation="horizontal",
            enable_events=True,
            font=("Helvetica", 12),
            key="slider",
        ),
    ],
    [  # Checkbox for draw lines around faces.
        sg.Checkbox(
            "Draw lines around the faces.",
            enable_events=True,
            default=False,
            key="drawline",
        )
    ],
]

# create the window and show it without the plot.
window = sg.Window("Webcam GUI", layout, location=(800, 400))

# ---===--- Event LOOP Read and display frames, operate the GUI --- #
# Setup the OpenCV capture device (webcam).
cap = cv2.VideoCapture(0)
while True:
    # Read event and values.
    event, values = window.Read(timeout=1)
    # Quit event for close the GUI.
    if event is None:
        break
    # Catch slider events.
    if event == "slider":
        # Get value from slider.
        val_of_slider = values["slider"]
        # Call the setBlurStrength function.
        setBlurStrength(val_of_slider)
    # Catch dropdown events.
    if event == "dropdown":
        # Get value from dropdownlist.
        gloabalDropDownStringValue = values["dropdown"]
        # Call the gloabalDropDownStringValue function.
        setEffect(gloabalDropDownStringValue)
    # Catch checkbox events.
    if event == "drawline":
        # Get value from drawline checkbox.
        valueOfdrawlineCheckbox = values["drawline"]
        # Call the setIsFaceLineDrawActivated function.
        setIsFaceLineDrawActivated(valueOfdrawlineCheckbox)
    # Get the new frame from function.
    videoBytes = detectFace(cap)
    # Change the Image Element to show the new image.
    window.FindElement("_IMAGE_").Update(data=videoBytes)
