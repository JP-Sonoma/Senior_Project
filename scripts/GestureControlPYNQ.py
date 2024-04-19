#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install cflib')


# In[2]:


#get_ipython().system('pip install pynput')


# In[1]:


import tempfile
import os
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.positioning.motion_commander import MotionCommander
from pynq.overlays.base import BaseOverlay
from pynq.lib import Button
import logging
import time
import cv2
import numpy as np
from pynq.lib.video import VideoMode
from pynq.lib.video import PIXEL_RGB


# Set a writable directory for the cache
cache_dir = tempfile.mkdtemp()
os.environ["CF2_CACHE_DIR"] = cache_dir

# Initialize the Crazyradio USB dongle
cflib.crtp.init_drivers()

# Create a Crazyflie object for communication
drone = Crazyflie()

# Connect to the Crazyflie drone
drone.open_link("radio://0/80/2M")

# Wait for the Crazyflie to be connected
while not drone.is_connected:
    print("Waiting for the Crazyflie to be connected")
    time.sleep(0.1)

if drone.is_connected:
    print("Crazyflie connected successfully!")
else:
    print("Failed to connect to the Crazyflie.")






# In[2]:


class ButtonDroneController:
    def __init__(self, cf, mc, buttons):
        self.cf = cf
        self.mc = mc
        self.buttons = buttons

        self.velocity = 0.5  # Adjust the velocity value as needed
        self.fixed_height = 0.3  # Set the desired height in meters
        self.taken_off = False

        self.debounce_time = 0.2  # Debounce time in seconds
        self.last_button_press_time = [0] * len(buttons)  # Initialize last button press time for each button

        print('Press button 0 to take off!')
        print('Press button 1 to land!')
        print('Press button 2 for emergency stop!')

    def control_drone(self, button):
        try:
            current_time = time.time()
            if current_time - self.last_button_press_time[button] < self.debounce_time:
                # Button press occurred within debounce time, ignore
                return

            self.last_button_press_time[button] = current_time

            if button == 0:  # Button 0 for takeoff
                if not self.taken_off:
                    button_press_time = time.time()  # Record button press time
                    self.mc.take_off(self.fixed_height)
                    #time.sleep(1.0)  # Add a delay after take-off for stabilization
                    self.taken_off = True
                    command_execution_time = time.time()  # Record command execution time
                    delay_time = command_execution_time - button_press_time
                    print(f'Delay time: {delay_time:.2f} seconds')

                else:
                    self.mc.start_linear_motion(0, 0, 0, self.fixed_height)

            elif button == 1:  # Button 1 for landing
                print('Landing...')
                button_press_time = time.time()  # Record button press time
                self.mc.land()
                self.taken_off = False  # Reset takeoff flag after landing
                command_execution_time = time.time()  # Record command execution time
                delay_time = command_execution_time - button_press_time
                print(f'Delay time: {delay_time:.2f} seconds')

            elif button == 2:  # Button 2 for emergency stop
                print('Emergency stop!')
                button_press_time = time.time()  # Record button press time
                self.mc.stop()  # Stop the drone's motion
                command_execution_time = time.time()  # Record command execution time
                delay_time = command_execution_time - button_press_time
                print(f'Delay time: {delay_time:.2f} seconds')
                self.mc.close_link()  # Close the connection to the drone

            # Add more button mappings for other controls
        
        except Exception as e:
            print(f"Exception in control_drone: {e}")


# In[ ]:


if __name__ == '__main__':
    base = BaseOverlay('base.bit')
    buttons = [Button(base.buttons[i]) for i in range(3)]

    # Initialize the MotionCommander with the Crazyflie object
    mc = MotionCommander(drone)

    controller = ButtonDroneController(drone, mc, buttons)

    # Gesture detection setup
    net = cv2.dnn.readNet('yolov3-tiny-custom_best (4).weights', 'yolov3-tiny-custom.cfg')
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # monitor configuration: 640*480 @ 60Hz
    Mode = VideoMode(640,480,24)
    hdmi_out = base.video.hdmi_out
    hdmi_out.configure(Mode, PIXEL_RGB)
    hdmi_out.start()
    
    # monitor (output) frame buffer size
    frame_out_w = 640
    frame_out_h = 480
    # camera (input) configuration
    frame_in_w = 640
    frame_in_h = 480
    
    os.environ["OPENCV_LOG_LEVEL"]="SILENT"
    # initialize camera from OpenCV
    videoIn = cv2.VideoCapture(0)
    #videoIn = cv2.cvtColor(cv2.COLOR_BGR2RGB)
    videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
    videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);


    print("Capture device is open: " + str(videoIn.isOpened()))

    font = cv2.FONT_HERSHEY_PLAIN
    frame_id = 1
    
    try:
        while True:

            # Gesture detection
            ret, frame = videoIn.read()
            if ret:
                height, width, channels = frame.shape
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (64, 64), (0, 0, 0), True, crop=False)

                net.setInput(blob)
                outs = net.forward(output_layers)

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.20:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            label = str(classes[class_id])
                            # Execute corresponding drone control based on gesture label
                            if label == "Up":
                                controller.control_drone(0)  # Trigger takeoff button
                            elif label == "Down":
                                controller.control_drone(1)  # Trigger landing button
                            elif label == "Land":
                                controller.control_drone(2)  # Trigger emergency stop button

            # Button-based control
            for i, button in enumerate(buttons):
                if button.read():  # Check if button is pressed
                    controller.control_drone(i)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Exit.")

    # Release resources
    videoIn.release()
    cv2.destroyAllWindows()


# In[ ]:




