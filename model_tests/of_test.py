import requests
from PIL import Image
from transformers import pipeline
from perceiver.model.vision import optical_flow  # register optical flow pipeline
import cv2
import numpy as np

frame_1 = cv2.imread("../data_t/images/image_2.png")
frame_2 = cv2.imread("../data_t/images/image_3.png")


print("Frame Shape: ", frame_1.shape)
print("Frame Shape: ", frame_2.shape)


frame_1 = cv2.resize(frame_1, (512, 512))
frame_2 = cv2.resize(frame_2, (512, 512))


optical_flow_pipeline = pipeline(
    "optical-flow", model="krasserm/perceiver-io-optical-flow", device="cuda:0"
)
rendered_optical_flow = optical_flow_pipeline((frame_1, frame_2), render=True)

Image.fromarray(rendered_optical_flow).save("optical_flow.png")
