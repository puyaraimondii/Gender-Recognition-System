Gender Recognition
==================

To run the program, the following python packages are required:

1.opencv (version >=3)
2.tensorflow





Run the project from the root dir as below:

Step1: Add model download from our google drive into model_CNN folder
This is a trained model which can be used for gender recognition
download link:
https://drive.google.com/drive/folders/1oqU0FFJ-bv_VI3s10-FC5cea5sggFfDC?usp=sharing

Step1:*python step1_webcam.py*

Face on the image will be detected 
and image with detected face will be saved in the root path in order to be classified

Step2:*python step2_main.py* 

We will use trained model to differentiate the gender of model
The result of gender classification will be shown on the console for the image saved on root path
