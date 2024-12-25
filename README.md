Library Download
This code is written in Python and has several libraries imported. To streamline the download process there is a requirements file created. Simply run the following command on the in-line terminal in your IDE:

>> pip install -r requirements.txt

Testing
To test images, run the model_testing.py file. To add or delete images, simply edit the TestImages folder to include the images you would like to test. 

To navigate through paintings during testing, close the popup window of the current painting manually or press any key to have the same effect, and the next image in the testing folder will appear with its predicted classification. 

To create a new model, run the model_creation.py file and make any changes on the parameters such as epoch value and the file name it will be saved as. 

Dataset
The dataset is too large for submission, however, it can easily be downloaded from Roboflow at: https://universe.roboflow.com/art-styles/styles-tsogn/dataset/2 

However, the dataset is only needed for training new models, so for testing purposes, it is not necessary. 
