# Face Recognition Model incorporated with an Anti-Spoofing Layer done on OpenCV and PyTorch.


## 1. Face Recognition

This is a simple face recognition project using Python OpenCV.

### Requirements

- Python 3.6+ (Anaconda is recommended)
- OpenCV 
- Numpy
- Pandas

### How to Use

*First of all, navigate to the directory where the project is.*

### Photo Shooting

1. Run `python take-photos.py`.
2. Enter an ID and a name.
3. Enter a camera ID (the default is 0), if you only have one camera in your PC, just press `ENTER`.
4. Press the `s` key to take photos of your face (it will only work when your face is detected and there's 
enough light in the room).
5. Press the `q` key when you're done.

- Take at least 25 photos of the face of each person.
- Repeat this with at least one different person, otherwise you will get an error.

### Training

1. Run `python train.py`.
2. The program will generate the file "classifiers/lbphClassifier.yml"

- This may take some time depending on how many photos you took.

### Recognizing

1. Run `python recognize.py`


## 2. Face Anti-Spoofing

### Classifier
This folder contains the trained face detector and face recognition model used in step 1.


### Configuration
Contains the configuration file for the entire project.


### Test and Train Sets
Contains the test and trains sets made using the NUAA Photo Imposter Database.


### Dataset
Contains the script to make the custom dataset object.

### Experiments
Contains the scalars and graphs saved during training process by TensorBoard in the 'log' directory.
Contains the trained Anti-Spoofing Model in the 'output' directory. 'densenet_161_rose.pth' was the model used.

### Neural Nets
Contains the neural netowork model architecture script and the loss function script.

### NUAA Photo Imposter Database
The database used for testing and training.

### Training Module
Contains the trainer script used to train the model.
Contains methods for training, saving, evaluating and loading a model.

### Metrics
Contains score, metrics and utils scripts.
'score' script contains methods for score generation and accuracy calculation.
'metrics' script contains methods to keep track of the metrics used by SummaryWriter module of TensorBoard for visualization.
'utils' has general methods like 'get_device', 'get_optimizer', etc.

### Training and Testing
Use the train script to 'train' the model and 'test' script to test the model with your webcam.
Use the 'Acc' script to test the trained model on testsets for accuracy.

### Results
![image](https://user-images.githubusercontent.com/82205701/126898052-5e7b4407-6dea-42c6-bd98-e1d85a571e45.png)

![image](https://user-images.githubusercontent.com/82205701/126898057-c234383a-10d9-43da-8bd0-a9b12d71606c.png)


## Reference
[1] Deep Pixel-wise Binary Supervision for Face Presentation Attack Detection  
[2] https://github.com/leodlca/lbph-face-recognition
