# ANLY-590-Project
### Dataset Link: https://drive.google.com/drive/folders/1NXtMM8Zn0P59CNYggf79vOwIq19ilKY6?usp=share_link
### Environment setup
1. Clone and Setup Virtual Env
```
$ git clone https://github.com/prakhar7m/ANLY-590.git
```

```
$  python3 -m venv ANLY-590
```
```
$ ANLY-590/Scripts/activate
```
```
$ pip3 install -r requirements.txt
```

### Testing

#### If u wish to train the model on the dataset use train.py file with the following command in the terminal otherwise run detect.py as the saved model is already present in the directory
1. Train
```
$ python3 train.py --data dataset
```
2. Detection
```
$ python3 detect.py 
```
