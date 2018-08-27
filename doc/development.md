# Developing Petastorm

Please follow the following instructions to develop Petastorm:

1. ```virtualenv env```
2. ```source env/bin/activate```
3. ```pip install -U pip```
4. For tensorflow without GPU: ```pip install -e .[opencv,tf,test]```. For tensorflow with GPU: ```pip install -e .[opencv, tf_gpu, test]```

To run tests, please run the following: ```pytest```
