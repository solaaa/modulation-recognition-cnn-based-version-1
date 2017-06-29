# research-for-AMC
this project started in 2017/6
adopt python(both of 2.7ver. and 3.5ver.) + Tensorflow + Keras
PLZ use python 2.7ver. to use 'cPackle' first time when loading the dataset
after loading, I choose to 3.6ver. to do my research

after 2015, as the Deep learnig developing rapidly, AMC domain is filled with new power.
rather than using expert features, researchers focus on deep neural network like CNN, 
whose inputs can be the naive features.
hance this project continues this mainstream to do some research on deep learning in AMC

set_up_dataset.py :
  1. load the original dataset
  2. change it into 3 parts: (X: data), (Y: label) and (Z: SNR)
  3. randomly seperate these 3 parts into 2 part as the train-set and test-set
  4. save them
  
extract_digital_signal.py :
  mainly i am studying on digital modulation, so i throw the analog modulation signals
  1. start 2 processes by adopting python API(just for fun:) and trying to learning 
    about the multi-processing coding)
  2. (some processing...)
  3. save them

train_and_test.py :
  the main file i use to study
