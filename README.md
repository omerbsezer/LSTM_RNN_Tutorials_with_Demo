# LSTM and RNN Tutorial with Stock/Bitcoin Time Series Prediction Code Example

There are many LSTM tutorials, courses, papers in the internet. This one summarizes all of them. In this tutorial, there are different section: Introduction to Deep Learning, Recurrent Neural Networks (RNN), Long Short Term Memory (LSTM), Stock Price Prediction using LSTM.  

**Keywords: Deep Learning, LSTM, RNN, Stock/Bitcoin price prediction, Sample Code, Basic LSTM, Basic RNN**

**NOTE: This tutorial is only for education purpose. It is not academic study/paper. All related references are listed at the end of the file.**

# Table of Contents
- [What is Deep Learning?](#whatisDL)
- [What is RNN?](#whatisRNN)
    - [RNN Cell](#RNNCell)
    - [RNN Forward Pass](#RNNForward)
    - [RNN Backward Pass](#RNNBackward)
- [What is LSTM?](#whatisLSTM)
    - [LSTM Cell](#LSTMCell)
    - [LSTM Forward Pass](#LSTMForward)
- [SAMPLE LSTM CODE: Prediction of Stock Prices Using LSTM network](#Sample)
- [Resources](#Resources)
- [References](#References)
  
## What is Deep Learning (DL)? <a name="whatisDL"></a>

"Deep Learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks." There are different types of DL models: Convolutional Neural Network, Recurrent Neural Networks (RNN), Long Short Term Memory (LSTM), Restricted Boltzmann Machine (RBM), Deep Belief Networks, etc.

In this tutorial, we are focusing on recurrent networks, especially LSTM. Basic RNN structure, Basic LSTM structures and Stock/Bitcoin Price Prediction Sample code are presented in the following sections. 


## What is RNN? <a name="whatisRNN"></a>

* Recurrent neural network (RNN) is a type of deep learning model that is mostly used for analysis of sequential data (time series data prediction). 
* There are different application areas that are used: Language model, neural machine translation, music generation, time series prediction, financial prediction, etc. 
* The aim of this implementation is to help to learn structure of basic RNN (RNN cell forward, RNN cell backward, etc..).
* Code is adapted from Andrew Ng's Course 'Sequential models'.

### RNN Cell <a name="RNNCell"></a>

<img width="961" alt="rnn_step_forward" src="https://user-images.githubusercontent.com/10358317/44312581-5a33c700-a403-11e8-968d-a38dd0ab4401.png">

### RNN Forward Pass <a name="RNNForward"></a>

<img width="811" alt="rnn_fw" src="https://user-images.githubusercontent.com/10358317/44312584-6029a800-a403-11e8-9171-38cb22873bbb.png">

### RNN Backward Pass <a name="RNNBackward"></a>

<img width="851" alt="rnn_cell_backprop" src="https://user-images.githubusercontent.com/10358317/44312587-661f8900-a403-11e8-831b-2cd7fae23dfb.png">


## What is LSTM? <a name="whatisLSTM"></a>

"Long short-term memory (LSTM) units are units of a recurrent neural network (RNN). An RNN composed of LSTM units is often called an LSTM network. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell"

Long Short Term Memory (LSTM) is a type of deep learning model that is mostly used for analysis of sequential data (time series data prediction). There are different application areas that are used: Language model, neural machine translation, music generation, time series prediction, financial prediction, etc. 

LSTM was proposed in 1997 by Sepp Hochreiter and Jürgen Schmidhuber and improved in 2000 by Felix Gers' team.
[Paper](https://www.bioinf.jku.at/publications/older/2604.pdf) 


### LSTM Cell <a name="LSTMCell"></a>

<img width="886" alt="lstm_cell" src="https://user-images.githubusercontent.com/10358317/44312843-34a8bc80-a407-11e8-96c3-cc2bc07f1500.png">

### LSTM Forward Pass <a name="LSTMForward"></a>

<img width="860" alt="lstm_fw" src="https://user-images.githubusercontent.com/10358317/44312846-3a060700-a407-11e8-878e-f1ce14cc98b4.png">


## SAMPLE LSTM CODE: Prediction of Stock Prices Using LSTM network <a name="Sample"></a>
Stock and ETFs prices are predicted using LSTM network (Keras-Tensorflow).

- Stock prices are downloaded from finance.yahoo.com.
- Closed value (column[5]) is used in the network.
- Values are normalized in range (0,1).
- Datasets are splitted into train and test sets, 50% test data, 50% training data.
- Keras-Tensorflow is used for implementation.
- LSTM network consists of 25 hidden neurons, and 1 output layer (1 dense layer).
- LSTM network features input: 1 layer, output: 1 layer , hidden: 25 neurons, optimizer:adam, dropout:0.1, timestep:240, batchsize:240, epochs:1000 (features can be further optimized).
- Root mean squared errors are calculated.
- Output files:  lstm_results (consists of prediction and actual values), plot file (actual and prediction values).


![dis_prediction_and_actualprice](https://user-images.githubusercontent.com/10358317/37895737-e01ed832-30ea-11e8-9249-9b69ae2eccff.png)

## Resources: <a name="Resources"></a>
- [LSTM Original Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- Keras: [https://keras.io/](https://keras.io/)
- Tensorflow: [https://www.tensorflow.org/](https://www.tensorflow.org/)

## References: <a name="References"></a>
- Andrew Ng, Sequential Models Course, Deep Learning Specialization
- https://www.kaggle.com/pablocastilla/predict-stock-prices-with-lstm/notebook
- Basic LSTM Code is  adapted from Andrew Ng's Course 'Sequential models'.



