# LSTM and RNN Tutorial with Demo (with Stock/Bitcoin Time Series Prediction, Sentiment Analysis, Music Generation)

There are many LSTM tutorials, courses, papers in the internet. This one summarizes all of them. In this tutorial, RNN Cell, RNN Forward and Backward Pass, LSTM Cell, LSTM Forward Pass, Sample LSTM Project: Prediction of Stock Prices Using LSTM network, Sample LSTM Project: Sentiment Analysis, Sample LSTM Project: Music Generation. It will continue to be updated over time.

**Keywords: Deep Learning, LSTM, RNN, Stock/Bitcoin price prediction, Sentiment Analysis, Music Generation, Sample Code, Basic LSTM, Basic RNN**

**NOTE: This tutorial is only for education purpose. It is not academic study/paper. All related references are listed at the end of the file.**

# Table of Contents
- [What is Deep Learning?](#whatisDL)
- [What is RNN?](#whatisRNN)
    - [RNN Cell](#RNNCell)
    - [RNN Forward Pass](#RNNForward)
    - [RNN Backward Pass](#RNNBackward)
    - [RNN Problem](#RNNProblem)
- [What is LSTM?](#whatisLSTM)
    - [LSTM Cell](#LSTMCell)
    - [LSTM Forward Pass](#LSTMForward)
- [SAMPLE LSTM CODE: Prediction of Stock Prices Using LSTM network](#SampleStock)
- [SAMPLE LSTM CODE: Sentiment Analysis](#Sentiment)
    - [Results](#SentimentResults)
    - [DataSet](#SentimentDataSet)
    - [Embeddings](#SentimentEmbeddings)
    - [LSTM Model in Sentiment Analysis](#SentimentLSTM)
- [SAMPLE LSTM CODE: Music Generation](#MusicGeneration)
    - [How to Run Code?](#MusicHowToRunCode)
    - [Input File and Parameters](#MusicInput)
    - [LSTM Model in Music Generation](#MusicLSTM)
    - [Predicting and Sampling](#MusicPredictingAndSampling)
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

Code: https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/BasicRNN


### RNN Cell <a name="RNNCell"></a>

<img width="961" alt="rnn_step_forward" src="https://user-images.githubusercontent.com/10358317/44312581-5a33c700-a403-11e8-968d-a38dd0ab4401.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]

### RNN Forward Pass <a name="RNNForward"></a>

<img width="811" alt="rnn_fw" src="https://user-images.githubusercontent.com/10358317/44312584-6029a800-a403-11e8-9171-38cb22873bbb.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]

### RNN Backward Pass <a name="RNNBackward"></a>

<img width="851" alt="rnn_cell_backprop" src="https://user-images.githubusercontent.com/10358317/44312587-661f8900-a403-11e8-831b-2cd7fae23dfb.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]

### RNN Problem <a name="RNNProblem"></a>
- In theory, RNNs are absolutely capable of handling such “long-term dependencies.” 
- In practice, RNNs don’t seem to be able to learn them. 
- The problem was explored in depth by Hochreiter (1991) [German] and Bengio, et al. (1994) with [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)

## What is LSTM? <a name="whatisLSTM"></a>

- It is a special type of RNN, capable of learning long-term dependencies.

- "Long short-term memory (LSTM) units are units of a recurrent neural network (RNN). An RNN composed of LSTM units is often called an LSTM network. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell"

- Long Short Term Memory (LSTM) is a type of deep learning model that is mostly used for analysis of sequential data (time series data prediction). 

- There are different application areas that are used: Language model, Neural machine translation, Music generation, Time series prediction, Financial prediction, Robot control, Time series prediction, Speech recognition, Rhythm learning, Music composition, Grammar learning, Handwriting recognition, Human action recognition, Sign Language Translation,Time series anomaly detection, Several prediction tasks in the area of business process management, Prediction in medical care pathways, Semantic parsing, Object Co-segmentation.

- LSTM was proposed in 1997 by Sepp Hochreiter and Jürgen Schmidhuber and improved in 2000 by Felix Gers' team.
[Paper](https://www.bioinf.jku.at/publications/older/2604.pdf) 

Code: https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/BasicLSTM

### LSTM Cell <a name="LSTMCell"></a>

<img width="886" alt="lstm_cell" src="https://user-images.githubusercontent.com/10358317/44312843-34a8bc80-a407-11e8-96c3-cc2bc07f1500.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]

### LSTM Forward Pass <a name="LSTMForward"></a>

<img width="860" alt="lstm_fw" src="https://user-images.githubusercontent.com/10358317/44312846-3a060700-a407-11e8-878e-f1ce14cc98b4.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]


## SAMPLE LSTM CODE: Prediction of Stock Prices Using LSTM network <a name="SampleStock"></a>
Stock and ETFs prices are predicted using LSTM network (Keras-Tensorflow).

Code: https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/StockPricesPredictionProject

- Stock prices are downloaded from [finance.yahoo.com](https://finance.yahoo.com/). [Disneyland (DIS) Stock Price CSV file](https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Stock_Prices_Prediction/blob/master/Stock_Prices_Prediction_Example/DIS.csv).
- Closed value (column[5]) is used in the network, [LSTM Code](https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Stock_Prices_Prediction/blob/master/Stock_Prices_Prediction_Example/pricePredictionLSTM.py)
- Values are normalized in range (0,1).
- Datasets are splitted into train and test sets, 50% test data, 50% training data.
- Keras-Tensorflow is used for implementation.
- LSTM network consists of 25 hidden neurons, and 1 output layer (1 dense layer).
- LSTM network features input: 1 layer, output: 1 layer , hidden: 25 neurons, optimizer:adam, dropout:0.1, timestep:240, batchsize:240, epochs:1000 (features can be further optimized).
- Root mean squared errors are calculated.
- Output files:  [lstm_results](https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Stock_Prices_Prediction/blob/master/Stock_Prices_Prediction_Example/lstm_result.csv) (consists of prediction and actual values), plot file (actual and prediction values).

![dis_prediction_and_actualprice](https://user-images.githubusercontent.com/10358317/37895737-e01ed832-30ea-11e8-9249-9b69ae2eccff.png)

## SAMPLE LSTM CODE: Sentiment Analysis <a name="Sentiment"></a>

Sentiment Analysis is an analysis of the sentence, text at the document that gives us the opinion of the sentence/text. In this project, it will be implemented a model which inputs a sentence and finds the most appropriate emoji to be used with this sentence. Code is adapted from Andrew Ng's Course 'Sequential Models'.

**NOTE:This project is adapted from Andrew Ng, [Sequential Models Course](https://github.com/Kulbear/deep-learning-coursera/tree/master/Sequence%20Models), [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) for educational purpose**

Code: https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/SentimentAnalysisProject

### Results <a name="SentimentResults"></a>

![resultsemoji](https://user-images.githubusercontent.com/10358317/43802983-1fe753e4-9aa0-11e8-9b9e-b87fe91e0c18.jpg)

### DataSet <a name="SentimentDataSet"></a>
We have a tiny dataset (X, Y) where:

* X contains 127 sentences (strings)
* Y contains a integer label between 0 and 4 corresponding to an emoji for each sentence

<img width="847" alt="data_set" src="https://user-images.githubusercontent.com/10358317/43802586-eac883e6-9a9e-11e8-8f13-6471cc16a3d8.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]

### Embeddings <a name="SentimentEmbeddings"></a>

Glove 50 dimension, 40000 words of dictionary file is used for word embeddings. It should be downloaded from  https://www.kaggle.com/watts2/glove6b50dtxt (file size = ~168MB))


* word_to_index: dictionary mapping from words to their indices in the vocabulary (400,001 words, with the valid indices ranging from 0 to 400,000)
* index_to_word: dictionary mapping from indices to their corresponding words in the vocabulary
* word_to_vec_map: dictionary mapping words to their GloVe vector representation.

### LSTM Model in Sentiment Analysis <a name="SentimentLSTM"></a>

LSTM structure is used for classification.

<img width="833" alt="emojifier-v2" src="https://user-images.githubusercontent.com/10358317/43802664-22c08c8a-9a9f-11e8-83e1-fea4bf334f6e.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]

Parameters:

![lstm_struct](https://user-images.githubusercontent.com/10358317/43803021-416cc59e-9aa0-11e8-8b28-6045dd0ead87.jpg)



## SAMPLE LSTM CODE: Music Generation  <a name="MusicGeneration"></a>

With trained DL model (LSTM), new sequences of time series data can be predicted. In this project, it will be implemented a model which inputs a sample jazz music and samples/generates a new music. Code is adapted from Andrew Ng's Course 'Sequential models'.

**NOTE:This project is adapted from Andrew Ng, [Sequential Models Course](https://github.com/Kulbear/deep-learning-coursera/tree/master/Sequence%20Models), [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) for educational purpose**

Code: https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/MusicGenerationProject

### How to Run Code? <a name="MusicHowToRunCode"></a>
* To run code, download music21 toolkit from [http://web.mit.edu/music21/](http://web.mit.edu/music21/). "pip install music21". 
* Run main.py


### Input File and Parameters <a name="MusicInput"></a>
Model is trained with "data/original_music"
* "X, Y, n_values, indices_values = load_music_utils()"
* Number of training examples: 60,
* Each of training examples length of sequence:30
* Our music generation system will use 78 unique values. 

* X: This is an (m,  Tx , 78) dimensional array. We have m training examples, each of which is a snippet of  Tx=30Tx=30  musical values. At each time step, the input is one of 78 different possible values, represented as a one-hot vector. Thus for example, X[i,t,:] is a one-hot vector representating the value of the i-th example at time t.
* Y: This is essentially the same as X, but shifted one step to the left (to the past). 
* n_values: The number of unique values in this dataset. This should be 78.
* indices_values: python dictionary mapping from 0-77 to musical values.

### LSTM Model in Music Generation <a name="MusicLSTM"></a>
LSTM model structure is:

<img width="1163" alt="music_generation" src="https://user-images.githubusercontent.com/10358317/44003036-cde60b9a-9e54-11e8-88d4-88d8c9ad0144.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]

Model is implemented with "djmodel(Tx, n_a, n_values)" function.

### Predicting and Sampling: <a name="MusicPredictingAndSampling"></a>

Adding model, predicting and sampling feature, model structure is: 

<img width="1171" alt="music_gen" src="https://user-images.githubusercontent.com/10358317/44003040-d8f6335c-9e54-11e8-9260-ce930c271437.png">

[Andrew Ng, Sequential Models Course, Deep Learning Specialization]

Music Inference Model is similar trained model and it is implemented with "music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100)" function. Music is generated with "redict_and_sample" function.
Finally, your generated music is saved in output/my_music.midi.



## Resources: <a name="Resources"></a>
- [LSTM Original Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- Keras: [https://keras.io/](https://keras.io/)
- Tensorflow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- [LSTM in Detail](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Music Toolkit: http://web.mit.edu/music21/](http://web.mit.edu/music21/)

## References: <a name="References"></a>
- [Andrew Ng, Sequential Models Course, Deep Learning Specialization](https://github.com/Kulbear/deep-learning-coursera/tree/master/Sequence%20Models)
- https://www.kaggle.com/pablocastilla/predict-stock-prices-with-lstm/notebook
- Basic LSTM Code is  adapted from Andrew Ng's Course 'Sequential models'.




