# classifier-for-plants
A plant based classifier classifying wether its healthy or diseased

 --- --- -- - -- - - -- - - -- - 
## Method -1 (The Binary classifier 0:Healthy, 1:Diseased)
In this basically i was making a classifier, a multimodal methodds based classifier, the metadata was going through one neural network path, that was of feedforward type, the another images were going through ResNet-34 architecture, after that these two were being late fused, and then an output was being predicted, the model used ReLU for activation functions in FNN, sigmoid and softmax were used in the CNN part, K-fold stratified Validation was used. 
2 classes were taken of Peach and then binary classification was used on them, 1000 samples were there, data leakage was recorded the data set used was (https://www.kaggle.com/datasets/shubhamdivakar/multimodal-plant-disease-dataset-by-subham-divakar), while trainig on GPU Google colab, T4 GPU was used, 600 for training 200 for validation, , 4 folds were made, 5 epochs each, batch size of 16 was taken, 200 were for testing.

-- -- - -- - - - --- -- --- -- -- --
## Method - 2 (The Multiple classifier, )
In this one fruit is taken say i am taking tomato and it has multiple classes, classfication would done on the type of diseases that are there, the same pathway would be used. 