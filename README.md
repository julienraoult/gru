# RNN

https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201c.ipynb

During training, the training loss keeps decreasing and training accuracy keeps increasing slowly. But the validation loss started increasing while the validation accuracy is not improved => https://github.com/keras-team/keras/issues/3755

It is because the model is overfitting on the training data, thus becoming extremely good at classifying the training data but generalizing poorly and causing the classification of the validation data to become worse. You could solve this by stopping when the validation error starts increasing or maybe inducing noise in the training data to prevent the model from overfitting when training for a longer time.

The model is overfitting the training data.
To solve this problem you can try
1.Regularization (L2)
2.Try to add more add to the dataset or try data augumentation
3. Dropout

# GRU

Fix GRU layers instead of Convolution layer => https://github.com/lmoroney/dlaicourse/pull/184/files

https://www.kaggle.com/geektoday/generate-next-scene-of-jungle-book
