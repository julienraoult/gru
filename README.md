# RNN

https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%203%20-%20Lesson%201c.ipynb

Resultats :

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, None, 64)          523840
_________________________________________________________________
conv1d (Conv1D)              (None, None, 128)         41088
_________________________________________________________________
global_average_pooling1d (Gl (None, 128)               0
_________________________________________________________________
dense (Dense)                (None, 64)                8256
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 65
=================================================================
Total params: 573,249
Trainable params: 573,249
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
2021-06-06 23:09:48.969485: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-06-06 23:09:48.972430: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2592005000 Hz
2021-06-06 23:09:49.059687: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-06-06 23:09:50.652205: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-06-06 23:09:50.737016: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
391/391 [==============================] - 60s 141ms/step - loss: 0.5691 - accuracy: 0.6689 - val_loss: 0.3207 - val_accuracy: 0.8672
Epoch 2/10
391/391 [==============================] - 27s 69ms/step - loss: 0.2272 - accuracy: 0.9151 - val_loss: 0.2928 - val_accuracy: 0.8796
Epoch 3/10
391/391 [==============================] - 23s 59ms/step - loss: 0.1568 - accuracy: 0.9438 - val_loss: 0.3347 - val_accuracy: 0.8726
Epoch 4/10
391/391 [==============================] - 22s 55ms/step - loss: 0.1287 - accuracy: 0.9562 - val_loss: 0.4375 - val_accuracy: 0.8518
Epoch 5/10
391/391 [==============================] - 22s 55ms/step - loss: 0.1050 - accuracy: 0.9656 - val_loss: 0.4126 - val_accuracy: 0.8642
Epoch 6/10
391/391 [==============================] - 21s 53ms/step - loss: 0.0761 - accuracy: 0.9769 - val_loss: 0.5342 - val_accuracy: 0.8552
Epoch 7/10
391/391 [==============================] - 21s 53ms/step - loss: 0.0632 - accuracy: 0.9798 - val_loss: 0.5998 - val_accuracy: 0.8576
Epoch 8/10
391/391 [==============================] - 20s 51ms/step - loss: 0.0461 - accuracy: 0.9894 - val_loss: 0.6264 - val_accuracy: 0.8537
Epoch 9/10
391/391 [==============================] - 20s 51ms/step - loss: 0.0438 - accuracy: 0.9860 - val_loss: 0.7012 - val_accuracy: 0.8514
Epoch 10/10
391/391 [==============================] - 22s 55ms/step - loss: 0.0322 - accuracy: 0.9910 - val_loss: 0.8357 - val_accuracy: 0.8514
```

During training, the training loss keeps decreasing and training accuracy keeps increasing slowly. But the validation loss started increasing while the validation accuracy is not improved => https://github.com/keras-team/keras/issues/3755

It is because the model is overfitting on the training data, thus becoming extremely good at classifying the training data but generalizing poorly and causing the classification of the validation data to become worse. You could solve this by stopping when the validation error starts increasing or maybe inducing noise in the training data to prevent the model from overfitting when training for a longer time.

The model is overfitting the training data.
To solve this problem you can try
1.Regularization (L2)
2.Try to add more add to the dataset or try data augumentation
3. Dropout

# GRU

Fix GRU layers instead of Convolution layer (PR from anisayari) => https://github.com/lmoroney/dlaicourse/pull/184/files

https://www.kaggle.com/geektoday/generate-next-scene-of-jungle-book
