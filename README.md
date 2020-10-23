# Image_classifier
A Flask project to classify images using CNN. The project would be able to classify images such as: Windmills, Pumps, PCB, Damaged and Undamaged mobile screens or any other class depending on training images

## Files 
### Image_classifier_training
This is a jupyter notebook(Google Colab) used for training a CNN model. The model architecture and summary is defined in the notebook. It will work for any set of training images
### Flask app
The rest of the files constitute the flask app, which has a UI wherein users can upload their images and predict its results.

## Model

The model uses 4 layers on Convnet to identify distinguishing features within the training images. It flattens the images and implements a dropout layer to treat Overfitting of the model before passing it to a dense layer

      Layer (type)                 Output Shape              Param #   
      =================================================================
      conv2d (Conv2D)              (None, 348, 348, 64)      1792      
      _________________________________________________________________
      max_pooling2d (MaxPooling2D) (None, 174, 174, 64)      0         
      _________________________________________________________________
      conv2d_1 (Conv2D)            (None, 172, 172, 64)      36928     
      _________________________________________________________________
      max_pooling2d_1 (MaxPooling2 (None, 86, 86, 64)        0         
      _________________________________________________________________
      conv2d_2 (Conv2D)            (None, 84, 84, 128)       73856     
      _________________________________________________________________
      max_pooling2d_2 (MaxPooling2 (None, 42, 42, 128)       0         
      _________________________________________________________________
      conv2d_3 (Conv2D)            (None, 40, 40, 128)       147584    
      _________________________________________________________________
      max_pooling2d_3 (MaxPooling2 (None, 20, 20, 128)       0         
      _________________________________________________________________
      flatten (Flatten)            (None, 51200)             0         
      _________________________________________________________________
      dropout (Dropout)            (None, 51200)             0         
      _________________________________________________________________
      dense (Dense)                (None, 512)               26214912  
      _________________________________________________________________
      dense_1 (Dense)              (None, 1)                 513       
      =================================================================
      Total params: 26,475,585
      Trainable params: 26,475,585
      Non-trainable params: 0
      
     
