# Doodly

<h3>Can a neural network identify what you just doodled?</h3>

Draw any doodle from a hat, tree, an apple or an alarm clock.
The neural network would identify which one of them did you draw.

<h3>Architecture</h3>

I have used Convolutional Neural Network(CNN) for classifying the doodle. CNN's work well for images instead of vanilla neural network(fully connected). This is because of CNN's ability to take into account the locality of pixels.
I have created this architecture in keras and the code for this is available in the train.py file.

<h3>Frontend</h3>

I have encorporated this CNN into an Android app. I trained the CNN separately and then freezed the tensorflow model and imported it in Android for inference.

![alt text](/Screenshot_2018-03-14-21-50-54-707_com.ninhydrin.doodly.png )

![alt text](/Screenshot_2018-03-14-21-50-36-396_com.ninhydrin.doodly.png )
