Brain computer interface (BCI) provides a direct communication between the brain 
and external devices by decoding brain signals into commands. The main goal
of this study is to implement a deep learning classification algorithm for the task
two-class motor imagery. The deep learning algorithm is based on learning from
Convolutional Neural Networks (CNN) due to the spatial and temporal resolution
of the trials. Some pre-processing is required before performing the modeling of
the classifier. Since the model is likely to over-fit due to the lack of trials and
high dimensions of each of the trails, some control-complexity parameters are analyzed 
before training the final model. Four models based on CNN with different
architectures are compared: 1D temporal convolutions, 2D temporal and spatial
convolutions, shallow CNN with single temporal and spatial convolution, and deep
CNN with temporal-spatial convolution combined with temporal convolutions. The
best result is obtained with the 1D temporal convolution architecture reaching an
accuracy of 75.23%.
