# hand_gesture_project
Quarter Project for Machine Learning Theory - EECS 298 at UCI

*Given the context of machine learning theory, this project does not make use of prepackaged libraries from tensorflow / keras / sci-kit learn / pytorch

Andrew Muzik
EECS 298 – ML Theory, Dr. Sanger
UCI, Fall 2020

Machine Learning for Hand Gesture Identification

Abstract- Using the electrical signals generated from muscle activity, patient trials are conducted to classify a number of
hand gestures while recording the signal measurements. This project explores a cluster approach to classify each hand
gesture, first using a standard k-means model from the Sklearn library, followed by a custom k-means model that takes
advantage of the supervised nature of the trials. Dimensionality reduction using PCA is also explored, showing a notable
improvement in model accuracy through clustering. The standard k-means model was found to have 23% while the
custom k-means model scored 27%. Future considerations involve additional data cleansing, improving cluster
initialization and looking into a hybrid model approach. 

The MYO armband is a wireless device worn above the elbow that is used to identify and act on different hand gestures.
Using a Bluetooth connection, the device is capable of detecting up to 6 gestures: 1 – at rest 2 – fist, 3 – wrist flexion
(palm in), 4 – wrist extension (palm out), 5 – radial deviation (tilt left) and 6 – ulnar deviation (tilt right). Each of these
gestures is pictured below in Figure 1.


![gestures](https://user-images.githubusercontent.com/22182524/109402992-49873480-790f-11eb-8bbf-811daf5535b1.JPG)

Figure 1: MYO Gesture Classes [1].

The MYO armband uses a sample rate of 200Hz, with signals measured on the scale of µV, as shown in the first output of
Appendix 1, using 8 channels, time and expected output. Many different models – both supervised and unsupervised
have been tested using the MYO armband data. For example, one paper uses a hybrid approach – using clustering along
with a dynamic time-warping algorithm to utilize the time-dependent nature of the data [1]. Another approach taken by
Zhang et al. uses a sliding window of the data fed into an artificial neural net (ANN) resulting in high accuracy and
response time, but uses an offline model as opposed to many clustering techniques [1, 2]. In general, the model can be
broken down into the stages; preprocessing, training, prediction, testing. The approach taken in this project is to analyze
the clustering component in more detail, using a random initialization followed by a semi-supervised approach taken by
initializing the clusters from the expected outputs. 

Approach:
Preprocessing. A total of 36 patients each performed 2 trials of each hand gesture, making a total of over 4 million
samples collected. The data was split into 1 trial for training and 1 for testing. To reduced computational complexity,
only the first 8 patients were used, leaving 551,135 samples total as shown in the program output of cell #72 in
Appendix A. The values shown fall around 100µV, leaving a high quantization error used itself in the clustering algorithm
with Euclidean distance between points. To overcome the quantization error, the input data is normalized before further
reduction, as shown in the program output of cell #73 in Appendix A. The class 0 is used to represent unmarked data
during the trial, which represents the initial and final phases of trials where patients could be stretching, moving and so
forth. These instances are trimmed from the dataset, leaving 184,911 samples left in the data with expected outputs
from {1, 2, 3, 4, 5, 6}, as shown in the program output of cell #74 in the Appendix.

With the unlabeled data discarded and normalized, the dimensionality of the data is then analyzed through principal
components analysis (PCA). Calculating the variances in channels, we can see three principal components from the data,
as shown in the output of cell #7 of Appendix A. The reduced dataset calculated from PCA is then tested against the
normalized, trimmed dataset using a quick, prepackaged Kmeans model from Sklearn, with the results showing an
improvement through PCA.

Training. The training of the K means model is shown in the output of cell #67 in Appendix A, where the max number of
iterations is set to 3 and the number of fittings is set to 2 with a tolerance of 0.0005 in order to constrain the program
output and runtime. The training stage is not optimized, for example using lists to update the cluster means instead of
data frames. From a theoretical standpoint, the runtime of the training stage takes grows exponentially as the number
of iterations increases alongside the number of fittings, yielding psuedopolynomial time. However, by using a custom
model we can see the convergence happen alongside the cluster updates. The clusters are then plotted on a 3d graph
for visualization, but as expected the clusters are very close, layering over top each other, with very unclear decision
boundaries as shown in the Appendix.

Prediction. The prediction of the model is done by outputting the assigned cluster, using the expected output as cluster
initialization the model outputs a direct result to the cluster prediction. Comparing the predicted results to the actual
results taken from the trial the accuracy yields 27%, a slight improvement from the 24% accuracy of the packaged
model. If the optimal number of clusters is to be explored, then cluster symbols can be used for visualization.

Testing. By increasing the number of iterations during training, the model’s accuracy drops, indicating that the model is
overfitting – and pushing the clusters closer toward each other. This is also indicated by the cluster updates shown in cell
#67, occasionally showing samples on the boundary between clusters, leading to a lack of convergence. On occasion the
model does diverge to a solution, the accuracy is wildly unpredictable due to the random initializations of the clusters. In
order to take advantage of the time series nature of the data, a sliding window could be used – leading to the question
of how many samples to use, with one study looking at 200 samples / window to ensure that enough time is given for
each gesture (1 second with sample rate at 200Hz) [1]. To ensure that windows aren’t overlapping between patients, it
would be bests to capture each window while reading in patient trials.

Findings: The k-means model showed a notable improvement using the expected classes as initializations. Looking at the
expected outputs, we can see that both the packaged and custom models were able to extrapolate the 1st class, when
the hand is at rest. Although accuracy was fairly low, the improvement shows the strength of the clustering algorithm
exploiting correlations in the data. The improvement in accuracy also shows the susceptibility of the k-means algorithm
to noisy data, showing that the data could be cleaned up a bit more, one study reports improvement in accuracy after
rectifying the input data [2].

Future Work: Possible future directions include further refining the initialization of the clusters by looking at k-means++
algorithm, used to maximize the distance between cluster initializations. Another direction is to analyze the optimal
number of clusters, possibly using 2*n clusters, where n = number of classes, to reduce the sum square error of the
model. Noticing that a simple clustering scheme is able to correctly classify the first class (hand at rest), another future
direction involves looking into a hybrid approach, using an ANN to further classify between classes in motion along with
exploiting correlations between the time series nature of the data. Also note, in lieu of time the unsupervised k-means
class option was untested after refinements to the cluster initialization.

Concluding Remarks:
In conclusion, PCA is shown as a powerful tool for both data compression and visualization, while showing some
of the strengths and drawbacks of the k-means clustering. Although the k-means algorithm can’t be proven to converge
to a global optimum, it can be used to find identify useful correlations in an unsupervised manner. Utilizing the expected
class output, the clustering accuracy can be improved by initializing with respect to the class output. Future work in
clustering analysis may involve improving the initialization of the clusters by the k-means++ algorithm, maximizing
distance between the clusters. 



Works Cited:

[1]. Benalcazar, M. E., Jaramillo, A. G., Jonathan, Zea, A., Paez, A., & Andaluz, V. H. (2017). Hand gesture
recognition using machine learning and the Myo armband. 2017 25th European Signal Processing
Conference (EUSIPCO). doi:10.23919/eusipco.2017.8081366
[2]. Zhang, Z., Yang, K., Qian, J., & Zhang, L. (2019). Real-Time Surface EMG Pattern Recognition for Hand
Gestures Based on an Artificial Neural Network. Sensors, 19(14), 3170. doi:10.3390/s19143170

