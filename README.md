Download Link: https://assignmentchef.com/product/solved-eee443-assignment-3
<br>
In this question you will implement an autoencoder neural network with a single hidden layer for unsupervised feature extraction from natural images. The following cost function will be minimized:

(1) The first term is the average squared-error between the desired response and the network output across training samples. Note that the desired output is the same as the input. The second term enforces Tykhonov regularization on the connection weights with parameter <em>λ</em>. The last term enforces that the hidden unit activations are sparse with parameter <em>β </em>for controlling the relative weighting of this term. The level of sparsity is tuned via <em>ρ </em>in the <em>KL </em>term (Kullback-Leibler divergence) between a Bernouilli variable with mean <em>ρ </em>and another with mean ˆ<em>ρ<sub>b</sub></em>. ˆ<em>ρ<sub>b </sub></em>is the average activation of hidden unit <em>b </em>across training samples.

<ol>

 <li>The file mat contains a collection of 16×16 RGB patches extracted from various natural images in data. Preprocess the data by first converting the images to graysale using a luminosity model: <em>Y </em>= 0<em>.</em>2126 ∗ <em>R </em>+ 0<em>.</em>7152 ∗ <em>G </em>+ 0<em>.</em>0722 ∗ <em>B</em>. To normalize the data, first remove the mean pixel intensity of each image from itself, and then clip the data range at ±3 standard deviations (measured across all pixels in the data). To prevent saturation of the activation function, map the ±3 std. data range to [0<em>.</em>1 0<em>.</em>9]. Display 200 random sample patches in RGB format, and separately display the normalized versions of the same patches. Comment on your results.</li>

 <li>Prior to training, initialize the weights and the bias terms as uniform random numbers from the interval [−<em>w<sub>o</sub>,w<sub>o</sub></em>], where ) and <em>L<sub>pre,post </sub></em>are the number of neurons on either side of the connection weights. Write a cost function for the network [<em>J,J<sub>grad</sub></em>] = <em>aeCost</em>(<em>W<sub>e</sub>,data,params</em>) that calculates the cost and its partial derivatives. <em>W<sub>e </sub></em>= [<em>W</em><sub>1 </sub><em>W</em><sub>2 </sub><em>b</em><sub>1 </sub><em>b</em><sub>2</sub>], a vector containing the weights for the first and second layers followed by the bias terms; <em>data </em>is of size <em>L<sub>in</sub></em>×<em>N</em>; <em>params </em>is a structure with the following fields <em>Lin </em>(<em>L<sub>in</sub></em>), <em>Lhid </em>(<em>L<sub>hid</sub></em>), <em>lambda </em>(<em>λ</em>), <em>beta </em>(<em>β</em>), <em>rho </em>(<em>ρ</em>). Use <em>J </em>and <em>J<sub>grad </sub></em>as inputs to a gradientdescent solver to minimize the cost. Assuming <em>L<sub>hid </sub></em>= 64, <em>λ </em>= 5 × 10<sup>−4</sup>, experiment with <em>β</em>, <em>ρ </em>to find parameters that work well. Note that performance here is defined based on the ‘quality’ of the features extracted by the network.</li>

 <li>The solver will return the trained network parameters. Display the first layer of connection weights as a separate image for each neuron in the hidden layer. What do the hidden-layer features look like? Are these features representative of natural images?</li>

 <li>Retrain the network for 3 different values (low, medium, high) of <em>L<sub>hid </sub></em>∈ [10 100], of <em>λ </em>∈ [0 10<sup>−3</sup>], while keeping <em>β,ρ </em> Display the hidden-layer features as separate images. Comparatively discuss the results you obtained for different combinations of training parameters.</li>

</ol>

<h1>Question 2.</h1>

The goal of this question is to introduce you CNN models. You will be experimenting with two demos, one on a CNN model in Python, and a second on a CNN model in one of two popular frameworks (PyTorch or TensorFlow). Download demo_cnn.zip from Moodle and unzip it. The demos are given as Jupyter Notebooks along with relevant code and data. The easiest way to install Jupyter with all Python and related dependencies is to install Anaconda. After that you should be able to run through demos in your browser easily. The point of these demos is that they take you through the training algorithms step by step, and you need to inspect the relevant snippets of code for each step to learn about implementation details.

<ol>

 <li>The notebook ipynb contains demonstrations on a CNN model. You need to run the demo till the end without any errors. You are supposed to convert the outputs of the completed demo to a PDF file, and attach it to the project report. You should also comment on your results.</li>

 <li>The notebooks ipynb and TensorFlow.ipynb contain demonstrations on a CNN model in deep learning frameworks. Please pick a single framework to work with (PyTorch has a Python like feeling but might have limited visualization options, and TensorFlow might have a steeper learning curve but is better equipped with supporting tools). You need to run the selected demo till the end without any errors. You are supposed to convert the outputs of the completed demo to a PDF file, and attach it to the project report. You should also comment on your results.</li>

</ol>