# Extreme Multilabel Classification with Deep Learning

Extreme multi-label learning (XML) or classification has been a practical and im-
portant problem since the boom of big data. It’s object is to learn a classifier to
automatically tag a datapoint with the most relevant subset of labels from an ex-
tremely large label set. In this report I am going to explain the procedure I followed
in order to build a model that can respond respectively well with the one in XML
Repository using the same data for training, Mediamill, Bibtex, Delicious, RCV
and Eurlex.
First let’s set the problem. For input{Xi,Yi}ni=1, whereXis a matrix ofN M-
dimensional data points andY is the matrix ofN K-sized binary vectors indicating
the labels assigned to eachxidata point, we consider a supervised learning problem of
Multilabel Classification, consisting of a Multilayer Perceptron and the minimazation
of the binary cross entropy loss function:

## Load Data
To load the data from the XML Repository was quite a difficult task. First I had
to use matlab to load the data. For the small datasets I loaded them and converted
them to csv for easier use. For the larger datasets this was not possible so I used
sparse matrices, more specifically CSR.
Also the small ones, had seperate files for train and test data that contained
indices of the points that are in the test and in the training set for different splits
but in the large datasets, only one file was given for train and test that was the only
split.
Another problem I faced with the data was that some of them did not have labels
at all. In those cases I added one more class that represented this category so when
I trained the model I noticed a small improvement.

## Model
In this work propose a simple deep learning procedure to tackle this problem. Here’s
what the model consists of:

* Model: A Multilayer Perceptron that consist 2 or 4 hidden layers with ReLU
activation functions. Tanh is also a good choice, but slower.
* Output Activation Function: Sigmoid instead of softmax. Softmax is useful
in multiclass classification problems where you want the outputs to be proba-
bilities that sum to 1, having a distribution over the class predictions. In our
case we need probabilities for each class separately as we have multiple labels
for each data point thus sigmoid is much more suited.
* Optimizer: For optimizer I used Adam [?] as it incorporates momentum,
RMS and Bias Correction leading to faster convergence. Another good choice
is NAdam[4] and the newly introduced RAdam[5].
* Regularization and Normalization: As regularizers we used Dropout in
every hidden layer, Early Stopping and also batch normalization to adjust and
scale the activations, make learning faster and introduce a bit of regularization
through noise.

### Tools  
In order to create my model I used keras with tensroflow 2.0 backend with Eager Execution enabled

### Fit
To fit the model I have used fit generator with some callbacks. Early Stopping to stop the training when validation loss stops improving and ReduceLRonPlateau to reduce learning rate in the same case.

### Hyperparameter Tuning
For hyperparameter tuning I used \href{https://github.com/autonomio/talos}{talos library}, but I tested it only on bibtex due to each small size. The parameters I tried were the number of layers, activation function, tanh or relu and the dropout, whether it would be 0.25 or 0.5 in each different layer. The way is works is like grid search, so there's no clever inbuilt procedure of sampling good sets the more you try, thus it can be slow when testing bad parameters. It's very easy to use though.

## Metrics
For metrics I converted the matlab code that I found in \href{http://manikvarma.org/downloads/XC/XMLRepository.html}{XML Repository} into python and I added some additional methods. $\hat{\mathbf{y}} \in \mathcal{R}^{L}$ stands for predicted score vector, $\mathbf{y} \in\{0,1\}^{L}$ for truth label vector and $\operatorname{rank}_{k}(\mathbf{y})$ returns the $k$ largest indices of $y$ ranked in descending order. $p_l$ is the propensity score for label $l$ which helps in making metrics unbiased.
    * Mean Average Precision @ k: $$\mathrm{P} @ k :=\frac{1}{k} \sum_{l \in \operatorname{rank}_{k} (\hat{\mathbf{y}})} \mathbf{y}_{l}$$
    Computes the average precision at k between two lists of items. I created 4 such methods. \textbf{Apk} is the one that is implemented in the XML Repository and it simply counts how many labels from those there were predicted are correct and divides the score with $k$. In \textbf{Soft Apk}, in case $k > $\#labels, it divides with \#labels instead of $k$. \textbf{Strict-Soft Apk} counts the correct labels and divides them with $(i + 1)$ (i stands for the iteration of the predicted labels). And last, \textbf{Strict Apk} for every correct label divides with $(i + 1)* weights[i]$. I tried all of them out of interest as they express different aspects of the precision metric in a setting like that.
    
    * Propensity-weighted Mean Average Precision @ k: 
    $$\mathrm{PSP} @ k :=\frac{1}{k} \sum_{l \in \operatorname{rank}_{k} (\hat{\mathbf{y}})} \frac{\mathbf{y}_{l}}{p_{l}}$$
    For all the apks, it divides with a weight $p_{l}$ for each correct label that was found. To calculate the weight I used the inv\_propensity from the XML Repository.
    
    * nDCG @ k: $$\mathrm{DCG} @ k :=\sum_{l \in \mathrm{rank}_{k} (\hat{\mathbf{y}})} \frac{\mathbf{y}_{l}}{\log (l+1)}$$
    For every correct label, it adds to the score $\frac{1}{\log(l + 1)}$. If it was propensity weighted it multiplies the denominator with the weight of the label.
    
    * Recall @ k:
    $$\mathrm{Recall} @ k :=\frac{1}{len(\hat{\mathbf{y}})} \sum_{l \in\operatorname{rank}_{k}(\hat{\mathbf{y}})} \mathbf{y}_{l}$$
    The sum of the correct labels devided with the score with the length of the true labels that is equal with the true positive and the false negative.
    
    * Coverage @ k:
    $$\mathrm{Coverage} @ k :=\frac{\sum_{l \in \operatorname{rank}_{k} (\hat{\mathbf{y}})} \mathbf{y}_{l}}{\sum{y}} $$
    Sums the correct labels it predicted and divides them with the total correct labels.
    
    * f1score:
     $$ F_{1}=2 \cdot \frac{\text { precision } \cdot \text { recall }}
     { \text {precision} + \text {recall}} $$
     
     The harmonic mean of the precision and the recall.
    
