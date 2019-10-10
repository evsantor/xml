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

E(w) =n∑i=1K∑k=1[yi,klogσi,k+ (1−yi,k)log(1−σi,k)

