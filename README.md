# About
The main goal of this project was to build a Neural Network without any deep learning frameworks like Tensorflow or PyTorch. 
Just good old:
1) `Math` (linear algebra + calculus) <br>
(check `NN_basis.jpg` - that old 1-sheet note explains 70% of this project)
3) `Programming` (python)
4) `Vectorization using numpy` (aka parallel computing) <br>
  (with vanilla python (like for loops etc.) it was too slow, so I decided to use numpy for matrix multiplications, vectorizing activation functions and so on) <br>

# How to quickly explore
1) `fullly_connected_nn.py` - first version of classic fully connected NN and the simplest to explore. <br>
  All later versions were just like "lets add a couple of new features, fix some buggs". The main added feature is in the file name.
2) `notebook.ipynb` - demonstration of usage. <br>
For each .py version there is corresponding .ipynb notebook.

# Comments
- It was a challenge to prove understanding of key concepts how NN and, in particular, the process of optimization (backpropagation) works. I've really enjoyed it.
- It was not meant to be a production-quality project, so the code and notebooks migth me clumsy and I cant guarantee its reliability. Sorry about that ;)
- The sources of theory concepts and inspiration were series of excellent courses in "Machine learning specialization" and "Deep learning specialization" by Andrew Ng (ex StanfordOnline, now DeepLearning.ai) on coursera.
