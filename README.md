# ML Coursework

Machine learning assignments from my Master's programme in Artificial Intelligence
at Kazan Federal University (2026). All notebooks were written and run in Google Colab;
outputs and plots are saved, so everything can be read right here on GitHub.

Two parts:

- **[algorithms-from-scratch](algorithms-from-scratch)** — classic ML algorithms implemented
  by hand in NumPy, plus image, video and audio processing.
- **[deep-learning-practice](deep-learning-practice)** — deep learning with TensorFlow / Keras
  and PyTorch: CNNs, `tf.data` pipelines, custom training loops, bias/variance and data mismatch analysis.

Notebook text is in Russian or English, as the assignments were given.

## Algorithms from scratch

| # | Notebook | What is implemented | Tools |
|---|----------|---------------------|-------|
| 1 | [Image compression](algorithms-from-scratch/01-image-compression.ipynb) | Downscaling a colour image k times by averaging pixel blocks with array slicing; averaging colour inside a grid of regions of a mask | NumPy, scikit-image, Matplotlib |
| 2 | [Image, video and audio processing](algorithms-from-scratch/02-image-video-audio-processing.ipynb) | Drawing on images and on every frame of a video; finding a dog bark in an ESC-50 recording by RMS energy and trimming the silence automatically; collecting weather data from the OpenWeather REST API into CSV and plotting it | OpenCV, librosa, SciPy, requests, pandas |
| 3 | [Clustering](algorithms-from-scratch/03-clustering.ipynb) | K-Means with random centroid initialisation, data-driven tolerance and 4 distance metrics; fuzzy C-Means; DBSCAN with a heuristic for choosing `eps` | NumPy, scikit-learn (synthetic datasets), Matplotlib |
| 4 | [Linear regression](algorithms-from-scratch/04-linear-regression.ipynb) | Vectorised linear regression class for any number of features, with feature scaling that is transparent to `predict()` | NumPy, pandas, Matplotlib |
| 5 | [Classifiers](algorithms-from-scratch/05-classifiers.ipynb) | kNN (accuracy vs k), logistic regression, linear SVM and Gaussian naive Bayes written by hand; for a provided decision tree class — a text `print_tree()` | NumPy, scikit-learn (datasets, metrics, reference tree), Matplotlib |

## Deep learning practice

| # | Notebook | Topic | Tools |
|---|----------|-------|-------|
| 1 | [Orthogonalization](deep-learning-practice/01-orthogonalization.ipynb) | Gram–Schmidt process, PCA on Iris, Ridge regularisation strength search | NumPy, scikit-learn |
| 2 | [NumPy basics](deep-learning-practice/02-numpy-basics.ipynb) | Array creation, broadcasting, boolean masks, solving linear systems | NumPy |
| 3 | [Avoidable bias](deep-learning-practice/03-avoidable-bias-pytorch-cnn.ipynb) | CNN baseline, bias/variance against human-level accuracy, learning-rate × weight-decay search, error analysis, augmentation | PyTorch, torchvision |
| 4 | [End-to-end CNN](deep-learning-practice/04-end-to-end-cnn-keras.ipynb) | Building and training an end-to-end CNN on Fashion-MNIST | TensorFlow / Keras |
| 5 | [Data mismatch](deep-learning-practice/05-data-mismatch.ipynb) | Covariate shift: KL divergence, PSI, KS tests with Benjamini–Hochberg, domain classifier AUC, importance weighting | NumPy, SciPy, scikit-learn |
| 6 | [TensorFlow basics](deep-learning-practice/06-tensorflow-basics.ipynb) | Tensors, a dense network for MNIST, `tf.data` pipeline, custom training step with `GradientTape` | TensorFlow |
| 7 | [Keras on MNIST](deep-learning-practice/07-keras-mnist.ipynb) | Data preparation, dense model, training curves, CNN | TensorFlow / Keras |
| 8 | [TensorFlow practice](deep-learning-practice/08-tensorflow-practice.ipynb) | Dense and convolutional models, `tf.data`, custom training loop | TensorFlow |

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

The weather section of notebook 2 needs a free OpenWeather API key in the environment:
`OPENWEATHER_API_KEY=<your key>`. The ESC-50 dataset is not included; download it from
[github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50).
