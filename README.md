# n-digit MNIST

MNIST handwritten digits have been arguably the most popular dataset for machine learning research.
Although the state-of-the-art learned models have long ago reached possibly the best achievable performances on this benchmark,
the dataset itself remains useful to the research community, providing a simple sanity check for new methods:
if it doesn't work on MNIST, it doesn't work anywhere!

We introduce n-digit variants of MNIST here. 
By adding more digits per data point, one can exponentially increase the number of classes for the dataset.
Nonetheless, they still take advantage of the simpleness and light-weighted nature of data.
These datasets provide a simple and useful toy examples for e.g. face embedding.
One can furthermore draw an analogy between individual digits and e.g. face attributes.
In this case, the dataset serves to provide quick insights into the embedding algorithm to be scaled up to more realistic, slow-to-train problems.

Due to potential proprietarity issues and greater flexibility, we release the code for _generating_ the dataset from the original MNIST dataset,
rather than releasing images themselves. 
For benchmarking purposes, we release four _standard_ datasets which are, again, generated via code, but deterministically. 

# Dataset protocols

Given `n`, the number of digits per sample, we generate data samples which are horizontal concatenations of the original MNIST digit images.
We introduce training and test sets, each of which are built from individual digit images from original training and test sets, respectively.
In both training and test splits, each n-digit class has exactly the same number of examples.

## Generating dataset

### Dependencies

Only `numpy` is required.

### Download the original MNIST dataset

Running
``` shell
./download.sh
```
will download the original MNIST dataset from [official MNIST website](http://yann.lecun.com/exdb/mnist/)
and unzip the files in the `data/` folder:
``` shell
data/train-images.idx3-ubyte
data/train-labels.idx1-ubyte
data/t10k-images.idx3-ubyte
data/t10k-labels.idx1-ubyte
```

### Creating the standard n-digit MNIST datasets

We have four _standard n-digit MNIST_ datasets ready: *mnist_2_instance*, *mnist_2_number*, *mnist_3_instance*, *mnist_3_number*.
Unlike custom-built datasets, they are deterministically generated from pre-computed random arrays.
These datasets are suitable for benchmarking model performances. 

Above four datasets can be created by attaching the `--use_standard_dataset` flag.


``` shell
python n_digit_mnist.py --num_digits 2 --domain_gap instance --use_standard_dataset
python n_digit_mnist.py --num_digits 2 --domain_gap number --use_standard_dataset
python n_digit_mnist.py --num_digits 3 --domain_gap instance --use_standard_dataset
python n_digit_mnist.py --num_digits 3 --domain_gap number --use_standard_dataset
```

To optionally check samples from the dataset, run our example script (requires `pillow` package)

``` shell
python example_visualization.py
```

which extracts a few samples in the visualization subfolder (e.g. `data/dataset_mnist_2_number/visualization`). 

### Create your own dataset

See `n_digit_mnist.py` argument options and configure a new dataset yourself.
Example of 4-digit MNIST with `number` domain gap:

``` shell
python n_digit_mnist.py --num_digits 4 --domain_gap number
```

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

This is not an officially supported Google product.
