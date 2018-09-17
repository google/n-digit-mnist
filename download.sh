mkdir -p data

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O data/train-images-idx3-ubyte.gz
  wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O data/train-labels-idx1-ubyte.gz
  wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O data/t10k-images-idx3-ubyte.gz
  wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O data/t10k-labels-idx1-ubyte.gz

elif [[ "$OSTYPE" == "darwin"* ]]; then
  curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -o data/train-images-idx3-ubyte.gz
  curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -o data/train-labels-idx1-ubyte.gz
  curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -o data/t10k-images-idx3-ubyte.gz
  curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -o data/t10k-labels-idx1-ubyte.gz

fi

gzip -d data/train-images-idx3-ubyte.gz
gzip -d data/train-labels-idx1-ubyte.gz
gzip -d data/t10k-images-idx3-ubyte.gz
gzip -d data/t10k-labels-idx1-ubyte.gz

mv data/train-images-idx3-ubyte data/train-images.idx3-ubyte
mv data/train-labels-idx1-ubyte data/train-labels.idx1-ubyte
mv data/t10k-images-idx3-ubyte data/t10k-images.idx3-ubyte
mv data/t10k-labels-idx1-ubyte data/t10k-labels.idx1-ubyte

