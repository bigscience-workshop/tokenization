sudo sudo apt-get update
sudo apt-get install git -y
sudo apt-get install wget -y
sudo apth-get install tmux -y

# Install Rust
sudo apt install build-essential -y
sudo apt-get install pkg-config
sudo apt-get install libssl-dev
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

# Setup conda
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
sh Miniconda3-py38_4.10.3-Linux-x86_64.sh
# You should enter a bunch of things manually

# Clone tokenizers
mkdir ~/code
cd ~/code
git clone https://github.com/huggingface/tokenizers.git
cd tokenizers
git checkout bigscience_fork
cd bindings/python
pip install setuptools_rust
pip install -e .

# Setup tokenization repo
mkdir ~/code
cd ~/code
git clone https://github.com/bigscience-workshop/tokenization.git
cd tokenization
git checkout thomas/train
pip install -r requirements.txt

# install datasets locally
mkdir -p ~/tokenization_dataset
cd ~/tokenization_dataset
gsutil -m cp -r gs://bigscience-backups/dataset/tokenization_dataset/* .
