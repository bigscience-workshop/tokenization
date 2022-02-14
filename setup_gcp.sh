sudo sudo apt-get update
sudo apt-get install git -y
sudo apt-get install wget -y
sudo apth-get install tmux -y

# Setup conda
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
sh Miniconda3-py38_4.10.3-Linux-x86_64.sh
# You should enter a bunch of things manually

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
