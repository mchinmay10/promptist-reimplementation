# colab_setup.sh

echo "Running Colab setup..."
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-distutils

echo "Creating virtual environment..."
python3.10 -m venv /content/py310

echo "Python version:"
source /content/py310/bin/activate && python --version

# echo "Cloning git repo..."
# git clone https://github.com/mchinmay10/promptist-reimplementation.git

echo "Changing the working directory..."
cd promptist-reimplementation/

echo "Installing project requirements..."
source /content/py310/bin/activate && pip install -r requirements.colab.txt

!/content/py310/bin/python -c "import sys, torch, transformers, diffusers; import clip, ImageReward; print('Python:', sys.version); print('CUDA:', torch.cuda.is_available()); print('Transformers:', transformers.__version__); print('Diffusers:', diffusers.__version__); print('CLIP OK'); print('ImageReward OK')"

echo "Setup done..."
echo "Ready to run python scripts now"