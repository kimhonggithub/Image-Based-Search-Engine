# Image-Based-Search-Engine
Welcome to the repository for the Capstone Project II (2022-2023) at CADT, supervised by Mr. Him Soklong. This project focuses on developing an image-based search engine for an ecommerce website.

### Team members
This project is under the advisorship of mr.Him Soklong and involves four students in the fourth year of the bachelor's in Computer Science:
- Set Kimhong
- Phat Soklieng
- Kun Raksa
- Taing Molika
  
## Abstract
Conventional text-based search in e-commerce platforms faces limitations, especially when customers express product preferences in diverse ways or cannot articulate precise names. This project introduces an innovative image-based search engine to overcome these challenges. Leveraging visual search, users can effortlessly find products through images, transcending language barriers and accommodating varied expressions. This promises to revolutionize the user experience in online shopping, providing a more accurate and inclusive platform.

## Repository structure
- `README.md`: Contains important information about the project.
- `Fashion_Items_Dataset/`: Folder containing the full image dataset used in the project.
- `Fashion_Items_Dataset_Test/`: Folder containing 20% of the full image dataset for testing or experimentation.
- `Image_Similarity_Search.ipynb`: Jupyter Notebook file containing data loader, data preprocessing, feature extraction, model, and evaluation.

## Install instructions
1. Clone our repository :
```sh
git clone https://github.com/kimhonggithub/Image-Based-Search-Engine.git
```
2. Install Pytorch:
```sh
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
3. Install Additional Libraries :
```sh
conda install scikit-learn
```
```sh
conda install Pillow
```
```sh
conda install matplotlib
```
```sh
conda install numpy
```
4. In other to run our code, including in [Code](Code) or our Web app. we would recommend using a virtual environment. This can be done by following the instructions from the [Python website](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
