# Image Captioning
This project was created as part of a submission for Computer Vision, Nanodegree  via [Udacity](https://eu.udacity.com/course/computer-vision-nanodegree--nd891). 

## Project Overview
In this work we combine Deep Convolutional Nets for image classification  with Recurrent Networks for sequence modeling, to create a single network that generates descriptions of image using [COCO Dataset - Common Objects in Context](http://cocodataset.org/). 

<p align="center"> <img src="images/encoder-decoder.png" align="middle" alt="drawing" width="900px"> </p> 

## Project Structure
The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order:

__Notebook 0__ : 0_Dataset.ipynb

__Notebook 1__ : 1_Preliminaries.ipynb

__Notebook 2__ : 2_Training.ipynb

__Notebook 3__ : 3_Inference.ipynb

## Results
Following are a few results obtained after training the model for 3 epochs.
Image | Caption 
--- | --- 
<img src="images/Surf.png" width="200">
| **Generated Caption:** a person riding a surf board on a wave .
<img src="images/boy.png" width="200"> 
| **Generated Caption:** a young boy brushing his teeth with a toothbrush.
<img src="images/motorcycle.png" width="200"> 
| **Generated Caption:** a group of people riding motorcycles down a street .
<img src="images/vase.png" width="200"> 
| **Generated Caption:** a vase with a flower on a table.

### References

<ul>
<li>[A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)</li>
<li>[Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)</li>
</ul>
