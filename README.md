# Rover-SolarPredNet

This repository contains all the code necessary to train and evaluate SolarPredHybrid and SolarPredBaseline, two deep neural network models with a CNN backbone
trained to predict solar energy generation for a mobile robot. This project was completed as part of coursework for ROB501: Computer Vision for Robotics at the
University of Toronto. Models are trained using the The Canadian Planetary Emulation Terrain Energy-Aware Rover Navigation Dataset by the UTIAS STARS Laboratory
(Olivier Lamarre, Oliver Limoyo, Filip Maric, Jonathan Kelly).

Link to the dataset: https://starslab.ca/enav-planetary-dataset/

This repository contains: 
- data_loader.py: 	Generates custom PyTorch datasets for solar energy generation prediction
- SolarPredHybrid.py:	Defines a hybrid model with two CNN components: IrradianceNet and SolarPoseNet that are combined to predict solar energy generation
- SolarPredConv.py: 	Defines a fully-convolutional model serving as a baseline Neural Network model to compare to SolarPredHybrid
- train.py: 		Trains neural network components on solar energy generation dataset
- evaluate.py: 		Evaluates solar energy generation prediction models on representative test


