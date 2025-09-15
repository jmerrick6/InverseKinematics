# Project Overview
This project implements a pipeline for estimating robotic arm inverse kinematic solutions using machine learning: data generation, preprocessing, model training, evaluation, and visualization.

## File Reference

### `/docs/`: Data files and model artifacts  
- `/docs/images`: Contains images for report.
- `/docs/ik_model_for_viz.npz`: IK results for visualization. 
- `/docs/ik_loss_curves.npz`: IK results for loss per epoch visualization. 
- `/docs/per_cluster_regressor.joblib`: Regressor for per-cluster prediction.
- `/docs/MidtermReport.md`: Midterm Report. 
- `/docs/FinalReport.md`: Final Report. 


### `/classifer_src/`: Source code for classifier pipeline
- `/classifer_src/__init__.py`: Marks `classifier_src/` as a Python package.  
- `/classifer_src/config_classifier_6DOF.py`: Hyperparameters and other utils for configuring classifier.
- `/classifer_src/data_analysis_6DOF.py`: Sanity check to ensure labels of training dataset.
- `/classifer_src/data_generation_6DOF.py`: Sanity check to ensure labels of training dataset.
- `/classifer_src/dataset_classifier_6DOF.py`: Loads classificaiton dataset for training, transforms to torch-compatible.
- `/classifer_src/model_classifier_6DOF.py`: Defines the MLP model used for classification.
- `/classifer_src/train_classifier_6DOF.py`: Trains and evaluates the model. Plots performance metrics. 


### `/inv_kin_src/`: Source code for inverse kinematics pipeline
- `/inv_kin_src/__init__.py`: Marks `inv_kin_src/` as a Python package.  
- `/inv_kin_src/InverseKinematicsEstimator.ipynb`: Jupyter notebook for inverse kinematics pipeline.
- `/inv_kin_src/Methods.py`: Support and client methods for the notebook.
- `/inv_kin_src/simple_4dof.urdf`: Describes robot geometry.
- `/inv_kin_src/Visualize_Robot.py`: Visualizes results of IK data acquisition.
- `/inv_kin_src/VisualizeResults.py`: Visualization for IK training results.
- `/inv_kin_src/LossFunctionPlot.py`: Visualization for IK training loss function per epoch.

### `/checkpoints/`: Model training checkpoints for quick resume in case of crash.


### `/legacy/`: Legacy code of previous iterations of the IK and classifier models.
