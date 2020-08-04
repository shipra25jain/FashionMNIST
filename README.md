# FashionMNIST
Image Classification on FashionMNIST dataset
CNN based model for image classification on Fashion-MNIST dataset. Best test accuracy - 94.55 percent. Regularization methods like Dropout, L2 regularization, Data Augmentation and Batch Normalization have been explored.
This code supports python3, keras and tensorflow-v1 and runs on GPU and CPU machines. 

For SmallNet model training, 


python3 main.py --use_data_aug True --use_dropout True --dropout_p 0.2 --lr 0.001 --l2_lambda 0.0 --data_aug random_erasing

For BigNet model training, 


python3 main.py --use_data_aug True --use_dropout True --dropout_p 0.2 --lr 0.001 --l2_lambda 0.0 --model BigNet --use_bn True --batch_size 96

Trained checkpoints can be evaluated by adding :
--ckpt checkpoint_path --inference_only True
