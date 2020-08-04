
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):

    '''
    Function that performs Random Erasing in Random Erasing Data Augmentation 
    -------------------------------------------------------------------------------------
    p: The probability that the operation will be performed.
    s_l: min erasing area
    s_h: max erasing area
    r_1: min aspect ratio
    r_2: max aspect ratio
    -------------------------------------------------------------------------------------
    '''
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

def plot_confusion_matrix(val_p,test_labels):

    '''
    Computes accuracy of the model and save confusion matrix png file
    -------------------------------------------------------------------------------------
    val_p: predictions on val/test dataset
    test_labels : groundtruth labels
    -------------------------------------------------------------------------------------
    '''

    acc = 0.0
    for i in range(val_p.shape[0]):
      if(val_p[i]==test_labels[i]):
        acc = acc + 1
    print(acc/float(val_p.shape[0]))
    error = 0
    confusion_matrix = np.zeros([10,10])
    for i in range(test_labels.shape[0]):
        confusion_matrix[test_labels[i],val_p[i]] += 1
        if test_labels[i]!=val_p[i]:
            error +=1
    print("Accuracy on test dataset is : ")
    print(acc)
    f = plt.figure(figsize=(10,8.5))
    f.add_subplot(111)

    plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")
    plt.colorbar()
    plt.tick_params(size=5,color="white")
    plt.xticks(np.arange(0,10),np.arange(0,10))
    plt.yticks(np.arange(0,10),np.arange(0,10))

    threshold = confusion_matrix.max()/2 

    for i in range(10):
        for j in range(10):
            plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")
            
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("Confusion_matrix.png")
    plt.show()


def train_data_generator(args):
    '''
    Creates appropriate image generator for training depending upon input arguments
    -------------------------------------------------------------------------------------
    standardize_data : centered images divided by vairance
    use_data_aug : If True. do other data augmentation also apart from standardizatoin
    data_aug : 'Simple' or 'RandomErasing' - value of parameres for both depend upon the model used
    -------------------------------------------------------------------------------------
    '''

    if(args.use_data_aug):
        if(args.data_aug=='simple'):
            if(args.model=='BigNet'):
                train_datagen = ImageDataGenerator(
                    rescale= 1/255.0,
                    featurewise_center=args.standardize_data,  # set input mean to 0 over the dataset
                    featurewise_std_normalization=args.standardize_data,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    fill_mode='constant',
                    cval=0)
            else:
                train_datagen = ImageDataGenerator(
                    rescale= 1/255.0,
                    featurewise_center=args.standardize_data,  # set input mean to 0 over the dataset
                    featurewise_std_normalization=args.standardize_data,
                    horizontal_flip=True,
                    fill_mode='constant',
                    cval=0)


        elif(args.data_aug=='random_erasing'):
            if(args.model=='BigNet'):
                train_datagen = ImageDataGenerator(
                    rescale= 1/255.0,
                    featurewise_center=args.standardize_data,
                    featurewise_std_normalization=args.standardize_data,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    fill_mode='constant',
                    cval=0,
                    preprocessing_function=get_random_eraser(p = 0.2,s_h =0.4,v_l=0, v_h=1, pixel_level=False))
            else:
                train_datagen = ImageDataGenerator(
                    rescale= 1/255.0,
                    featurewise_center=args.standardize_data,
                    featurewise_std_normalization=args.standardize_data,
                    horizontal_flip=True,
                    fill_mode='constant',
                    cval=0,
                    preprocessing_function=get_random_eraser(p = 0.2,s_h =0.2,v_l=0, v_h=1, pixel_level=False))

    else:
        train_datagen = ImageDataGenerator(
            rescale= 1/255.0,
            featurewise_center=args.standardize_data,
            featurewise_std_normalization=args.standardize_data)

    return train_datagen