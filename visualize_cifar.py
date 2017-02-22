import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize image patches")
    parser.add_argument('-mode',
                        type=int,
                        help="Which image set to visualize (1 for kernel challenge data, 0 for cifar data)")
    parser.add_argument('-img_id',
                        type=int,
                        help="No. of image or randomly choose an image (if set as 0)")
    args = parser.parse_args()
    return args

def unpickle(filepath='data/cifar-10-batches-py/data_batch_1'):
    import cPickle
    fo = open(filepath, 'rb')
    data_dict = cPickle.load(fo)
    fo.close()
    return data_dict

if __name__ == '__main__':
#def main():

    args = parse_arguments()
    mode = args.mode
    img_id = args.img_id

    if mode==0:
        data_dict = unpickle()
        X = data_dict['data'].astype('float')
        Y = data_dict['labels']
        X /= 255
    elif mode==1:
        X = np.genfromtxt('data/Xte.csv', delimiter=',')
        X = np.delete(X, np.s_[-1:], 1)
        X += 0.5
    else:
        sys.exit("mode should be either 0 or 1!")

    n_img = X.shape[0]
    X = X.reshape(n_img, 3, 32, 32).transpose(0,2,3,1)
                                       
    if img_id==0:
        print("Randomly display 36 images")
        fig, axes1 = plt.subplots(6,6,figsize=(8,8))
        for j in range(6):
            for k in range(6):
                i = np.random.choice(range(len(X)))
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(X[i:i+1][0])
        plt.show()
    elif 1 <= img_id <=n_img:
        plt.figure()
        plt.imshow(X[img_id-1])
        plt.title("Image No.{}".format(img_id))
        plt.show()
    else:
        sys.exit("incorrect image id!")

 #   main()
