from classifier.train import train
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="data_dir", type=str, help="path to dataset folder")
    parser.add_argument("--b", dest="batch", type=int, help="batch_size")
    parser.add_argument("--aug", dest="aug", type=bool, help= "Apply augmentation")
    parser.add_argument("--epochs", dest="epochs", type=int, help= "Number of epochs to train")
    parser.add_argument("--max_lr", dest="lr", type=float, help= "Initial learning rate to learning rate schedule")
    args = parser.parse_args()
    train(data_dir=args.data_dir,batch_size=args.batch,augmentation=args.aug,epochs=args.epochs, max_lr= args.lr)