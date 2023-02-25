from classifier.train import train
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="data_dir", type=str, help="path to dataset folder")
    parser.add_argument("--n_classes", dest="n_classes", type=int, help="number of classes that will be predicted")
    parser.add_argument("--b", dest="batch", type=int, help="batch_size")
    parser.add_argument("--aug", dest="aug", type=bool, help= "Apply augmentation")
    parser.add_argument("--epochs", dest="epochs", type=int, help= "Number of epochs to train")
    parser.add_argument("--max_lr", dest="lr", type=float, help= "Initial learning rate to learning rate schedule")
    parser.add_argument("--n_devices", dest="n_devices", type=int, help= "Number of GPU devices available on Machine")
    parser.add_argument("--accelerator", dest="accelerator", type=str, help= "Type of the device available on Machine, 'cuda' if has a GPU with cuda available 'CPU' if not has")
    args = parser.parse_args()
    train(data_dir=args.data_dir,n_classes=args.n_classes, batch_size=args.batch,augmentation=args.aug,epochs=args.epochs, max_lr= args.lr,n_devices=args.n_devices,
          accelerator=args.accelerator)