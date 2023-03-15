from classifier.models.models import *



def config_models(num_classes,lr):

    models_configs = {
                      #'res_net12':ResNet12(3,num_classes,lr), 
                      #'LeNet':LeNet(3, num_classes,lr),
                      #'AlexNet': AlexNet(3, num_classes,lr),
                      'DenseNet': DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=num_classes,lr = lr)
                      }
    
    return models_configs