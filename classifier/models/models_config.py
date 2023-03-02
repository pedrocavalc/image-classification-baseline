from classifier.models.models import ResNet12, LeNet



def config_models(num_classes,lr):

    models_configs = {'res_net12':ResNet12(3,num_classes,lr), 'LeNet':LeNet(3, num_classes,lr)}
    
    return models_configs