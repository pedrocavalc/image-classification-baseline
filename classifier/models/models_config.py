from classifier.models.models import ResNet12, LeNet



def config_models(num_classes):

    models_configs = {'res_net12':ResNet12(3,num_classes), 'LeNet':LeNet(3, num_classes)}
    
    return models_configs