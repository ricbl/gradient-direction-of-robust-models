"""Training script
Use this file to train and validate the results from 
"Quantifying the Preferential Direction of the Model Gradient 
in Adversarial Training With Projected Gradient Descent"
"""

import os

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.optim as optim
from . import opts 
from . import outputs
from . import metrics
from . import penalties
from . import attacks
from. import classifier
from. import generator
from . import synth_dataset
from . import utils_dataset
import numpy as np
import h5py

def init_optimizer(opt, net_d):
    if opt.optimizer_type=='adam':
        optimizer_d = optim.Adam(net_d.parameters(), lr=opt.learning_rate_d, betas=(
                    0.5, 0.999))
    elif opt.optimizer_type=='sgd':
        optimizer_d = optim.SGD(net_d.parameters(), lr=opt.learning_rate_d, momentum = opt.optimizer_momentum)
    return optimizer_d

# precalculate all estimated Delta x from all samples from the datasets to all 
# possible destination classes, so that it does not slow down training.
# For each newly trained cgan, this function will take a long time to run. 
# After running it once for that cgan weights, it will be fast since results
# are saved to an h5 file
def get_correct_gradient(opt, output):
    filepath_dict = './dicts_transform/'+opt.dataset_to_use+'_transformation_all_gens_'+ os.path.normpath(opt.load_checkpoint_g).split(os.path.sep)[-2]+'.h5'
    
    # get all images of the datasets, such that the returned indices fr each sample
    # are ordered
    if opt.dataset_to_use=='mnist':
        from .mnist import get_mnist_dataset as get_dataset
    elif opt.dataset_to_use=='cifar':
        from .cifar import get_cifar_dataset as get_dataset
    elif opt.dataset_to_use=='squares':
        from .synth_dataset import get_squares_dataset
        get_dataset = lambda train: get_squares_dataset(train,opt)
    
    # if not precalculated in previous runs...
    # to check if it was precalculated before, use the name of the dataset and
    # the experiment name from the cgan training run
    if not os.path.exists(filepath_dict):
        net_correct_gradient = generator.init_model(opt)
        net_correct_gradient.eval()
        dict_all_images = []
        d = utils_dataset.return_dataloaders(lambda: utils_dataset.ImageDataset(list(get_dataset(True))+list(get_dataset(False))), opt, opt.split_validation)
        #for all batches of images in the dataset
        for i, (( x,y), index) in enumerate(d):
            output.log_fixed(x, None)
            if i%100==0:
                print(i)
            # calculates the projected images
            out = net_correct_gradient(x.cuda(),y.cuda(), output)
            for j in range(out.size(0)):
                dict_all_images.append(out[j,...].cpu().numpy())
        dict_all_images = np.array(dict_all_images)
        with h5py.File(filepath_dict, 'w') as hf:
            hf.create_dataset(name = 'dict_all_images', data=dict_all_images)
    else:
        dict_all_images = h5py.File(filepath_dict, 'r')['dict_all_images']
    dict_all_images = torch.cat([torch.tensor(dict_all_images[i]).unsqueeze(0) for i in range(dict_all_images.shape[0])],dim=0)
    print(dict_all_images.size())
    def get_correct_gradient_( x,y,index):
        # for estimated Delta x, returns Equation (16)
        return dict_all_images[index].cuda()-x.unsqueeze(1)
    
    return get_correct_gradient_

def correct_square_gradient(x,y, index):
    return torch.stack([torch.tensor(synth_dataset.get_clean_square(0.8 if y[i] else 0.5, 32)-synth_dataset.get_clean_square(0.5 if y[i] else 0.8, 32)).float().cuda().unsqueeze(0) for i in range(x.size(0))])

# defining the losses functions and the function used to get the vector pointing
# from the current input example to the closest example of the opposite class
def get_fns(opt, output):
    loss_fn = lambda d_x, y: torch.nn.CrossEntropyLoss(reduction='none')(d_x, y)
    correct_gradient_fn = get_correct_gradient(opt, output)
    return loss_fn, correct_gradient_fn

def apply_cuda_to_nested_iterators(structure):
    if isinstance(structure, list) or isinstance(structure, tuple):
        return [apply_cuda_to_nested_iterators(item) for index, item in enumerate(structure)]
    elif isinstance(structure, dict):
        return {key: apply_cuda_to_nested_iterators(item) for key, item in structure.items()}
    else:
        return structure.cuda()

# generic training class that is inherited for the two training cycles used for 
# the paper
class TrainingLoop():
    def __init__(self, opt, output,metric):
        self.metric = metric
        self.opt = opt
        self.output = output
        self.loss_fn, self.correct_gradient_fn = get_fns(self.opt, self.output)
    
    def train(self, train_dataloader, val_dataloader, net_d, optim_d):
        # initializing the fixed set of validation examples where to do
        # a limited visualization of results on the validation set
        (fixed_x, fixed_y), fixed_indexes = iter(val_dataloader).next()
        if self.opt.dataset_to_use!='spheres':
            pass
            self.output.log_fixed(fixed_x, fixed_y)
        
        fixed_x = fixed_x.cuda()
        
        fixed_y = fixed_y.cuda()
        fixed_indexes = apply_cuda_to_nested_iterators(fixed_indexes)
        last_best_validation_metric = self.opt.initialization_comparison
        
        for epoch_index in range(self.opt.nepochs):
            if not self.opt.skip_train:
                net_d.train()
                for batch_index, batch_example in enumerate(train_dataloader):
                    if batch_index%100==0:
                        print(batch_index)
                    
                    (x, y), index = batch_example
                    x = apply_cuda_to_nested_iterators(x)
                    y = apply_cuda_to_nested_iterators(y)
                    
                    # call the train_fn, to be defined 
                    # in the child classes. This function does the 
                    # training of the model for the current batch of examples
                    self.train_fn(x,y,index,net_d,optim_d)
                
                self.output.log_batch(epoch_index, self.metric)
            
            #validation
            if not self.opt.skip_validation:
                with torch.no_grad():
                    net_d.eval()
                    # call the validation_of_fixed_images_fn, to be defined 
                    # in the child classes. This function does visualization for a fixed
                    # subset of validation images
                    self.validation_of_fixed_images_fn(fixed_x,fixed_y, fixed_indexes, net_d, epoch_index)
                    
                    if not self.opt.skip_long_validation:
                        for batch_index, batch_example in enumerate(val_dataloader):
                            (x, y), index = batch_example
                            x = apply_cuda_to_nested_iterators(x)
                            y = apply_cuda_to_nested_iterators(y)
                            if batch_index%10==0:
                                print(batch_index)
                            # call the validation_fn function, to be defined 
                            # in the child classes, that defines what to do
                            # during the validation loop
                            self.validation_fn(x,y, index, net_d)
                
                    net_d.train()
            average_dict = self.output.log_added_values(epoch_index, self.metric)
            
            #if training, check if the model from the current epoch is the 
            # best model so far, and if it is, save it
            if not self.opt.skip_train:
                this_validation_metric = average_dict[self.opt.metric_to_validate]
                if self.opt.save_best_model:
                    if self.opt.function_to_compare_validation_metric(this_validation_metric,last_best_validation_metric):
                        self.output.save_models(net_d, 'best_epoch')
                        last_best_validation_metric = this_validation_metric
                if not self.opt.skip_train and self.opt.save_models: 
                    self.output.save_models(net_d, str(epoch_index))

def calculate_jacobian(d_x, x):
    gradients = torch.zeros_like(x).unsqueeze(1).repeat([1,d_x.size(1), 1,1,1])
    for i in range(d_x.size(1)):
        gradients[:,i,...] = torch.autograd.grad(outputs=d_x[:,i].sum(), inputs=x,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    return gradients

#calculates c star, as defined in Theorem 1
def get_closest_class_m(x,y,d_x):
    chosen_class = torch.argmax(d_x, dim=1)
    differences = torch.gather(d_x,1,chosen_class.unsqueeze(1))-d_x
    gradients = calculate_jacobian(d_x, x)
    indices = chosen_class.view([gradients.size(0),1]+[1]*(len(gradients.size())-2)).expand([gradients.size(0),1] + list(gradients.size()[2:]))
    differences_gradients = torch.gather(gradients,1,indices) - gradients
    distances = differences/torch.norm(differences_gradients.view([differences_gradients.size(0),differences_gradients.size(1),-1]), dim = 2)
    distances.scatter_(1, chosen_class.unsqueeze(1), float('inf'))
    return torch.argmin(distances, dim = 1)

#calculates c tilde, according to Equation (5)
def get_closest_class(x,y,d_x):
    chosen_class = y
    differences = torch.gather(d_x,1,chosen_class.unsqueeze(1))-d_x
    gradients = calculate_jacobian(d_x, x)
    indices = chosen_class.view([gradients.size(0),1]+[1]*(len(gradients.size())-2)).expand([gradients.size(0),1] + list(gradients.size()[2:]))
    differences_gradients = torch.gather(gradients,1,indices) - gradients
    distances = differences/torch.norm(differences_gradients.view([differences_gradients.size(0),differences_gradients.size(1),-1]), dim = 2)
    distances.scatter_(1, y.unsqueeze(1), float('inf'))
    return torch.argmin(distances, dim = 1)

def get_alignment(x,y, index, d_x,closest_class,suffix, correct_gradient_fn, metric):
    chosen_class = torch.argmax(d_x, dim=1)
    gradient_chosen_class = torch.autograd.grad(outputs=(torch.gather(d_x,1,chosen_class.unsqueeze(1)).squeeze(1)).sum(), inputs=x,
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_y = torch.autograd.grad(outputs=(torch.gather(d_x,1,y.unsqueeze(1)).squeeze(1)).sum(), inputs=x,
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_closest_class_minus_gradient_correct_class = torch.autograd.grad(outputs=(torch.gather(d_x,1,closest_class.unsqueeze(1)).squeeze(1)-torch.gather(d_x,1,y.unsqueeze(1)).squeeze(1)).sum(), inputs=x,
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients_generator = correct_gradient_fn(x,y, index).cuda()
    indices = closest_class.view([gradients_generator.size(0),1]+[1]*(len(gradients_generator.size())-2)).expand([gradients_generator.size(0),1] + list(gradients_generator.size()[2:]))
    correct_gradients = torch.gather(gradients_generator,1,indices)
    
    closest_class_m = get_closest_class_m(x,y,d_x)
    gradient_closest_class_minus_gradient_correct_class_m = torch.autograd.grad(outputs=(torch.gather(d_x,1,closest_class_m.unsqueeze(1)).squeeze(1)-torch.gather(d_x,1,chosen_class.unsqueeze(1)).squeeze(1)).sum(), inputs=x,
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    for i in range(x.size(0)):
        #calculates the baseline metric, given in Eq. 18 in the paper
        metric.add_value('alpha_x_'+suffix, 
            (torch.abs(penalties.get_cosine_similarity(x[i:i+1], gradient_chosen_class[i:i+1]))).mean())
        #Calculates the baseline metric as defined in the 4th column of Table S2
        metric.add_value('alpha_x_y_'+suffix, 
            (torch.abs(penalties.get_cosine_similarity(x[i:i+1], gradient_y[i:i+1]))).mean())
        #Calculates the baseline metric as defined in the 2nd column of Table S2
        metric.add_value('alpha_x_diffy_'+suffix, 
            (torch.abs(penalties.get_cosine_similarity(x[i:i+1], gradient_closest_class_minus_gradient_correct_class[i:i+1]))).mean())
        #Calculates the baseline metric as defined in the 3rd column of Table S2
        metric.add_value('alpha_x_diff_'+suffix, 
            (torch.abs(penalties.get_cosine_similarity(x[i:i+1], gradient_closest_class_minus_gradient_correct_class_m[i:i+1]))).mean())
        if correct_gradient_fn is not None:
            #calculates the alignment metric, as proposed in Eq. 8 in the paper
            metric.add_value('cosine_similarity_gradient_vs_correctfn_'+suffix, 
                penalties.get_cosine_similarity(
                    gradient_closest_class_minus_gradient_correct_class[i:i+1], 
                    correct_gradients[i:i+1].detach()
                ).mean())
            
            #Calculates the proposed metric, using the gradient as defined in the header of the 3rd column of Table S2
            metric.add_value('cosine_similarity_gradient_vs_correctfn_diffm_'+suffix, 
                penalties.get_cosine_similarity(
                    gradient_closest_class_minus_gradient_correct_class_m[i:i+1], 
                    correct_gradients[i:i+1].detach()
                ).mean())
            
            #Calculates the proposed metric, using the gradient as defined in the header of the 5th column of Table S2
            metric.add_value('cosine_similarity_gradient_vs_correctfn_m_'+suffix, 
                penalties.get_cosine_similarity(
                    gradient_chosen_class[i:i+1], 
                    correct_gradients[i:i+1].detach()
                ).mean())
            
            #Calculates the proposed metric, using the gradient as defined in the header of the 4th column of Table S2
            metric.add_value('cosine_similarity_gradient_vs_correctfn_y_'+suffix, 
                penalties.get_cosine_similarity(
                    gradient_y[i:i+1], 
                    correct_gradients[i:i+1].detach()
                ).mean())

# training of classifiers, considering three types: 
# - baseline , vanilla supervised training
# - Trained with loss penalty L_{\alpha}
# - Adversarially-trained with PGD
class RobustTraining(TrainingLoop):    
    def get_penalty_loss(self,x,y, index, d_x,closest_class):
        chosen_class = y
        gradient_to_penalize = torch.autograd.grad(outputs=(-torch.gather(d_x,1,chosen_class.unsqueeze(1)).squeeze(1)+torch.gather(d_x,1,closest_class.unsqueeze(1)).squeeze(1)).sum(), inputs=x,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = self.correct_gradient_fn(x,y, index).cuda()
        indices = closest_class.view([gradients.size(0),1]+[1]*(len(gradients.size())-2)).expand([gradients.size(0),1] + list(gradients.size()[2:]))
        penalty_loss = penalties.get_cosine_similarity(
            gradient_to_penalize, 
            torch.gather(gradients,1,indices).detach()
        ).mean()
        return penalty_loss
    
    def train_fn(self,x,y,index,net_d,optim_d):
        #sample adversarial example if using this training method
        if self.opt.attack_to_use_training!='none': 
            adversarial = attacks.get_attack(self.opt, x, y, net_d, self.opt.attack_to_use_training, self.loss_fn, self.opt.epsilon_attack_training, k=self.opt.k_pgd_training, alpha_multiplier = self.opt.alpha_multiplier)
            x = adversarial.detach()
        x.requires_grad = True
        
        d_x = net_d(x)
        classifier_loss = self.loss_fn(d_x, y).mean()
        self.metric.add_value('vanilla_loss', classifier_loss)
        closest_class = get_closest_class(x, y, d_x)
        if self.opt.cosine_penalty:
            #if training with alignment penalty, calculates it and add to the classifier loss
            penalty_loss = self.get_penalty_loss(x,y,index,d_x,closest_class)
            penalty_multiplier = -self.opt.lambda_penalty
            #putting together Eq. 9 from the paper
            classifier_loss += penalty_multiplier*penalty_loss
            self.metric.add_value('penalty_loss', penalty_loss)
        get_alignment(x, y, index, d_x, closest_class, 'train', self.correct_gradient_fn, self.metric)
        optim_d.zero_grad()
        classifier_loss.backward()
        optim_d.step()
        self.metric.add_value('classifier_loss', classifier_loss)
        self.metric.add_score(y, d_x, 'train')
    
    # saves images for a fixed set of validation images
    # including gradients, attacked images, generated estimated Delta x, among other things
    def validation_of_fixed_images_fn(self,fixed_x,fixed_y, fixed_indexes, net_d, epoch_index):
        adversarial = attacks.get_attack(self.opt, fixed_x, fixed_y, net_d, self.opt.attack_to_use_val, self.loss_fn, epsilon=0.1, alpha_multiplier = self.opt.alpha_multiplier)
        self.output.log_adversarial( 'val_attack' + '{:05d}'.format(epoch_index) , adversarial)
        if self.opt.attack_to_use_training!='none':
            adversarial  = attacks.get_attack(self.opt, fixed_x, fixed_y, net_d, self.opt.attack_to_use_training, self.loss_fn, self.opt.epsilon_attack_training, k=self.opt.k_pgd_training, alpha_multiplier = self.opt.alpha_multiplier)
            self.output.log_adversarial('attack_with_train_opts' + '{:05d}'.format(epoch_index), adversarial)
        
        prev_grad_enbled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        fixed_x.requires_grad=True
        fixed_d_x = net_d(fixed_x)
                              
        closest_class = get_closest_class(fixed_x,fixed_y, fixed_d_x)
        #save the gradients, for checking if it is more interpretable
        gradient_closest_class_minus_gradient_correct_class = torch.autograd.grad(outputs=(torch.gather(fixed_d_x,1,closest_class.unsqueeze(1)).squeeze(1)-torch.gather(fixed_d_x,1,fixed_y.unsqueeze(1)).squeeze(1)).sum(), inputs=fixed_x,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        self.output.log_adversarial('gradient_val' + '{:05d}'.format(epoch_index), gradient_closest_class_minus_gradient_correct_class/torch.max(torch.abs(gradient_closest_class_minus_gradient_correct_class.view([gradient_closest_class_minus_gradient_correct_class.size(0), -1])), dim=1)[0][(...,)+(gradient_closest_class_minus_gradient_correct_class.ndim-1)*(None,)])
        
        
        if self.opt.dataset_to_use=='squares':
            self.output.log_fixed(correct_square_gradient(fixed_x,fixed_y, fixed_indexes),fixed_y, 'correct')
            self.output.log_fixed(torch.stack([self.correct_gradient_fn(fixed_x,fixed_y, fixed_indexes)[i,1-fixed_y[i]] for i in range(fixed_x.size(0))]), fixed_y, 'generateddelta')
            self.output.log_fixed(torch.stack([self.correct_gradient_fn(fixed_x,fixed_y, fixed_indexes)[i,1-fixed_y[i]]+fixed_x[i] for i in range(fixed_x.size(0))]), fixed_y, 'generatedxprime')
        else:
            # for visualization purposes, you may force the destination class of the gradients to draw to images.
            # Do not set opt.force_closest_class when training
            if self.opt.force_closest_class is not None:
                for fcc in self.opt.force_closest_class:
                    self.output.log_fixed(torch.stack([self.correct_gradient_fn(fixed_x,fixed_y, fixed_indexes)[i,fcc]+fixed_x[i] for i in range(fixed_x.size(0))]), fixed_y, 'difference'+str(fcc))
                    self.output.log_fixed(torch.stack([self.correct_gradient_fn(fixed_x,fixed_y, fixed_indexes)[i,fcc] for i in range(fixed_x.size(0))]), torch.ones_like(fixed_y)*fcc, 'destination'+str(fcc))
            else:
                self.output.log_fixed(torch.stack([self.correct_gradient_fn(fixed_x,fixed_y, fixed_indexes)[i,closest_class[i]]+fixed_x[i] for i in range(fixed_x.size(0))]), fixed_y, 'difference')
                self.output.log_fixed(torch.stack([self.correct_gradient_fn(fixed_x,fixed_y, fixed_indexes)[i,closest_class[i]] for i in range(fixed_x.size(0))]), closest_class, 'destination')
        gradients = self.correct_gradient_fn(fixed_x,fixed_y, fixed_indexes).cuda()
        indices = closest_class.view([gradients.size(0),1]+[1]*(len(gradients.size())-2)).expand([gradients.size(0),1] + list(gradients.size()[2:]))
        self.output.log_delta_x_gt(torch.gather(gradients,1,indices).squeeze(1), '{:05d}'.format(epoch_index))
        
        torch.set_grad_enabled(prev_grad_enbled)
    
    def validation_fn(self,x,y, index, net_d):
        prev_grad_enbled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        x.requires_grad = True
        d_x = net_d(x)
        
        if self.opt.attack_to_use_val=='cwl2':
            gradients = calculate_jacobian(d_x, x)
        closest_class = get_closest_class(x, y, d_x)
        get_alignment(x,y, index, d_x,closest_class,'val', self.correct_gradient_fn, self.metric)
        
        if self.opt.dataset_to_use=='squares':
            # if it is the squares dataset, calculate alignment of the generated Delta x compared 
            # with the correct Delta x calculated from noiseless squares.
            for i in range(x.size(0)):
                self.metric.add_value('cosine_similarity_correct_vs_net_g_val', penalties.get_cosine_similarity(correct_square_gradient(x[i:i+1],y[i:i+1], index[i:i+1]), self.correct_gradient_fn(x[i:i+1],y[i:i+1], index[i:i+1])[:,1-y[i:i+1]]).mean())
            
        torch.set_grad_enabled(prev_grad_enbled)
        self.metric.add_score(y, d_x, 'val', epsilon = 0)
        #adding the predictions of the adversarial attack calculated for several epsilons
        for epsilon in self.opt.epsilons_val_attack:
            adversarial = attacks.get_attack(self.opt, x, y, net_d, self.opt.attack_to_use_val, self.loss_fn , epsilon, alpha_multiplier = self.opt.alpha_multiplier, k=self.opt.k_pgd_training)
            d_out_attack = net_d(adversarial)
            self.metric.add_score(y, d_out_attack, 'attacked_val_epsilon_' + str(epsilon), epsilon)
            if self.opt.attack_to_use_val=='cwl2':
                differences = torch.gather(d_x,1,y.unsqueeze(1))-d_x
                indices = y.view([gradients.size(0),1]+[1]*(len(gradients.size())-2)).expand([gradients.size(0),1] + list(gradients.size()[2:]))
                differences_gradients = torch.gather(gradients,1,indices) - gradients
                distances = differences/torch.norm(differences_gradients.view([differences_gradients.size(0),differences_gradients.size(1),-1]), dim = 2)
                distances.scatter_(1, y.unsqueeze(1), float('inf'))
                self.metric.add_score(torch.linalg.norm((adversarial-x).view([x.size(0),-1]), dim=1).unsqueeze(1), torch.min(distances, dim = 1).values.unsqueeze(1) , 'robustness_vs_approximation')

def main():
    #get user options/configurations
    opt = opts.get_opt()
    torch.backends.cudnn.benchmark = True
    #load Outputs class to save metrics, images and models to disk
    output = outputs.Outputs(opt)
    output.save_run_state(os.path.dirname(__file__))
    
    #get the correct dataset/dataloader
    if opt.dataset_to_use == 'squares':
        from .synth_dataset import get_dataloaders
    elif opt.dataset_to_use=='mnist':
        from .mnist import get_dataloaders
    elif opt.dataset_to_use=='cifar':
        from .cifar import get_dataloaders
    
    #load class to store metrics and losses values
    metric = metrics.Metrics(opt)
    
    loader_train = get_dataloaders(opt, mode='train')
    loader_val_all = get_dataloaders(opt, mode=opt.split_validation)
    
    #load the deep learning architecture for the critic and the generator        
    net_d = classifier.init_model(opt)
    
    
    #load the optimizer
    optim_d = init_optimizer(opt, net_d=net_d)
    
    training_class = RobustTraining
    training_class(opt, output,metric).train(loader_train, loader_val_all,
        net_d=net_d, optim_d=optim_d)

if __name__ == '__main__':
    main()
