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
import copy
import math

def init_optimizer(opt, net_g, net_d):
    if len(list(net_g.parameters()))>0:
        optimizer_g = optim.Adam(net_g.parameters(), lr=opt.learning_rate_g, betas=(
                0.0, 0.9), weight_decay=0)
    else:
        optimizer_g = None
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.learning_rate_d, betas=(
                0.0, 0.9), weight_decay=0)
    return optimizer_g, optimizer_d

# defining the losses functions and the function used to get the vector pointing
# from the current input example to the closest example of the opposite class
def get_fns(opt, net_g):
    # The loss function had to be redefined, because numerical issues in 
    # the original pytorch function causes it to be assymetrical between the
    # two classes.
    loss_fn = lambda d_x, y: y*torch.nn.BCEWithLogitsLoss(reduction='none')(-d_x, torch.zeros_like(y.float()))+(1-y)*torch.nn.BCEWithLogitsLoss(reduction='none')(d_x, torch.zeros_like(y.float()))
    
    # to prevent numerical issues in some cases where only the direction of 
    # the gradient of the loss mmatters, and not its magitude, we use this 
    # loss that uses the logits values directly, only changing their signs
    logits_loss_fn = lambda d_x, y: ((y.float()-0.5)*-2*d_x)
    
    #regularization function of the adapted VRGAN training, as defined in Eq. 12
    # in the paper.
    reg_fn = lambda x: torch.norm(x.view([x.size(0),-1]), dim = 1)/math.sqrt(x.view([x.size(0),-1]).size(1))
    
    if opt.load_checkpoint_g is not None and not opt.vrgan_training:
        net_correct_gradient = copy.deepcopy(net_g)
        net_correct_gradient.eval()
        correct_gradient_fn = lambda x,y: net_correct_gradient(x,1-y,y)
    elif opt.dataset_to_use =='spheres':
        correct_gradient_fn = lambda x,y: (y-0.5)*-2*x
    elif opt.dataset_to_use =='squares':
        correct_gradient_fn = lambda x,y: torch.stack([torch.tensor(synth_dataset.get_clean_square(0.8 if y[i] else 0.5, 224)-synth_dataset.get_clean_square(0.5 if y[i] else 0.8, 224)).float().cuda().unsqueeze(0) for i in range(x.size(0))])
    else:
        # for the MNIST and the COPD dataset, this function should be defined
        # by loading a generator checkoint above
        correct_gradient_fn = None
    return loss_fn, logits_loss_fn, reg_fn, correct_gradient_fn

# generic training class that is inherited for the two training cycles used for 
# the paper
class TrainingLoop():
    def __init__(self, opt, net_g, output,metric):
        self.metric = metric
        self.opt = opt
        self.output = output
        self.loss_fn, self.logits_loss_fn, self.reg_fn, self.correct_gradient_fn = get_fns(opt, net_g)
    
    def train(self, train_dataloader, val_dataloader, net_g, net_d, optim_g, optim_d):
        
        # initializing the fixed set of validation examples where to do
        # a limited visualization of results on the validation set
        fixed_x, fixed_y = iter(val_dataloader).next()
        if self.opt.dataset_to_use!='spheres':
            self.output.log_fixed(fixed_x, fixed_y)
        fixed_x = fixed_x.cuda()
        
        fixed_y = fixed_y.cuda()
        if self.opt.dataset_to_use!='spheres':
            if self.correct_gradient_fn is not None:
                self.output.log_delta_x_gt(self.correct_gradient_fn(fixed_x,fixed_y))
        
        last_best_validation_metric = self.opt.initialization_comparison
        
        for epoch_index in range(self.opt.nepochs):
            if not self.opt.skip_train:
                net_d.train()
                net_g.train()
                for batch_index, batch_example in enumerate(train_dataloader):
                    if batch_index%1000==0:
                        print(batch_index)
                    
                    x, y = batch_example
                    x = x.cuda()
                    y = y.cuda()
                    x.requires_grad =  True;
                    # call the train_fn, to be defined 
                    # in the child classes. This function does the 
                    # training of the model for the current batch of examples
                    self.train_fn(x,y,net_d,net_g,optim_d,optim_g)
                
                self.output.log_batch(epoch_index, self.metric)
            
            #validation
            with torch.no_grad():
                net_d.eval()
                net_g.eval()
                # call the validation_of_fixed_images_fn, to be defined 
                # in the child classes. This function does visualization for a fixed
                # subset of validation images
                self.validation_of_fixed_images_fn(fixed_x,fixed_y, net_d, net_g, epoch_index)
                
                for batch_index, batch_example in enumerate(val_dataloader):
                    x, y = batch_example
                    x = x.cuda()
                    y = y.cuda()
                    if batch_index%10==0:
                        print(batch_index)
                    # call the validation_fn function, to be defined 
                    # in the child classes, that defines what to do
                    # during the validation loop
                    self.validation_fn(x,y, net_d, net_g)
            
                net_d.train()
                net_g.train()
            average_dict = self.output.log_added_values(epoch_index, self.metric)
            
            #if training, check if the model from the current epoch is the 
            # best model so far, and if it is, save it
            if not self.opt.skip_train:
                this_validation_metric = average_dict[self.opt.metric_to_validate]
                if self.opt.save_best_model:
                    if self.opt.function_to_compare_validation_metric(this_validation_metric,last_best_validation_metric):
                        self.output.save_models(net_d, 'best_epoch', net_g)
                        last_best_validation_metric = this_validation_metric
                if not self.opt.skip_train and self.opt.save_models: 
                    self.output.save_models(net_d,str(epoch_index), net_g)

# training of classifiers, considering three types: 
# - baseline , vanilla supervised training
# - Trained with loss penalty L_{\alpha}
# - Adversarially-trained with PGD
class RobustTraining(TrainingLoop):
    def train_fn(self,x,y,net_d,net_g,optim_d,optim_g):
        #sample adversarial example if using this training method
        if self.opt.attack_to_use_training!='none': 
            adversarial = attacks.get_attack(self.opt, x, y, net_d, self.opt.attack_to_use_training, self.logits_loss_fn, self.opt.epsilon_attack_training, k=self.opt.k_steps_attack, alpha_multiplier = self.opt.alpha_multiplier)
            x = adversarial.detach()
            x.requires_grad = True
        
        d_x = net_d(x)
        classifier_loss = self.loss_fn(d_x, y).mean()
        
        if self.opt.cosine_penalty:
            #if training iwth alignment penalty, calculates it and add to the classifier loss
            correct_gradient = self.correct_gradient_fn(x,y)
            gradients = torch.autograd.grad(outputs=classifier_loss, inputs=x, create_graph=True, retain_graph=True, only_inputs=True)[0]
            penalty_loss = penalties.get_cosine_similarity(gradients,correct_gradient.detach()).mean()
            penalty_multiplier = -self.opt.lambda_penalty
            #putting together Eq. 9 from the paper
            classifier_loss += penalty_multiplier*penalty_loss
            self.metric.add_value('penalty_loss', penalty_loss)
            if self.correct_gradient_fn is not None:
                for i in range(x.size(0)):
                    self.metric.add_value('cosine_similarity_gradient_vs_correctfn_train', penalties.get_cosine_similarity(gradients[i:i+1], self.correct_gradient_fn(x[i:i+1],y[i:i+1])).mean())
        optim_d.zero_grad()
        classifier_loss.backward(retain_graph = True)
        optim_d.step()
        self.metric.add_value('classifier_loss', classifier_loss)
        self.metric.add_score(y, d_x, 'train')
    
    def validation_of_fixed_images_fn(self,fixed_x,fixed_y, net_d, net_g, epoch_index):
        if self.opt.dataset_to_use!='spheres':
            adversarial = attacks.get_attack(self.opt, fixed_x, fixed_y, net_d, self.opt.attack_to_use_val, self.logits_loss_fn, epsilon=0.1, alpha_multiplier = self.opt.alpha_multiplier)
            self.output.log_adversarial( 'val_attack' + '{:05d}'.format(epoch_index) , adversarial)
            if self.opt.attack_to_use_training!='none':
                adversarial  = attacks.get_attack(self.opt, fixed_x, fixed_y, net_d, self.opt.attack_to_use_training, self.logits_loss_fn, self.opt.epsilon_attack_training, k=self.opt.k_steps_attack, alpha_multiplier = self.opt.alpha_multiplier)
                self.output.log_adversarial('attack_with_train_opts' + '{:05d}'.format(epoch_index), adversarial)
            
            prev_grad_enbled = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            fixed_x.requires_grad=True
            fixed_d_x = net_d(fixed_x)
            
            fixed_l_dx = self.logits_loss_fn(fixed_d_x, fixed_y).mean()
            fixed_gradients = torch.autograd.grad(outputs=fixed_l_dx, inputs=fixed_x,
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
            #save the gradients, for checking if it is more interpretable
            self.output.log_adversarial('gradient_val' + '{:05d}'.format(epoch_index), fixed_gradients/torch.max(torch.abs(fixed_gradients.view([fixed_gradients.size(0), -1])), dim=1)[0][(...,)+(fixed_gradients.ndim-1)*(None,)])
            torch.set_grad_enabled(prev_grad_enbled)
        else:
            self.output.plot_score_evolution(net_d,epoch_index, self.opt)
    
    def validation_fn(self,x,y, net_d, net_g):
        
        prev_grad_enbled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        x.requires_grad = True
        d_x = net_d(x)
        l_dx = self.logits_loss_fn(d_x, y).sum()
        
        gradients_logits = torch.autograd.grad(l_dx, inputs=x,create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        for i in range(x.size(0)):
            #calculates the baseline metric, given in Eq. 18 in the paper
            self.metric.add_value('alpha_x_val', (torch.abs(penalties.get_cosine_similarity(x[i:i+1], gradients_logits[i:i+1]))).mean())
            if self.correct_gradient_fn is not None:
                #calculates the alignment metric, as proposed in Eq. 8 in the paper
                self.metric.add_value('cosine_similarity_gradient_vs_correctfn_val', penalties.get_cosine_similarity(gradients_logits[i:i+1], self.correct_gradient_fn(x[i:i+1],y[i:i+1])).mean())
        torch.set_grad_enabled(prev_grad_enbled)
        self.metric.add_score(y, d_x, 'val', epsilon = 0)
        #adding the predictions of the adversarial attack calculated for several epsilons
        for epsilon in self.opt.epsilons_val_attack:
            adversarial = attacks.get_attack(self.opt, x, y, net_d, self.opt.attack_to_use_val, self.logits_loss_fn , epsilon, alpha_multiplier = self.opt.alpha_multiplier, k=self.opt.k_steps_attack)
            d_out_attack = net_d(adversarial)
            self.metric.add_score(y, d_out_attack, 'attacked_val_epsilon_' + str(epsilon), epsilon)
            if self.opt.attack_to_use_val=='cwl2':
                distances = torch.abs(d_x).squeeze(1)/torch.norm(gradients_logits.view([gradients_logits.size(0),-1]), dim = 1)
                self.metric.add_score(torch.norm((adversarial-x).view([x.size(0),-1]), dim=1).unsqueeze(1), distances.unsqueeze(1) , 'robustness_vs_approximation')

# calculates the norms of differents stages in the VRGAN algorithm. Mainly used
# to check for norms in the spheres dataset
def add_norms_metrics_vrgan(metric, y, x, correct_gradient):
    for i in range(x.size(0)):
        metric.add_value('average_norm_original_' + str(y[i].item()), torch.norm(x[i]))
        metric.add_value('average_norm_delta_' + str(y[i].item()), torch.norm(correct_gradient[i]))
        metric.add_value('average_norm_sum_' + str(y[i].item()), torch.norm(x[i]+correct_gradient[i]))

class VRGANTraining(TrainingLoop):
    def train_fn(self,x,y,net_d,net_g,optim_d,optim_g):
        neg_y = 1-y
        
        #classifier steps
        delta_x = net_g(x, neg_y,y)
        x_prime = x + delta_x
        d_xprime = net_d(x_prime)
        l_dxprime = self.loss_fn(d_xprime, y).mean()
        
        d_x = net_d(x)
        l_dx = self.loss_fn(d_x, y).mean()
        
        #using terms defined in Eq. 11
        classifier_loss = self.opt.lambda_dx * l_dx + self.opt.lambda_dxprime * l_dxprime
        
        optim_d.zero_grad()
        classifier_loss.backward(retain_graph = True)
        optim_d.step()
        self.metric.add_value('l_dx', l_dx)
        self.metric.add_value('l_dxprime', l_dxprime)
        self.metric.add_value('classifier_loss', classifier_loss)
        
        #generator steps
        optim_g.zero_grad()
        l_g = self.loss_fn(d_xprime, neg_y).mean()
        
        l_regg = self.reg_fn(delta_x).mean()
        #using terms defined in Eq. 11 and Eq. 12
        gen_loss = self.opt.lambda_regg*l_regg + self.opt.lambda_g*l_g
        gen_loss.backward(retain_graph = True)
        optim_g.step()
        self.metric.add_value('gen_loss', gen_loss)
        self.metric.add_value('l_regg', l_regg)
        self.metric.add_value('l_g', l_g)
    
    def validation_of_fixed_images_fn(self,fixed_x,fixed_y, net_d, net_g, epoch_index):
        if self.opt.dataset_to_use!='spheres':
            # save images of the generated approximations of \Delta x
            fixed_neg_y = 1-fixed_y
            delta_x = net_g(fixed_x, fixed_neg_y, fixed_y)
            self.output.log_images(epoch_index, fixed_x, delta_x)
            
    def validation_fn(self,x,y, net_d, net_g):
        neg_y = 1-y
        delta_x_val = net_g(x, neg_y, y)
        add_norms_metrics_vrgan(self.metric, y, x, delta_x_val)
        if self.correct_gradient_fn is not None:
            for i in range(x.size(0)):
                #calculate the similarity between the generated \Delta x and the true
                # \Delta x for the Spheres and Squares datasets
                self.metric.add_value('cosine_similarity_correct_vs_net_g_val', penalties.get_cosine_similarity(delta_x_val[i:i+1] , self.correct_gradient_fn(x[i:i+1],y[i:i+1])).mean())

def main():
    #get user options/configurations
    opt = opts.get_opt()
    if opt.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    
    #load Outputs class to save metrics, images and models to disk
    output = outputs.Outputs(opt)
    output.save_run_state(os.path.dirname(__file__))
    
    #load class to store metrics and losses values
    metric = metrics.Metrics(opt)
    
    #get the correct dataset/dataloader
    if opt.dataset_to_use=='spheres':
        from .sphere_dataset import get_dataloaders
    elif opt.dataset_to_use == 'squares':
        from .synth_dataset import get_dataloaders
    elif opt.dataset_to_use=='copd':
        from .xray_loader import get_dataloaders
    elif opt.dataset_to_use=='mnist':
        from .mnist import get_dataloaders
    loader_train = utils_dataset.ChangeLoaderSize(get_dataloaders(opt, mode='train'), opt.total_iters_per_epoch)
    loader_val_all = get_dataloaders(opt, mode=opt.split_validation)
    
    #load the deep learning architecture for the critic and the generator
    net_d = classifier.init_model(opt)
    net_g = generator.init_model(opt)
    
    #load the optimizer
    optim_g, optim_d = init_optimizer(opt, net_g=net_g, net_d=net_d)
    
    #choose the model to train, generator (of the vector poiting to the closest example of the oppositee class ) 
    # or classifier 
    if opt.vrgan_training:
        training_class = VRGANTraining
    else:
        training_class = RobustTraining
    
    # train the model
    training_class(opt, net_g, output,metric).train(loader_train, loader_val_all,
          net_g=net_g if opt.vrgan_training else generator.NullGenerator(), net_d=net_d, optim_g=optim_g, optim_d=optim_d)

if __name__ == '__main__':
    main()
