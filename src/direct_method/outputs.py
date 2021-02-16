"""Output to disk management

This module provides a class that manages all the disk outputs that the 
training requires. This includes saving images, models, tensorboard files and logs.
"""

from tensorboardX import SummaryWriter
import logging
import os
from PIL import Image
import numpy as np
import torch
import glob
import shutil
import sys
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LinearLocator

def permute_channel_last(x):
    return x.clone().permute((0,) + tuple(range(2,len(x.size()))) + (1,))

# plot graph of error rate as a function of perturbation norm
def plot_robust(series_gt, series_predicted, epoch, output_folder, suffix):
    colourWheel =['#99cc00',
            '#008080',
            '#333399',
            '#993366',
            '#ff0000',
            '#ff9900']
    plt.style.use('./plotstule.mplstyle')
    plt.close('all')
    _, ax = plt.subplots()
    series0 = series_gt
    series1 = series_predicted
    alphaVal = 1.0
    _, ax = plt.subplots()
    j = 0
    ax.set_xlabel('Epsilon')
    plt.ylabel('Accuracy')
    if max(series_gt)>0:
        ax.set_xlim(0,max(series_gt)*1.1)        
    
    ax.plot(series0,
                    1-np.array(series1),
                    color=colourWheel[(j)%len(colourWheel)],
                    alpha=alphaVal,zorder=1)
    
    
    ax.scatter(series0,
                1-np.array(series1),
                color=colourWheel[(j+2)%len(colourWheel)],
                s=10,
                alpha=alphaVal,zorder=2)
    

    ax.yaxis.set_minor_locator(  AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(  AutoMinorLocator(5))
    ax.yaxis.set_major_locator(  LinearLocator(6))
    ax.xaxis.set_major_locator(  LinearLocator(6))
    ax.yaxis.tick_right()
    ax.grid(zorder=-100)
    ax.set_axisbelow(True)

def plot_histogram(xhist):
    plt.close('all')
    _, ax = plt.subplots()
    ax.set_xlabel('Cosine similarity')
    plt.ylabel('Ocurrences in set')
    ax.hist(xhist,
                    bins = 16,
                    color='#99cc00',
                    lw = 0.5,
                    edgecolor='black', linewidth=1.0,
                    alpha=1.0,zorder=1, label= 'Histogram of labels')
    ax.yaxis.set_minor_locator(  AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(  AutoMinorLocator(5))
    ax.yaxis.set_major_locator(  LinearLocator(6))
    ax.xaxis.set_major_locator(  LinearLocator(6))
    ax.yaxis.tick_right()
    ax.grid(zorder=-100)
    ax.set_axisbelow(True)

def plot_scores(outputs, norms):
    colourWheel =['#99cc00',
            '#008080',
            '#333399',
            '#993366',
            '#ff0000',
            '#ff9900']
    plt.style.use('./plotstule.mplstyle')
    plt.close('all')
    _, ax = plt.subplots()
    alphaVal = 1.0
    _, ax = plt.subplots()
    j = 0
    ax.set_xlabel('Radius')
    plt.ylabel('Output')
    if max(norms)>0:
        ax.set_xlim(min(norms)*0.95,max(norms)*1.05)
    for j in range(outputs.size(1)):
        ax.plot(norms,
            outputs[:,j],
            color=colourWheel[(j)%len(colourWheel)],
            alpha=alphaVal,zorder=1)
    
    ax.yaxis.set_minor_locator(  AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(  AutoMinorLocator(5))
    ax.yaxis.set_major_locator(  LinearLocator(6))
    ax.xaxis.set_major_locator(  LinearLocator(6))
    ax.yaxis.tick_right()
    ax.grid(zorder=-100)
    ax.set_axisbelow(True)

def save_image(filepath, numpy_array):
    numpy_array = np.clip(numpy_array, -1, 1)
    im = Image.fromarray(((numpy_array*0.5 + 0.5)*255).astype('uint8'))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(filepath)

class Outputs():
    def __init__(self, opt):
        if not os.path.exists(opt.save_folder):
            os.mkdir(opt.save_folder)
        output_folder = opt.save_folder+'/'+opt.experiment+'_'+opt.timestamp
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        logging.basicConfig(filename = output_folder +'/log.txt' ,level = logging.INFO)
        self.log_configs(opt)
        self.writer = SummaryWriter(output_folder + '/tensorboard/')
        #get number of rows for the saved images as approximally the sqrt of the number of batch examples to save in the same image
        self.nrows_fixed = int(math.sqrt(opt.batch_size))-[(opt.batch_size%i==0) for i in range(int(math.sqrt(opt.batch_size)),0,-1)].index(True)
    
    def log_configs(self, opt):
        logging.info('-------------------------------used configs-------------------------------')
        for key, value in sorted(vars(opt).items()):
            logging.info(key + ': ' + str(value).replace('\n', ' ').replace('\r', ''))
        logging.info('-----------------------------end used configs-----------------------------')
    
    def log_fixed(self,fixed_x, fixed_y):
        self.save_batch(self.output_folder+'/real_samples.png', fixed_x)
        with open(self.output_folder+'/real_samples_gt.txt', 'w') as f:
            out_gt = np.vstack(np.hsplit(np.hstack(fixed_y.cpu()), self.nrows_fixed))
            for i in range(out_gt.shape[0]):
                f.write(str( out_gt[i,:]))
                f.write('\n')
    
    def log_adversarial(self, suffix, fixed_x):
        path = self.output_folder + '/adversarial_samples_'+suffix+'.png'
        self.save_batch(path, fixed_x)
    
    # activate the average calculation for all metric values and save them to log and tensorboard
    def log_added_values(self, epoch, metrics):
        if 'cosine_similarity_gradient_vs_correctfn_val' in metrics.values.keys():
            plot_histogram([value for value in metrics.values['cosine_similarity_gradient_vs_correctfn_val'] if (value!=float("-inf") and value!=float("inf") and value==value)])
            self.save_plt(self.output_folder+'/cosine_similarity_correct_val_'+str(epoch)+'.png')
        averages, score_epsilon = metrics.get_average()
        logging.info('Metrics for epoch: ' + str(epoch))
        for key, average in averages.items():
            self.writer.add_scalar(key, average, epoch)
            logging.info(key + ': ' + str(average))
        if ('gen_loss' in averages.keys()) and ('classifier_loss' in averages.keys()):
            averages['total_loss'] = averages['gen_loss']+averages['classifier_loss']
            self.writer.add_scalar('total_loss', averages['total_loss'],epoch)
        self.writer.flush() 
        epsilons = []
        scores = []
        for key, score in score_epsilon.items():
            epsilons.append(key)
            scores.append(score)
        if len(scores)>0:
            print(epsilons)
            print(scores)
            plot_robust(epsilons, scores, epoch, self.output_folder, 'val_attacked')
            self.save_plt(self.output_folder+'/robust_'+str(epoch)+'.png')
        return averages
    
    def log_images(self, epoch, fixed_x, delta_x):
        path = '{:}/delta_x_samples_{:05d}.png'.format(self.output_folder, epoch)
        self.save_batch(path, delta_x)
        path = '{:}/xprime_samples_{:05d}.png'.format(self.output_folder, epoch)
        self.save_batch(path, fixed_x+ delta_x)
    
    def log_batch(self, epoch, metric):
        try:
            gen_loss = metric.get_last_added_value('gen_loss').item()
            gen_loss_defined = True
        except IndexError:
            gen_loss = 0
            gen_loss_defined = False
        try: 
            classifier_loss = metric.get_last_added_value('classifier_loss').item()
            classifier_loss_defined = True
        except IndexError:
            classifier_loss = 0
            classifier_loss_defined = False
        loss_string = {(True, True):"Total", (True, False):"Generator", (False, True):"Classifier", (False, False): "Undefined"}[(gen_loss_defined, classifier_loss_defined)]
        loss_string = "; " + loss_string + " loss: "
        logging.info('Epoch: ' + str(epoch) + loss_string + str(gen_loss + classifier_loss))
    
    def log_delta_x_gt(self, delta_x_gt):
        path = '{:}/delta_x_gt.png'.format(self.output_folder)
        self.save_batch(path, delta_x_gt)
    
    #save the source files used to run this experiment
    def save_run_state(self, py_folder_to_save):
        if not os.path.exists('{:}/src/'.format(self.output_folder)):
            os.mkdir('{:}/src/'.format(self.output_folder))
        [shutil.copy(filename, ('{:}/src/').format(self.output_folder)) for filename in glob.glob(py_folder_to_save + '/*.py')]
        self.save_command()
    
    # plot the output of the model as a function of input l2 norm for the spheres dataset
    def plot_score_evolution(self, net_d, epoch, opt):
        xs = []
        norms = np.exp(np.arange(-2.3,3.4125,0.0125))
        zoomed_norm_i0 = 180
        zoomed_norm_i1 = 210
        for norm in norms: #-2.3 to 3.4, in 0.05 steps
            x = np.random.normal(size=[opt.n_dimensions_sphere])
            x = norm*x/np.linalg.norm(x)
            xs.append(torch.tensor(x).float())
        outputs = net_d(torch.stack(xs, dim=0).cuda())
        outputs = torch.sigmoid(outputs)
        plot_scores(outputs.cpu(), norms)
        self.save_plt(self.output_folder+'/score_evolution_'+str(epoch)+'.png')
        plot_scores(outputs.cpu()[zoomed_norm_i0:zoomed_norm_i1], norms[zoomed_norm_i0:zoomed_norm_i1])
        self.save_plt(self.output_folder+'/score_evolution_zoomed_'+str(epoch)+'.png')
    
    def save_plt(self, filepath):
        plt.savefig(filepath, bbox_inches = 'tight', pad_inches = 0)
        plt.show()
    
    #saves the command line command used to run these experiments
    def save_command(self, command = None):
        if command is None:
            command = ' '.join(sys.argv)
        with open("{:}/command.txt".format(self.output_folder), "w") as text_file:
            text_file.write(command)
    
    def save_models(self, net_d, suffix, net_g=None):
        torch.save(net_d.state_dict(), '{:}/state_dict_d_'.format(self.output_folder) + str(suffix)) 
        if net_g is not None:
            torch.save(net_g.state_dict(), '{:}/state_dict_g_'.format(self.output_folder) + str(suffix))
            
    def save_batch(self, filepath, tensor):
        if tensor.ndim==4:
            tensor = permute_channel_last(tensor)
            if tensor.size(3)==1:
                tensor = tensor.squeeze(-1)
        numpy_array = tensor
        try:
            numpy_array = numpy_array.detach().cpu().numpy()
        except:
            pass
        numpy_array = np.vstack(np.hsplit(np.hstack(numpy_array), self.nrows_fixed))
        save_image(filepath, numpy_array)
    