"""Metrics

File that contain a class Metrics for storing losses and metrics during 
mini batch inferences, so that you can get an epoch summary after the epoch 
is complete.
"""

import collections
from .classifier import get_correct_examples
import statistics

def accuracy(y_corr, y_pred):
    assert(len(y_pred)==len(y_corr))
    correct = (1-get_correct_examples(y_pred, y_corr)).sum()
    return correct/float(len(y_pred))

class Metrics():
    def __init__(self, opt):
        self.values = collections.defaultdict(list)
        self.score_fn = accuracy 
        self.registered_score = {}
        self.attack_to_use_val = opt.attack_to_use_val
    
    def add_list(self, key, value):
        value = value.detach().cpu().tolist()
        self.values[key] += value
        
    def add_value(self, key, value):
        value = value.detach().cpu()
        self.values[key].append( value)
    
    def calculate_average(self):
        self.average = {}
        self.score_epsilon = {}
        for key, element in self.values.items():
            if key[:7]=='y_true_' or key[:12]=='y_predicted_':
                continue
            n_values = len(element)
            if n_values == 0:
                self.average[key] = 0
                continue
            sum_values = sum(element)
            self.average[key] = sum_values/float(n_values)
        if self.attack_to_use_val=='cwl2':
            self.average['score_cw'] = accuracy(self.values['y_true_attacked_val_epsilon_0'], self.values['y_predicted_attacked_val_epsilon_0'])
            self.average['epsilon_0.5'] = statistics.median(self.values['y_true_robustness_vs_approximation'])
            self.average['epsilon_average'] = sum(self.values['y_true_robustness_vs_approximation'])/len(self.values['y_true_robustness_vs_approximation'])
        else:
            for suffix in self.registered_score.keys():
                self.average['score_' + suffix] = self.score_fn(self.values['y_true_' + suffix], self.values['y_predicted_' + suffix])
            previous_score_average = 0
            previous_suffix = None
            self.average['epsilon_0.5'] = 0
            if len(list(self.registered_score.keys()))>0:
                threshold = min(0.5,self.average['score_' + list(self.registered_score.keys())[-1]]-0.01)
            
            # for all epsilons
            for suffix in self.registered_score.keys():
                if self.registered_score[suffix]>=0:
                    self.score_epsilon[self.registered_score[suffix]] = self.average['score_' + suffix]
                if self.average['score_' + suffix]>=threshold and previous_score_average<threshold:
                    if previous_suffix is not None:
                        #calculate linear interpolation for finding where the curve crosses 0.5
                        self.average['epsilon_0.5'] = (threshold-previous_score_average)*(self.registered_score[suffix]-self.registered_score[previous_suffix])/(self.average['score_' + suffix]-previous_score_average)+self.registered_score[previous_suffix]
                previous_score_average = self.average['score_' + suffix]
                previous_suffix = suffix
        self.values = collections.defaultdict(list)
    
    def get_average(self):
        self.calculate_average()
        return self.average, self.score_epsilon
        
    def get_last_added_value(self,key):
        return self.values[key][-1]
    
    # for adding predictions and groundtruths, used later to calculate accuracy.
    # The input epsilon is greater than 0 for predctions calculated against 
    # adversarial attacks
    def add_score(self, y_true, y_predicted, suffix, epsilon = -1):
        self.registered_score[suffix] = epsilon
        y_predicted = y_predicted.detach().squeeze(1)
        if len(y_true.size())>1:
            y_true = y_true.detach().squeeze(1)
        self.add_list('y_true_' + suffix, y_true)
        self.add_list('y_predicted_' + suffix, y_predicted)