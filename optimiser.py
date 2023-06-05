from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import pickle
import ipywidgets as widgets
import random



from IPython import display as disp
if 'google.colab' in str(get_ipython()):
    from BayesianOptimiserMgAlloy.model_paths import models
    from BayesianOptimiserMgAlloy.BO import alloys_bayes_opt
    from BayesianOptimiserMgAlloy.BO_append import alloys_bayes_opt_append
else:
    from model_paths import models
    from BO import alloys_bayes_opt
    from BO_append import alloys_bayes_opt_append

class AlDatapoint:
    def __init__(self, settings):
        self.constant_inputs = settings.constant_inputs
        self.categorical_inputs = settings.categorical_inputs
        self.categorical_inputs_info = settings.categorical_inputs_info
        self.range_based_inputs = settings.range_based_inputs

    def formatForInput(self):
        my_input = [*self.constant_inputs.values(), *self.categorical_inputs.values(), self.getAl(),
                    *self.range_based_inputs.values()]
        return np.reshape(my_input, (1, -1))

    def print(self):
        for key, value in self.constant_inputs.items():
            print(f"{key}: {value}")
        for key, value in self.categorical_inputs.items():
            print(f"{key}: {self.categorical_inputs_info[key]['tag'][self.categorical_inputs_info[key]['span'].index(value)]}")
        print(f"Al%: {round(self.getAl(), 2)}")
        for key, value in self.range_based_inputs.items():
            if value:
                print(f"{key}: {value}")

    def getAl(self):
        return 100 - sum(self.range_based_inputs.values())


class scanSettings:
    def __init__(self, mode):
        self.mode = mode
        if self.mode == 'Bayesian Optimization':
            self.sampling_size = 100000
            self.num_of_suggestions = 20
            self.num_elems = 6
            self.sum_elems = 20
            self.output_names = ['UTS', 'Ductility']
            self.normalize_target = 'Yes'
            self.append_suggestion = 'Yes'
            self.HT = 'True'
            
            # todo: check arrange of bound dict
            self.range_based_inputs =  {
                'Mg': (80.0, 95.0), 'Y': (0.0, 19.0),
                'Zn': (0.0, 14.3), 'Al': (0.0, 11.0),
                'Ca': (0.0, 10.0), 'Gd': (0.0, 10.0),
                'Sn': (0.0, 9.56), 'Nd': (0.0, 8.05),
                'La': (0.0, 6.0), 'Er': (0.0, 6.0),
                'Si': (0.0, 5.0), 'Ce': (0.0, 3.92),
                'Zr': (0.0, 3.0), 'Li': (0.0, 3.0),
                'Yb': (0, 3), 'Sr': (0.0, 2.45),
                'Mn': (0.0, 2.0), 'Pr': (0.0, 1.76),
                'Ho': (0.0, 1.4), 'Sb': (0.0, 1.0001),
                'Ni': (0.0, 1.0), 'Ga': (0, 1.0),
                'Tb': (0.0, 1.0),
                'Extruded': (0, 1), 'ECAP': (0, 1), 'Cast_Slow': (0, 1),
                'Cast_Fast': (0, 1), 'Cast_HT': (0, 1), 'Wrought': (0, 1),
                'Cu': (0.0, 0.5), 'Ag': (0.0, 0.5),
                'Bi': (0.0, 0.5), 'Sc': (0.0, 0.5),
                'Be': (0.0, 0.0), 'Fe': (0.0, 0.0),
                'Th': (0.0, 0.0), 'Dy': (0.0, 0.0)}

### This is hard coded ***sorted original bound dict*** that is equal to the arrangment of training data (X).
### Models have been trained with this order of X, so we want to have the exact same order in sampler.df.

        if self.mode == 'Mechanical':
            pass
            


class optimiser:
    def __init__(self, settings):        
        self.num_of_suggestions = settings.num_of_suggestions
        self.normalize_target = settings.normalize_target
        self.sampling_size = settings.sampling_size
        self.output_names = settings.output_names
        self.range_based_inputs = settings.range_based_inputs
        self.settings = settings
        self.mode = settings.mode
        self.models = models
        self.run()

    def run(self):
        ############ here
        if self.normalize_target:
            if len(self.output_names)==1:
                if 'UTS' in self.output_names:
                    gp_model_list = [self.models['gp_UTS_normalized']]
                else:
                    gp_model_list = [self.models['gp_Ductility_normalized']]
            else:
                gp_model_list = [self.models['gp_UTS_normalized'], self.models['gp_Ductility_normalized']]
        else:
            if len(self.output_names)==1:
                if 'UTS' in self.output_names:
                    gp_model_list = [self.models['gp_UTS']]
                else:
                    gp_model_list = [self.models['gp_Ductility']]
            else:
                gp_model_list = [self.models['gp_UTS'], self.models['gp_Ductility']]
        if 'UTS' in self.output_names:
            if 'Ductility' in self.output_names:
                rf_model_list = [self.models['rf_UTS'],self.models['rf_Ductility']]
            else:
                rf_model_list = [self.models['rf_UTS']]
        else:
            rf_model_list = [self.models['rf_Ductility']]
        iter_num = self.num_of_suggestions
        bound_dict = self.range_based_inputs
        # categorical_dict = {'Extruded': (0,1), 'ECAP': (0,1), 'Cast_Slow': (0,1), 'Cast_Fast': (0,1), 'Cast_HT': (0,1),'Wrought': (0,1)}
        # bound_dict.update(categorical_dict)


        if self.settings.append_suggestion:
            opt = alloys_bayes_opt_append(output_names = self.output_names, 
                               num_elems=self.settings.num_elems, sum_elems = self.settings.sum_elems, 
                               sample_size=self.sampling_size, iter_num=iter_num,
                               append_suggestion=True, bound_dict = bound_dict, 
                               normalize_target=self.normalize_target)

            df = opt.get_suggestions()


            
        else:
            opt = alloys_bayes_opt(gp_model_list, rf_model_list, output_names = self.output_names, 
                               num_elems=self.settings.num_elems, sum_elems = self.settings.sum_elems, 
                               sample_size=self.sampling_size, iter_num=iter_num,
                               append_suggestion=False, bound_dict = bound_dict, 
                               normalize_target=self.normalize_target)

            df = opt.get_suggestions()

        pd.set_option("display.max_columns", 42)
        
        df.to_csv(str(random.random())+' suggestions.csv',index=False)
        disp.display(df)
        print('========== Bayesian Optimisation Finished ==========')
        print()
        

    def calculateStep(self, best_datapoint, step_number, target_var):
        if target_var == 'all':
            batch_size = self.step_batch_size
        else:
            batch_size = self.finetune_batch_size
        loss = [0] * batch_size
        datapoints = []
        std = self.step_final_std * (self.max_steps / float(step_number + 1))
        for i in range(batch_size):
            datapoints.append(deepcopy(best_datapoint))
            for key in self.categorical_inputs.keys():
                if target_var == key or target_var == 'all':
                    datapoints[i].categorical_inputs[key] = np.random.choice(self.categorical_inputs[key])
            for key in self.range_based_inputs.keys():
                if target_var == key or target_var == 'all':
                    if max(self.range_based_inputs[key]) != min(self.range_based_inputs[key]):
                        a = (min(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
                        b = (max(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
                        datapoints[i].range_based_inputs[key] = round(
                            float(truncnorm.rvs(a, b, loc=np.mean(best_datapoint.range_based_inputs[key]), scale=std)),
                            2)
                    else:
                        datapoints[i].range_based_inputs[key] = min(self.range_based_inputs[key])
            loss[i] = self.calculateLoss(datapoints[i])
        return min(loss), datapoints[loss.index(min(loss))]
