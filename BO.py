# todo: uncomment this?!
# import alloysHT
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import warnings
from sklearn.ensemble import RandomForestRegressor
import copy
import time


OG_bound_dict = {'Mg': (80.0, 95.0), 'Zn': (0.0, 14.3),'Al': (0.0, 11.0), 'Mn': (0.0, 2.0),
                           'Nd': (0.0, 8.05), 'Ce': (0.0, 3.92), 'La': (0.0, 6.0), 'Zr': (0.0, 3.0),
                           'Cu': (0.0, 0.5), 'Si': (0.0, 5.0), 'Y': (0.0, 19.0), 'Ca': (0.0, 10.0),
                           'Pr': (0.0, 1.76), 'Ni': (0.0, 1.0), 'Be': (0.0, 0.0), 'Fe': (0.0, 0.0),
                           'Li': (0.0, 3.0), 'Gd': (0.0, 10.0), 'Th': (0.0, 0.0), 'Sn': (0.0, 9.56),
                           'Sb': (0.0, 1.0001), 'Ag': (0.0, 0.5), 'Ga': (0, 1.0), 'Yb': (0, 3),
                           'Bi': (0.0, 0.5), 'Sc': (0.0, 0.5), 'Dy': (0.0, 0.0), 'Sr': (0.0, 2.45),
                           'Tb': (0.0, 1.0), 'Er': (0.0, 6.0), 'Ho': (0.0, 1.4),
                           'Extruded': (0,1), 'ECAP': (0,1), 'Cast_Slow': (0,1),
                           'Cast_Fast': (0,1), 'Cast_HT': (0,1),'Wrought': (0,1)}


OG_bound_dict = dict(sorted(OG_bound_dict.items(), key=lambda item: item[1][1], reverse=True))
CHT = list(OG_bound_dict.keys())


data = alloysHT.data.drop_duplicates(subset = CHT, inplace=False)
X = data[CHT] # Comp + HT
UTS = data["UTS(MPa)"] #UTS
Ductility = data["Ductility"]



# todo: we need two models for normalized output and not normalized one
class alloys_bayes_opt:
    def __init__(self, gp_model_list, rf_model_list, x=X, y=None, z=None, output_names = ['UTS', 'Ductility'] ,kernel='rat_quad', normalize_y=False, 
                 num_elems= 6, sum_elems = 20, sample_size=10000,
                 append_suggestion=False, iter_num=10, 
                 model_name='rf', kappa = 0.05, util_type = 'ucb', bound_dict = [] , normalize_target=True):
        self.num_elems= num_elems
        self.sum_elems= sum_elems
        self.sample_size=sample_size
        
        self.kernel = self.get_kernel(kernel)
        self.kernel_name = kernel
        self.normalize_y=normalize_y
       
        self.append_suggestion = append_suggestion
        self.iter_num = iter_num
        self.model_name = model_name
        self.kappa = kappa
        self.util_type = util_type # ucb, ei, poi
        self.x = x
        self.y = y
        self.z = z
        self.output_names = output_names
        
        self.normalize_target = normalize_target
        self.elem_og_sorted = list(bound_dict.keys())
        bound_dict = dict(sorted(bound_dict.items(), key=lambda item: item[1][1], reverse=True))
        self.bound_dict = bound_dict
        self.samples_df = pd.DataFrame()

        
        if len(self.output_names)==1:            
            self.bo_suggestion_df = pd.DataFrame(columns= self.elem_og_sorted +
                                             [self.output_names[0] + "_model" 
                                              ])
            self.gp = gp_model_list[0]
            self.model = rf_model_list[0]
            
        else:
            self.MO_bo_suggestion_df = pd.DataFrame(columns= self.elem_og_sorted +
                                                [output_names[0] + "_model" ,
                                                 
                                                 output_names[1] + "_model"  
                                                 ])
            #self.gp1 = gp_model_list[0]
            #self.gp2 = gp_model_list[1]  
            self.model1 = rf_model_list[0]
            self.model2 = rf_model_list[1]
            
            
        self.gp_result_df = pd.DataFrame(columns=['Kernel','iteration', 'GP_Score', 
                                                  'GP_LOGlikelihood' , 'GP_Predicted_Max', 
                                                  'GP_Std_Max',  'GP_VARcriteria'])

        # self.gp = GaussianProcessRegressor(kernel=self.kernel,
        #                                    n_restarts_optimizer=9,
        #                                    normalize_y= self.normalize_y,
        #                                    alpha=1e-2, random_state=0)
        # self.gp.fit(self.x, self.y)
            
        self.utility = UtilityFunction(kind=self.util_type, kappa=self.kappa, xi=0)
        
       
    @staticmethod
    def get_kernel(kernel):
        if kernel == 'dot_white':
            k = kernel = DotProduct() + WhiteKernel()
        elif kernel == 'rat_quad':
            k = RationalQuadratic(length_scale=1, alpha=1.5)
        elif kernel == 'c_rbf':
            k = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-1, 1e1))
        elif kernel == 'dot_cnt':
             k = ConstantKernel(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2)
        elif kernel == 'mat':
            k = Matern(length_scale=1.0, nu=1.5)
        else:
            print('invalid kernel name')
            return None
        return k
       
    @staticmethod
    def define_rf_model():
        model = RandomForestRegressor(n_estimators = 800 , 
                                      min_samples_split = 5,
                                      min_samples_leaf = 1,
                                      max_features = 'sqrt',
                                      max_depth = 100, 
                                      bootstrap = False,
                                      random_state=0 )
        return model
           
   
    def refit_gp_model(self, x, y):
        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                           n_restarts_optimizer=9,
                                           normalize_y= self.normalize_y,
                                           alpha=1e-2, random_state=0)
        self.gp.fit(x, y)
       
    
    def sampler(self):  # the sum of all random variables should be equal to one
        sample_size = self.sample_size
        num_elems = self.num_elems
        summ = self.sum_elems
        random_set = {}
        #bound_dict = dict(sorted(self.bound_dict.items(), key=lambda item: item[1][1], reverse=True))
        bound_dict = self.bound_dict
        keys = list(bound_dict.keys())
        
        # all the alloying elements except Fe and Be and Th
        ht_list = ['Extruded', 'ECAP','Cast_Slow', 'Cast_Fast', 'Cast_HT','Wrought']
        for i in ht_list:
            keys.remove(i)
        
        for key in keys:
            if bound_dict[key][1]==0.0:
                keys.remove(key)
        
        if 'Mg' in keys:
            keys.remove('Mg')
        else:
            keys = keys
            
        num_metals = np.random.randint(3,num_elems+1, size=sample_size) ### 3 means that minimum 3 alloying elements exist        
        final_samples = []
        
        s = 0
        while s <self.sample_size:
        #for s in range(sample_size):
            chosen_materials = np.random.choice(keys, num_metals[s], replace=False)
            for i, k in enumerate(bound_dict):
                if k in chosen_materials:
                    lower = max(summ-sum([v for _, v in random_set.items()])-bound_dict[k][1], 0) # this is upper bound of element k in chosen materials
                    # it means we are considering all chosen materials are sticking to their upper bounds.
                    if summ-sum([v for _, v in random_set.items()])-bound_dict[k][0] < 0:
                        upper = summ-sum([v for _, v in random_set.items()])
                    else:
                        upper = summ-sum([v for _, v in random_set.items()])-bound_dict[k][0]
                    search_bound = (lower,  upper)
                    r = np.random.uniform(search_bound[0],  search_bound[1])
                    random_set[k]= summ - r - sum([v for _, v in random_set.items()])
                else:
                    random_set[k]=0  
            random_set["Mg"]= 100 - sum([v for k, v in random_set.items()])            
            
            for l,j in enumerate(ht_list):
                temp_sample = copy.copy(random_set)
                temp_sample[ht_list[l]]= 1
                if temp_sample["Mg"]<bound_dict['Mg'][1] and bound_dict[ht_list[l]][1]==1: #### ignore samples with Mg content greater than upper bound of Mg (here 95%)
                    temp_sample
                    final_samples.append(temp_sample)
            s+=1                    
        
        sampler_df = pd.DataFrame (final_samples, columns= list(bound_dict.keys()))
        sampler_df = sampler_df[self.elem_og_sorted]


        #print(self.elem_og_sorted)
        #print(sampler_df.columns)

        return sampler_df
    
    def get_suggestions(self):
        if len(self.output_names)==1:
            return self.next_suggestions()
        else:
            return self.MO_next_suggestions()
    
    
    def next_suggestions(self):
        n = self.iter_num
        for i in range(n):
            print("Sample ", i+1, " is preparing...")
            print("It takes a few minutes")
            print()
            self.samples_df = self.sampler() ### if we don't want to append sampler within the i iterations
            samples_df = self.samples_df
                
            utils = self.utility.utility(samples_df, self.gp, 0)
            next_point_index = np.argmax(utils)
            
            predicted_Y, std = self.gp.predict(samples_df, return_std=True)
            
            X_next = self.samples_df.iloc[next_point_index, :]
            Y_next = self.model.predict(X_next.values.reshape(1,-1))  
            Next_alloy = pd.concat([pd.Series(X_next), pd.Series(Y_next)], axis=0, ignore_index=True)
            #Next_alloy = pd.concat([Next_alloy, pd.Series(predicted_Y[next_point_index])], axis=0, ignore_index=True)
            self.bo_suggestion_df.loc[self.bo_suggestion_df.shape[0]] = list(Next_alloy) # to add a new row to dataframe with the same columns order
            
            # if self.append_suggestion:                
            #     self.x = pd.concat([self.x , pd.DataFrame(X_next).transpose()], ignore_index=True)                
            #     self.y = pd.concat([self.y, pd.Series(Y_next)], ignore_index=True)
                
                # self.model = self.define_rf_model()
                # self.model.fit(self.x, self.y)
                # self.refit_gp_model(self.x, self.y)                 

        return self.bo_suggestion_df
                 
    
    ## MO: Mulitu-Objective function (UTS and Duuctility)
    def MO_next_suggestions(self):        
        ## gp1 = GaussianProcessRegressor(kernel=self.kernel,
        ##                                    n_restarts_optimizer=9,
        ##                                    normalize_y= self.normalize_y,
        ##                                    alpha=1e-2, random_state=0)
        ## gp2 = GaussianProcessRegressor(kernel=self.kernel,
        ##                                    n_restarts_optimizer=9,
        ##                                    normalize_y= self.normalize_y,
        ##                                    alpha=1e-2, random_state=0)
        ## if self.normalize_target:
        ##     gp1.fit(self.x, self.y_scaled)
        ##     gp2.fit(self.x, self.z_scaled)
        ##     
        ## else: 
        ##     gp1.fit(self.x, self.y)
        ##     gp2.fit(self.x, self.z)
        ##         
        ## rf1 = self.define_rf_model()
        ## rf2 = self.define_rf_model()
        ## rf1.fit(self.x, self.y)
        ## rf2.fit(self.x, self.z)
        
        #gp1 = self.gp1
        #gp2 = self.gp2
        rf1 = self.model1
        rf2 = self.model2
        
        gp = GaussianProcessRegressor(kernel=self.kernel,
                                           n_restarts_optimizer=9,
                                           normalize_y= self.normalize_y,
                                           alpha=1e-2, random_state=0)
                                           
        if self.normalize_target:
             gp.fit(self.x, self.y_scaled * self.z_scaled)
        
        
        else: 
             gp.fit(self.x, self.y * self.z)
             
             
        n = self.iter_num
        for i in range(n):
            print("Sample ", i+1, " is preparing...")
            print("It takes a few minutes")
            print()
            self.samples_df = self.sampler() ### if we don't want to append sampler within the i iterations
            samples_df = self.samples_df
                
                
            utils = gp.predict(samples_df, return_std = True)[0]+ self.kappa*(gp.predict(samples_df, return_std = True)[1].reshape(-1,1))
            
            
            
            next_point_index = np.argmax(utils)
            


            X_next = self.samples_df.iloc[next_point_index, :]
            Y_next = rf1.predict(X_next.values.reshape(1,-1)) 
            Z_next = rf2.predict(X_next.values.reshape(1,-1)) 
            
            Next_alloy = pd.concat([pd.Series(X_next), pd.Series(Y_next)], axis=0, ignore_index=True)
            #Next_alloy = pd.concat([Next_alloy, pd.Series(predicted_Y[next_point_index])], axis=0, ignore_index=True)
            Next_alloy = pd.concat([Next_alloy, pd.Series(Z_next)], axis=0, ignore_index=True)
            #Next_alloy = pd.concat([Next_alloy, pd.Series(predicted_Z[next_point_index])], axis=0, ignore_index=True)
            
            
            self.MO_bo_suggestion_df.loc[self.MO_bo_suggestion_df.shape[0]] = list(Next_alloy) # to add a new row to dataframe with the same columns order
            
            
      
      
    
        return  self.MO_bo_suggestion_df
        
        
        


