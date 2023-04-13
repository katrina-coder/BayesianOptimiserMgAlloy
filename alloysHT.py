"""
Module provides access to Mg-Alloy dataset with some predefined short names for convenience
to access only composition attributes and any or all of the properties.
"""

import pandas as pd



data = pd.read_excel('M-916 - wt%.xlsx') #.drop(columns=['Name', 'Condition', 'Sum'])
data["Coded"].replace({1: "Extruded", 2: "ECAP", 3: "Cast_Slow", 4: "Cast_Fast", 5: "Cast_HT", 6:"Wrought"}, inplace=True)
data.rename(columns= {"Coded": "Processing"} , inplace=True )
data = pd.get_dummies(data, columns=['Processing'])
data.rename(columns={'Processing_Extruded': 'Extruded', 'Processing_ECAP': 'ECAP', 'Processing_Cast_Slow': 'Cast_Slow', 'Processing_Cast_Fast': 'Cast_Fast', 'Processing_Cast_HT': 'Cast_HT',
                     'Processing_Wrought': 'Wrought'}, inplace=True)



target_names = ['YS(MPa)', 'UTS(MPa)', 'Ductility']
element_names = ['Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn', 
                 'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Be', 'Fe',
                 'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi',  'Extruded', 'ECAP',
       'Cast_Slow', 'Cast_Fast', 'Cast_HT',
       'Wrought']

compositions = data[element_names]
targets = data[target_names]

ys = data['YS(MPa)']
uts = data['UTS(MPa)']
duct = data['Ductility']
