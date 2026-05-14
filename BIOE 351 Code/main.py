# Modules

import os
import pandas as pd
# https://pandas.pydata.org/docs/user_guide/dsintro.html
import matplotlib as plt
import numpy as np
from scipy.optimize import fsolve
import shutil
from openpyxl import load_workbook
from openpyxl.styles import Font

# File path stuff

xcel_dir = "C:/Users/treyc/Documents/Github Repos/College/BIOE 351 Code"
xcel_file = "TEMPLATE_contact_angle_BIOE_351.xlsx"

xcel_path = os.path.join(xcel_dir, xcel_file)

xcel_data = pd.read_excel(xcel_path, sheet_name=0, header=2, engine='openpyxl')

student_name = 'Trey_Cherry'

# Hardcoded variables

y_s_ptfe = 18
y_p_ptfe = 0
y_d_ptfe = 18

# Shit im ripping from the xlsx

y_l_water = xcel_data.iloc[0,1] 

theta_ptfe_deg = xcel_data.iloc[0,2]
theta_ptfe_rad = np.radians(theta_ptfe_deg)

def goofy_silly_number_diddler(y_dl):
    
    lhs = y_l_water * (1 + np.cos(theta_ptfe_rad))
    rhs = 2 * np.sqrt(y_d_ptfe * y_dl)
    
    dif_left_right = lhs - rhs
    
    return dif_left_right 

y_dl_guess = 1.0 

y_dl_1_water = fsolve(goofy_silly_number_diddler, y_dl_guess)

print(y_dl_1_water)

dispersive_components = []
polar_components = []

for idx, row in xcel_data.iterrows():
    
    solvent = row['Solvent']
    y_l = row['tension']
    
    theta_ptfe = row['theta on PTFE']
    theta_rad = np.radians(theta_ptfe)
    
    def equation_1(y_dl):
        
        lhs = y_l * (1 * np.cos(theta_rad))
        rhs = 2 * np.sqrt(y_d_ptfe * y_dl)
        
        return lhs - rhs
    
    try: 
        
        y_dl = fsolve(equation_1, y_l*0.3)
        y_dl = max(0, min(y_dl, y_l))
        
        y_pl = y_l - y_dl 
        
        dispersive_components.append(y_dl)
        polar_components.append(y_pl)
        
        print(f'{solvent}: y_d = {y_dl}, y_p = {y_pl}')
        
        
    except:
        
        dispersive_components.append(np.nan)
        polar_components.append(np.nan)
        
        raise ValueError('it didnt work dumbass')
    
    
# Continue here