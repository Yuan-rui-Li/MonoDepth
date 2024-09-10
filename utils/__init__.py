from matplotlib import font_manager
import matplotlib.pyplot as plt
import os
currentPath = os.getcwd().replace('\\','/')
font_dir = 'utils/fonts'
font_manager.fontManager.addfont(os.path.join(currentPath,font_dir,'Times_New_Roman.ttf'))
font_manager.fontManager.addfont(os.path.join(currentPath,font_dir,'simsun.ttc'))
plt.rcParams['font.family'] = ['Times New Roman','SimSun']

