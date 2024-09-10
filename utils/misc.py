from cProfile import label
from ctypes.wintypes import SIZE
import os
import time
from tkinter import font
from tkinter.tix import tixCommand
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from matplotlib import rcParams
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.ticker as ticker

def get_nInt(number):
    n=int(number)
    return len(str(n))

def timestamp():
    this_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    return this_time
    
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def put_Text(image:np.ndarray=None,
            font=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            text_color=(255,128,0),
            thickness=None,
            text=None,
            coordinate=None,
            background_color=(0,47,167)):
    if  thickness == None:
        thickness = fontScale
    if text is None:
        return

    #文字的左下角坐标
    text_coordinate = coordinate
    #得到文字的尺寸大小，基线高度
    retval, baseLine = cv2.getTextSize(text,fontFace=font,fontScale=fontScale, thickness=thickness)

    tl = [text_coordinate[0],text_coordinate[1]-retval[1]]
    br = [tl[0]+retval[0],tl[1]+retval[1]]

    cv2.rectangle(image, tl, br, thickness=-1, color=background_color)

    cv2.putText(image, 
                f'{text}',#文字内容
                coordinate,#文字放置位置坐标
                font,#字体
                fontScale=fontScale,#文字大小
                color=text_color,#文字颜色
                thickness=thickness)#文字线宽


def mat2color(mat:np.ndarray, 
                title:str=None, 
                colormap:str = 'jet',
                norm:str = 'log', 
                figure_size:int=4,
                save_path:str = None,
                has_colorbar:bool=True,
                axis_visible:bool=True,
                cbar_fontsize:int = None,
                ax_fontsize:int =None,
                x_ax_top:bool = True,
                y_ax_top:bool = False,
                title_fontsize:int=None, 
                cbar_ticks_rotation:int=None,
                show_in_jupyter:bool=True,
                cbar_width:float=0.03,
                cbar_left = 0.92,
                cbar_bottom = 0.12,
                alpha:float = 0.,
                transparent:bool=False,
                cbar_ticks:list=None,
                vmin:int=None,
                vmax:int=None,
                interval_cal_base:int=50,
                cbar_auto_log_tick:bool=False,
                auto_tick_intervals:list = [1,5,10,20,50,100,200,500,1000,2000,5000,10000]):
    '''
    datas:(np.ndarray,...[H,W])
    '''
    pos_min=mat.max()//20
    logNorm=False
    ratio = mat.shape[0]/mat.shape[1]
    base_size = figure_size
    fig=plt.figure(figsize=(base_size*1 ,base_size*ratio), dpi=300)

    #设置背景透明度
    fig.patch.set_alpha(alpha) 

    plt.rcParams['axes.facecolor'] = 'black'

    #分割画布
    gscs=gridspec.GridSpec(1, 1)
    ax = [plt.subplot(gsc) for gsc in gscs][0]
    if norm == 'log' :
        if (vmin is not None) and (vmax is not None):
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        logNorm=True
    if norm == 'linear':
        if (vmin is not None) and (vmax is not None):
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    psm = ax.matshow(mat, norm=norm,cmap=colormap, aspect='auto')
    #翻转y坐标
    # axs[i].invert_yaxis()
    #将x轴的位置设置在顶部
    if x_ax_top:
        ax.axes.xaxis.set_ticks_position('top')
    #将y轴的位置设置在顶部
    if y_ax_top:
        ax.axes.yaxis.set_ticks_position('top')
    if ax_fontsize is not None:
        plt.xticks(fontsize=ax_fontsize)
        plt.yticks(fontsize=ax_fontsize)
    if title is not None:
        ax.set_title(title,fontsize=title_fontsize)
    if(not axis_visible):
        ax.get_xaxis().set_visible(False) # 隐藏x坐标轴
        ax.get_yaxis().set_visible(False) # 隐藏y坐标轴

    if has_colorbar:
        #设置右边空余10%位置
        fig.subplots_adjust(right=0.9)
        #colorbar 左 下 宽 高 （百分比）
        l = cbar_left
        b = cbar_bottom
        w = cbar_width
        h = 1 - 2*b 
        #对应 l,b,w,h；设置colorbar位置；
        rect = [l,b,w,h] 
        cbar_ax = fig.add_axes(rect) 
        cbar = fig.colorbar(psm, cax=cbar_ax,format=mpl.ticker.ScalarFormatter())#mpl.ticker.ScalarFormatter()设置十进制显示
        cbar.minorticks_on()
        
        if cbar_ticks is not None:
            tick_locator = ticker.MaxNLocator(nbins=len(cbar_ticks))  # colorbar上的刻度值个数
            cbar.locator = tick_locator
            cbar.set_ticks(cbar_ticks)
            
        #自动设置colorbar的坐标值
        if cbar_auto_log_tick:
            tick_interval = auto_tick_intervals#请根据需要添加值
            max_v = mat.max()
            base_v=min(tick_interval , key=lambda x: abs(x -  (max_v//interval_cal_base)))
            nBase = int(max_v//base_v)
            ticks = [int(i**1.67)*base_v for i in range(1,int(nBase**0.6)+1, 1)]
            ticks = [i for i in ticks if i >= pos_min]
            tick_locator = ticker.MaxNLocator(nbins=len(ticks))# colorbar上的刻度值个数
            cbar.locator = tick_locator
            cbar.set_ticks(ticks)
            
        if cbar_fontsize is not None:
            cbar.ax.tick_params(labelsize=cbar_fontsize)
        if cbar_ticks_rotation is not None:
            cbar.ax.tick_params(rotation=cbar_ticks_rotation)
        cbar.update_ticks()


    if save_path is not None:
        plt.savefig(save_path, transparent=transparent, dpi=300,bbox_inches = 'tight')
        print(f'Saved at {save_path}.')
    if not show_in_jupyter:
        plt.close()
        

def img_show(img:np.ndarray,
             bgr2rgb:bool=True, 
             title:str=None, 
             save_path:str = None,
             ax_fontsize:int=None,
             title_fontsize:int=None, 
             show_in_jupyter:bool=True,
             alpha:float=0.,
             transparent:bool=True,
             base_size:int=4,
             axis_visible:bool=True):
    '''
    img:(np.ndarray,...[H,W,C])
    '''

    ratio = img.shape[0]/img.shape[1]
    base_size = base_size
    fig=plt.figure(figsize=(base_size*1 ,base_size*ratio), dpi=300)
    #设置背景透明度
    fig.patch.set_alpha(alpha)
    if bgr2rgb: 
        img = Image.fromarray(img[:,:,[2,1,0]])
    else:
        img = Image.fromarray(img)
    #分割画布
    fig, ax = plt.subplots()
    ax.imshow(img,aspect='equal')
    #将x轴的位置设置在顶部
    ax.xaxis.set_ticks_position('top')
    if title is not None:
        ax.set_title(title,fontsize=title_fontsize)
    if(not axis_visible):
        ax.get_xaxis().set_visible(False) # 隐藏x坐标轴
        ax.get_yaxis().set_visible(False) # 隐藏y坐标轴
    
    plt.rcParams['axes.facecolor'] = 'black'

    if ax_fontsize is not None:
        plt.xticks(fontsize=ax_fontsize)
        plt.yticks(fontsize=ax_fontsize)

    if save_path is not None:
        plt.savefig(save_path, transparent=transparent, dpi=300,bbox_inches = 'tight')
        print(f'Saved at {save_path}.')
    if not show_in_jupyter:
        plt.close()

def imgsGridsPlot(imgs_list:list,
                nRows:int=None,
                nColumns:int=None,
                figure_size:int=1,
                dpi:int=300,
                left:float=0.1,
                right:float=0.9,
                top:float=0.9,
                bottom:float=0.1,
                wspace:float=0.01,
                hspace:float=0.01,
                axis_visible:bool=False,
                save_path:str=None,
                transparent:bool=True,
                show_in_jupyter:bool=True):
    '''
    imgs_list:List(np.ndarray)
    '''
    if type(imgs_list) == list:
        nimgs = len(imgs_list)
        assert  nimgs>0,'There are no images in the imgs_list!' 
    else:
        raise TypeError(f'List type input is required for imgs_list, but got {type(imgs_list)}')
    if (nRows is not None) and (nColumns is not None):
        assert nRows*nColumns == len(imgs_list), f'nRows*nColumns should equal to {len(imgs_list)}' 
    elif nRows is not None:
        if nRows>nimgs:
            nRows=nimgs
            nColumns=1
        else:
            nColumns = int(np.ceil(nimgs/nRows))
            nRows -= (nColumns*nRows-nimgs)//nColumns 
    elif nColumns is not None:
        if nColumns>nimgs:
            nColumns=nimgs
            nRows=1
        else:
            nRows = int(np.ceil(nimgs/nColumns))
            nColumns -= (nRows*nColumns-nimgs)//nRows
    else:
        raise ValueError('nRowsn or Columns can not be None in the same time.')
    (H,W) = imgs_list[0].shape[:2]
    ratio = H/W 
    fig=plt.figure(figsize=(figure_size*nColumns ,figure_size*ratio*nRows),dpi=dpi,constrained_layout=False)

    #分割画布
    gscs=gridspec.GridSpec(nRows, nColumns)
    axes = [plt.subplot(gsc) for gsc in gscs]

    for i, img in enumerate(imgs_list):
        if not axis_visible:
            axes[i].axis('off')
        axes[i].imshow(img,aspect='equal')


    fig.subplots_adjust(left=left,
                        right=right,
                        top=top,
                        bottom=bottom,
                        wspace=wspace,
                        hspace=hspace)
    # fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, transparent=transparent, dpi=dpi,bbox_inches = 'tight')
    if not show_in_jupyter:
        plt.close()