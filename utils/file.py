import numpy as np
import torch
import pandas as pd
import csv
from tqdm import tqdm
import pickle
import os
from typing import Tuple


def new_path(path:str, i:int):
    '''
    不改变扩展名，将指定编号加在文件名后上
    '''
    file_name_split = os.path.splitext(path)
    #得到除扩展名以外的部分
    base_name = file_name_split[0]
    #得到扩展名
    file_extension = file_name_split[1]
    #新文件名
    new_path = base_name+str(i)+file_extension

    return new_path

def save_eval(data:Tuple[str,dict],file_name:str):
    '''
    @brief:使用eval来保存文件
    @args:
        data(str):A str has expression format inside like ''dict(a=1,b=2)''.
        file_name(str): Absolutely file-direction of data.
    '''
    # if isinstance(data, str):
    #     data = eval(data)
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(str(data))  # dict to str
    


def load_eval(file_name:str):
    '''
    @brief:Open a file by eval
    @args:
        file_name(str): Absolutely file-direction of data.
    '''
    with open(file_name, 'r', encoding='utf-8') as f:
        data = eval(f.read())  # eval
        return data



def save_pickle(data,file_name:str):
    '''
    @brief:使用pickle来保存文件
    @args:
        data:Any data that you want to save as piclkle format,it could be str、array、dict etc.
        file_name(str): Absolutely file-direction of data.
    '''
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name:str):
    '''
    @brief:Open file by pickle
    @args:
        file_name(str): Absolutely file-direction of data.
    '''
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def rename_files(dir:str, 
                 suffix:str, 
                 start:int,
                 lenth_of_new_name:int=6):
    
    files = find_file(dir,suffix=suffix)
    pbar = tqdm(files,total=len(files),desc=f'[{0}][{len(files)}]')
    for i, old_file in enumerate(pbar):
        file_idx = start+i
        base_dir = os.path.split(old_file)[0]
        new_file = os.path.join(base_dir, f'{file_idx:0{lenth_of_new_name}}'+f'.{suffix}')
        os.rename(old_file, new_file)
        pbar.set_description(f'Set old file {old_file} to {new_file}.')


def CrossOver(dir,list,suffix,mode):
    for i in os.listdir(dir):  #遍历整个文件夹
        path = os.path.join(dir, i)
        if os.path.isfile(path):  #判断是否为一个文件，排除文件夹
            if os.path.splitext(path)[1] == suffix:
                if mode == 'abs':
                    list.append(path)
                elif mode == 'name':
                    list.append(i)
                else:
                    raise ValueError(f"Only support 'abs' and 'name' mode, but got {mode}" )
        elif os.path.isdir(path):
            newdir=path
            CrossOver(newdir,list,suffix,mode)
    return list

def find_file(dir:str=None, suffix:str=None, mode:str='abs')->list:
    '''
    @brief:Finds a file with the same extension in the specified path.
    @args:
        dir(str): Folder path.
        suffix(str): File suffix. For example: '.png'、'.jpg'、'.mp4'
        mode('abs' or 'name'): Select the output absolute path or file name.'
    '''
    list=[]
    output=CrossOver(dir, list, suffix, mode)
    return output

  

