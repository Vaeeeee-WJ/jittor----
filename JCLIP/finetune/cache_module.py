import pandas as pd
from   PIL import Image
import jittor as jt
import jittor.nn as nn
import jclip as clip
from utils import generate_prompt
jt.flags.use_cuda =   1   


def cache_module(root_TrainSet, class_4_path : str , classes_path : str, model, preprocess):
    
    '''
    root_TrainSet  : 训练集的根目录
    class_4_path   : train_4class.txt文件路径
    classes_path   : classes.txt or classes_b.txt文件路径
    model          : 加载的CLIP模型
    preprocess     : CLIP的预处理函数
    '''
    
    cached_img_keys = []
    cached_txt_keys = []
    cached_values   = []
    train_features  = []
    txt_features    = [] 

    train_img_path  = pd.read_csv(class_4_path, header=None)[0].tolist()                                               # 所有训练集图像的路径
    train_img_path  = list(map(lambda x: root_TrainSet + x, train_img_path))
    
    classes         = pd.read_csv(classes_path , delimiter = ' '  , header = None , index_col = False)                 # 所有类别名称    
    label_name      = list(map(lambda x: x.split('/')[-2], train_img_path))                                             # 训练集对应的类别名称     
    df              = {k: v for k, v in zip(list(map(lambda x:x.split('_',1)[-1], list(classes[0]))), list(classes[1]))} # 类别名称到类别ID的映射
    

    with jt.no_grad():
        for x,y in zip(train_img_path, label_name):
            img     = Image.open(x).convert('RGB') 
            img     = preprocess(img).unsqueeze(0)
            img_fea = model.encode_image(img)  
            img_fea /= img_fea.norm(dim = -1, keepdim = True)

            text_fea = model.encode_text(clip.tokenize(generate_prompt(y)))
            text_fea /= text_fea.norm(dim = -1, keepdim =True) 
            txt_features.append(text_fea)
            
            train_features.append(img_fea)
            cached_values.append(df[y])

        cached_img_keys   = jt.concat(train_features, dim = 0)                    #  [1496,512]
        cached_txt_keys   = jt.concat(txt_features, dim = 0)                    #  [1496,512]
        cached_values = nn.one_hot(jt.array(cached_values), num_classes=len(classes)).astype('float32') #  [1496,374]

    return cached_img_keys, cached_txt_keys, cached_values