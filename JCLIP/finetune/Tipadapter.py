import pandas as pd
from   PIL import Image
import jittor as jt
import jittor.nn as nn
from   loguru           import logger
import jclip as clip
from utils import get_date_format, normalize_tensor

def Tip_adapter(root_TrainSet : str, class_4_path : str , classes_path : str, model, preprocess , alpha :float= 2.4 , beta :float = 0.7 ):
    
    '''
    root_TrainSet  : 训练集的根目录
    alpha , beta   : 两个超参数
    class_4_path   : train_4class.txt文件路径
    classes_path   : classes.txt or classes_b.txt文件路径
    model          : 加载的CLIP模型
    preprocess     : CLIP的预处理函数
    return         : cached_keys --> [M,512] , cached_values --> [M,N]  M为训练集的样本数，N为类别数
    '''
    
    cached_keys    = []
    cached_values  = []
    train_features = []

    train_img_path  = pd.read_csv(class_4_path, header=None)[0].tolist()                                                     # 训练集所有图像的路径
    train_img_path  = list(map(lambda x: root_TrainSet + x, train_img_path))
    
    classes         = pd.read_csv(classes_path , delimiter = ' '  , header = None , index_col = False)                       # 所有类别名称    
    label_name      = list(map(lambda x: x.split('/')[-2], train_img_path))                                                  # 训练集对应的类别名称     
    df              = {k: v for k, v in zip(list(map(lambda x:x.split('_',1)[-1], list(classes[0]))),list(classes[1]))}      # 类别名称到类别ID的映射
    

    with jt.no_grad():
        for x,y in zip(train_img_path, label_name):
            img     = Image.open(x).convert('RGB')  
            img     = preprocess(img).unsqueeze(0)
            img_fea = model.encode_image(img)  
            img_fea /= img_fea.norm(dim = -1, keepdim = True)
            
            train_features.append(img_fea)
            cached_values.append(df[y])

        cached_keys   = jt.concat(train_features, dim = 0)                    #  [1496,512]
        cached_values = nn.one_hot(jt.array(cached_values), num_classes=len(classes)).astype('float32') #  [1496,374]

    return cached_keys , cached_values



def trainer_Tip_adapter_F(model, clip_model, train_loader, val_loader, scheduler, optimizer, EPOCHS, BETA, ALPHA, cache_values, text_features, save_path, ckt_gap):
    
   
    '''
    model        : 待训练的模型
    clip_model   : CLIP模型
    train_loader : 训练集的DataLoader
    val_loader   : 验证集的DataLoader
    scheduler    : 学习率衰减策略
    optimizer    : 优化器
    EPOCHS       : 训练的轮数
    ALPHA、BETA  : 超参数
    cache_values : 缓存模型的值
    text_features: clip_model在所有类别上生成的文本特征向量
    save_path    : 保存模型的路径
    ckt_gap      : 验证间隔
    '''
    
    best_acc   = 0
    for epoch in range(EPOCHS): 
        total_loss = 0   # 总损失
        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(total_loss) 
        else :
            scheduler.step() 
    
        # 开始训练 
        model.train()
        for images, label in train_loader:
            optimizer.zero_grad()
            
            with jt.no_grad():
                image_features  = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=1, keepdim=True)
            
            affinity      = model(image_features)
            cache_logits  = ((-1) * (BETA - BETA * affinity)).exp() @ cache_values    # --> [batch_size, 374]
            clip_logits   = (100.0 * image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
            # cache_logits = normalize_tensor(cache_logits)
            tip_logits    = clip_logits + cache_logits * ALPHA
            
            cur_loss     = nn.CrossEntropyLoss()(tip_logits, label)
            total_loss  += cur_loss
            optimizer.backward(cur_loss)   
            optimizer.step()
        
        logger.info('train epoch:{}     total_loss:{:.4f}'.format(epoch+1, total_loss))
    
        # 验证开始
        if (epoch+1) % ckt_gap == 0: 
            model.eval()
            val_acc  = 0 
            num      = 0
            with jt.no_grad():
                for images, label in val_loader:
                    model.eval()
                    image_features  = clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    affinity      = model(image_features)
                    cache_logits  = ((-1) * (BETA - BETA * affinity)).exp() @ cache_values
                    clip_logits   = (100.0 * image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
                    # cache_logits = normalize_tensor(cache_logits)
                    tip_logits    = (clip_logits + cache_logits * ALPHA).softmax(dim=-1)
                    
                    _, top_labels = tip_logits.topk(1)  
                    batch_acc = jt.sum(jt.equal(label.unsqueeze(1),top_labels)).tolist()[0]  
                    val_acc  += batch_acc 
              
            val_acc = val_acc / 3000
            logger.info('val epoch:{} acc:{:.4f}'.format(epoch+1, val_acc))   

            if val_acc > best_acc:
                best_acc = val_acc
                jt.save(model.state_dict(), save_path + f"Tip_adapter_F-{get_date_format()}.pkl")

