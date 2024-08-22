import os
import yaml
import jclip as  clip
import jittor as jt
import pandas as pd
import jittor.nn as nn
from   loguru  import logger
from   utils   import get_date_format

root_path = os.getenv('ROOT_PATH')
with open(f'{root_path}/JCLIP\configs\FD-Align.yaml', 'rb') as f:
    config         = yaml.safe_load(f)
    clip_init_path = config['pretrain_path']

jt.flags.use_cuda = 1
clip_init, preprocess = clip.load(clip_init_path)  # TAG

def prompt_templates(name):
    prompt = [f'a photo of a {name}.',
    f'a bad photo of a {name}.',
    f'a photo of many {name}.',
    f'a sculpture of a {name}.',
    f'a photo of the hard to see {name}.',
    f'a low resolution photo of the {name}.',
    f'a rendering of a {name}.',
    f'graffiti of a {name}.',
    f'a bad photo of the {name}.',
    f'a cropped photo of the {name}.',
    f'a tattoo of a {name}.',
    f'the embroidered {name}.',
    f'a photo of a hard to see {name}.',
    f'a bright photo of a {name}.',
    f'a photo of a clean {name}.',
    f'a photo of a dirty {name}.']
    
    return prompt


def Prototype(class_path):
    '''
    class_path : classes.txt or classes_b.txt文件路径
    return     : Class prototype  和 Prompt prototype
    '''
    df_classes   = pd.read_csv(class_path, delimiter = ' ', header = None, index_col = False)
    list_classes = list(map(lambda x: x.split('_',1)[-1], df_classes[0]))

    Text_Embedding = []
    for name in list_classes:
        prompt_group_name = prompt_templates(name)    # 某一个类别名的所有Prompt
        prompt_token_name = []                        # 经过tokenize后的Prompt
        for prompt in prompt_group_name:
            prompt_token_name.append(clip.tokenize(prompt))
        prompt_token_name = jt.concat(prompt_token_name, dim=0)                   # shape --> [M, 77], M为Prompt的数量
        txt_features      = clip_init.encode_text(prompt_token_name)              # shape --> [M,512]
        Text_Embedding.append(txt_features)    
    Text_Embedding   = jt.array(Text_Embedding)                                   # Text_Embedding.shape --> [N,M,512]  N为类别数量(374)
    class_prototype  = jt.mean(Text_Embedding, dim=1)                             # [N,512]
    class_prototype  = class_prototype / class_prototype.norm(dim=1, keepdim=True)
    prompt_prototype = jt.mean(Text_Embedding, dim=0)                             # [M,512]
    prompt_prototype = prompt_prototype / prompt_prototype.norm(dim=1, keepdim=True)

    return class_prototype, prompt_prototype
    
    


def trainer_FD_Align(model, clip_init, train_loader, val_loader, scheduler, optimizer, EPOCHS, ckt_gap, classes_path, save_path, class_prototype, prompt_prototype):

    '''
    model           : 待训练的模型
    clip_init       : 训练的CLIP模型
    train_loader    : 训练集的DataLoader
    val_loader      : 验证集的DataLoader
    scheduler       : 学习率衰减策略
    optimizer       : 优化器
    EPOCHS          : 训练的轮数
    ckt_gap         : 验证间隔
    classes_path    : classes.txt文件路径
    save_path       : 保存模型的路径
    class_prototype : 类原型
    prompt_prototype: 提示模板原型
    '''
    
    
    best_acc   = 0
    for epoch in range(EPOCHS): 
        # for param in model.transformer.parameters():
        #     param.requires_grad = False 

        total_loss = 0   # 总损失
        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(total_loss) 
        else :
            scheduler.step() 
    
        # 开始训练 
        model.train()
        for images, class_prototype_batch in train_loader:
            optimizer.zero_grad() 

            image_features  = model.encode_image(images)
            image_features /= image_features.norm(dim=1, keepdim=True)
            image_fea_0     = clip_init.encode_image(images)
            image_fea_0    /= image_fea_0.norm(dim=1, keepdim=True)
            ground_truth    = jt.arange(len(images), dtype='int64')
            loss_class      = nn.CrossEntropyLoss()(100.0 * image_features @ class_prototype_batch.t(), ground_truth) 
            
            P_spurious_0    = (image_fea_0    @  prompt_prototype.t()).softmax(dim=-1)
            P_spurious      = (image_features @  prompt_prototype.t()).softmax(dim=-1)
            loss_spurious   = nn.KLDivLoss()(jt.log(P_spurious), P_spurious_0)                # KL不具备对称性，顺序不能反

            cur_loss        = loss_class + 20 * loss_spurious 
            total_loss     += cur_loss 
            optimizer.backward(cur_loss)   
            optimizer.step()
        
        logger.info('train epoch:{}     total_loss:{:.4f}'.format(epoch+1, total_loss))

        # 验证开始
        if (epoch+1)%ckt_gap == 0: 
            model.eval()
            val_acc  = 0 
            num      = 0
            with jt.no_grad():
                for images, label in val_loader:
                    image_features  = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    text_probs      = (100.0 * image_features @ class_prototype.transpose(0, 1)).softmax(dim=-1)
                    _, top_labels   = text_probs.topk(1)  
                    batch_acc       = jt.sum(jt.equal(label.unsqueeze(1),top_labels)).tolist()[0]  
                    val_acc         += batch_acc  

            val_acc = val_acc / (val_loader.total_len)
            logger.info('val epoch:{} acc:{:.4f}'.format(epoch+1, val_acc))   

            if val_acc > best_acc:
                best_acc = val_acc
                jt.save(model.state_dict(), save_path + f"CLIP-FD_Align-{get_date_format()}.pkl")
