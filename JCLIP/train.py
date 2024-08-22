import os
import argparse
import random
import pandas as pd
from   PIL              import Image
import jittor           as     jt
from   jittor           import nn
from   jittor.dataset   import Dataset
from   jittor.dataset   import DataLoader
import jclip            as     clip
from   jittor           import optim
from   loguru           import logger
from   jittor_utils     import LOG
import numpy as np
import jittor.transform as  transforms
# from  utils import class_2_idx, generate_prompt, get_val_text_features, RandomErasing, normalize_tensor, format_args, get_date_format, load_yaml_params
from    utils import *





# 选择使用的优化器
def get_optimizer(args):

    if args.optimizer   == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.5, weight_decay=0.001)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate, alpha=0.9, eps=1e-8)
    elif args.optimizer ==   "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    else:
        raise ValueError("Invalid optimizer: {}".format(args.optimizer))
    return optimizer


# 选择使用的学习率策略
def get_scheduler(args):
    if args.scheduler   == "StepLR":
        return jt.lr_scheduler.StepLR(optimizer, step_size = 10, gamma  = 0.1)
    elif args.scheduler == "ReduceLROnPlateau":
        return jt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    elif args.scheduler == "CosineAnnealingLR":
        return jt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoches, eta_min=1e-7)
    else:
        raise ValueError("Invalid scheduler: {}".format(args.scheduler))


# 数据增强方法
def get_transform(args):
    if  args.augmentation :
        # 经尝试，作随机裁剪和随机擦除会有一定效果，其他增强手段反而会降低模型的性能
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3)), # 随机裁剪
            RandomErasing(p=0.5, scale=(0.002, 0.01), ratio=(0.3, 3.3), value=0),     # 随机擦除一个大小的矩形区域，像素置为0
            # transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
            # transforms.ColorJitter(hue=0.5),        # 色调
            # transforms.ColorJitter(contrast=0.5),   # 对比度
        ])
        return transform_train

    else :
        return None

# 参数设置 
root_path = os.getenv('ROOT_PATH')
parser    = argparse.ArgumentParser(description='CLIP微调')
parser.add_argument('--seed'            ,type = int,   default = 20                                ,help='随机数种子')
parser.add_argument('--root_TrainSet'   ,type = str,   default = root_path + '/'                   ,help='TrainSet数据集的根目录')
parser.add_argument('--train_path'      ,type = str,   default = f'{root_path}/train.txt'          ,help='train.txt文件路径')
parser.add_argument('--classes_path'    ,type = str,   default = f'{root_path}\classes.txt'        ,help='classes.txt文件的路径')
parser.add_argument('--classes_b_path'  ,type = str,   default = f'{root_path}\classes_b.txt'      ,help='classes_b.txt文件的路径')
parser.add_argument('--class_4_path'    ,type = str,   default = f'{root_path}/train_4class.txt'   ,help='存放有训练集图像路径的txt文件')
parser.add_argument('--pretrain_path'   ,type = str,   default = None                              ,help='预训练权重路径')
parser.add_argument('--TestSetZ_path'   ,type = str,   default = f'{root_path}\TestSetZ'           ,help='验证集的路径(自己划分的验证集TestSetZ,标签放在TestSetZ-label.txt文件中,比赛未给出)')
parser.add_argument('--label_path'      ,type = str,   default = f'{root_path}\TestSetZ-label.txt' ,help='TestSetZ-label.txt文件的路径')
parser.add_argument('--save_path'       ,type = str,   default = f'{root_path}\Weights/'           ,help='模型权重保存路径')
parser.add_argument('--epoches'         ,type = int,   default = 200                               ,help='训练的epoch数')
parser.add_argument('--augmentation'    ,type = str,   default = False                             ,help='是否采用数据增强')
parser.add_argument('--ckt_gap'         ,type = int,   default = 10                                ,help='每训练多少次保存一次')
parser.add_argument('--learning_rate'   ,type = int,   default = 1e-5                              ,help='学习率大小')
parser.add_argument('--alpha'           ,type = float, default = 3.0                               ,help='Tip-Adapter-F的超参数')
parser.add_argument('--beta'            ,type = float, default = 0.8                               ,help='Tip-Adapter-F的超参数')
parser.add_argument('--batch_size'      ,type = int,   default = 187                               ,help='训练的batchsize  (❗❗❗注意：CLIP训练时需要保证每次batch中的类别都不相同,后续代码逻辑原因，这里batchsize必须设置成能被 374 整除的数)')
parser.add_argument("--optimizer"       ,type = str,   default="AdamW", choices=["SGD", "RMSprop", "AdamW"]                          , help="优化器类型")
parser.add_argument("--scheduler"       ,type = str,   default="StepLR", choices=["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"], help="学习率策略")
parser.add_argument('--finetune'        ,type = str,   choices=['no_finetune', 'no_finetune_v1', 'Tip-Adapter-F', 'cross-modal-Adapter', 'FD-Align', 'Coop'], help='微调模型的方法')
args = parser.parse_args()



# 定义训练集和验证集
## 训练集：每个类别最多选择 4 张图片
## 验证集：从train.txt中随机选择的 3000 张图片

transform_train = None # 默认不使用数据增强
class MyDataset(Dataset):

    def __init__(self, mode, is_shuffle=False, transform=transform_train, is_exist_train=None, is_save=False):

        '''
        mode          : 训练集 or 验证集
        is_shuffle    : 是否打乱训练集(False: 1496张图片中0~373,374~747,748~1121,1122~1495的类别顺序和class.txt中顺序相同; True:每隔374张图片中类别顺序随机排列但不重复)
        transform     : 数据增强
        is_exist_train: 可以传入存有图片路径的txt文件路径，则直接加载文件中的图片路径作为训练集; 否则默认为None, 随机从训练集中选择4张图片作为训练集。
        is_save       : 是否将训练集的所有图片路径保存下来，方便后续使用，仅在is_exist_train=None情况下有效
        '''
        
        super().__init__()
        self.transform      = transform  
        self.mode           = mode           
        self.is_shuffle     = is_shuffle    
        self.is_exist_train = is_exist_train
        self.is_save        = is_save
        self.testZ          = pd.read_csv(args.label_path , delimiter = '\t' , header = None , index_col = False)
        
        if  self.mode == 'train':
            self.imgs  =  self.get_data()
            LOG.i(f"Found {374} classes and {len(self.imgs)} images in Training Data.") 
            self.set_attrs(total_len = len(self.imgs))
            
        elif self.mode == 'val':
            LOG.i(f"Found {len(self.testZ)} images in TestSetZ.")
            self.set_attrs(total_len = len(self.testZ))
            
        else:
            print('Error: mode should be train or val')
    
    def get_id_path(self , id:int) -> list:  # 返回指定类别的所有图片的路径
        
        img_paths     = pd.read_csv(args.train_path , delimiter = ' ' , header = None , index_col = False)
        path          = list(img_paths[0])
        label         = list(img_paths[1])
        filtered_list = [x for x, y in zip(path, label) if y == id]
        return filtered_list
    
    def four_img_of_train(self) -> list:  # 从TrainSet中每个类别随机选择4张图片，返回这些图片路径
        
        clasees = [x for x in range(374)]  
        paths   = []
        for x in clasees:
            path  =  self.get_id_path(x)
            paths += random.sample(path, k=   4) 
        return paths

    def get_data(self):    
        if self.is_exist_train:
            Train_paths = pd.read_csv(self.is_exist_train, header=None, index_col=None)[0].tolist()  # TAG 加载指定的训练集路径
            if 'no_finetune' in args.finetune or 'FD-Align' in args.finetune: 
                Train_paths = Train_paths + Train_paths
            else:
                Train_paths = Train_paths

            if self.is_shuffle:
                res = []
                for i in range(4):      
                    x = Train_paths[i::4]
                    random.shuffle(x)
                    res += x
                return res
            else:
                return Train_paths
            
        else:
            Train_paths = []
            for i in range(4):   
                if self.is_shuffle:
                    x = self.four_img_of_train()[i::4]
                    random.shuffle(x)
                    Train_paths += x
                else:
                    Train_paths += self.four_img_of_train()[i::4]  

            if self.is_save :
                TRAIN_PATHS  = list(map(lambda x: root_path + '/' + x, Train_paths))
                pd.DataFrame(TRAIN_PATHS).to_csv(f'train_4class.txt', header=None, index=None, sep='\n')
            
            return Train_paths
    
    def __getitem__(self, k):
        
        if self.mode == 'train':
            with open(args.root_TrainSet + self.imgs[k], 'rb') as f:
                img = Image.open(f).convert('RGB')
                if self.transform :
                    if k < 1496 and len(self.imgs) > 1496:
                        img = img
                    else:
                        img = self.transform(img)
                    
                img      = preprocess(img)
                # img   = Image_Transform(img, net).choose_best_img()  
                img_name = self.imgs[k].split('/')[-2]
                txt      = generate_prompt(img_name)

                if args.finetune == 'no_finetune':
                    return img, jt.flatten(clip.tokenize(txt))
                
                elif args.finetune == 'no_finetune_v1':
                    return img, jt.flatten(clip.tokenize(txt))

                elif args.finetune == 'Tip-Adapter-F':
                    return img, class_2_idx_dict[img_name]
                
                elif args.finetune == 'cross-modal-Adapter':
                    return img, jt.flatten(clip.tokenize(txt)), class_2_idx_dict[img_name]  
                
                elif args.finetune == 'FD-Align':
                    return img, class_prototype[class_2_idx_dict[img_name]]  
                
                elif args.finetune == 'Coop':
                    return img, class_2_idx_dict[img_name]  

        elif self.mode == 'val':

            with open(args.TestSetZ_path + '/' + list(self.testZ[0])[k],'rb') as f:

                img = Image.open(f).convert('RGB')
                img = preprocess(img)
                # img = Image_Transform(img, net).choose_best_img()
        
                return img, self.testZ[1][k]


def trainer_nofinetune(model, train_loader, val_loader, scheduler, optimizer, EPOCHS, classes_path, ckt_gap, save_path):

    '''
    model        : 待训练的模型
    train_loader : 训练集的DataLoader
    val_loader   : 验证集的DataLoader
    scheduler    : 学习率衰减策略
    optimizer    : 优化器
    EPOCHS       : 训练的轮数
    classes_path : classes.txt | classes_b.txt 文件路径
    ckt_gap      : 验证间隔
    save_path    : 保存模型的路径
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
        for images, label_tokens in train_loader:
            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, label_tokens)
            ground_truth = jt.arange(len(images), dtype='int64')
            cur_loss     = (nn.CrossEntropyLoss()(logits_per_image, ground_truth) + nn.CrossEntropyLoss()(logits_per_text, ground_truth))/2
            total_loss  += cur_loss
            optimizer.backward(cur_loss)   
            optimizer.step()
        
        logger.info('train epoch:{}     total_loss:{:.4f}'.format(epoch+1, total_loss))

        # 验证开始
        if (epoch+1)%ckt_gap == 0: 
            model.eval()
            val_acc  = 0 
            num      = 0
            with jt.no_grad():
                text_features = get_val_text_features(classes_path, model)
                
                for images, label in val_loader:
                    image_features  =  model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_probs      = (100.0 * image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
                    
                    _, top_labels = text_probs.topk(1)  
                    batch_acc = jt.sum(jt.equal(label.unsqueeze(1), top_labels)).tolist()[0]  # 每一批中正确的个数
                    val_acc  += batch_acc 

            val_acc = val_acc / (val_loader.total_len) 
            logger.info('val epoch:{} acc:{:.4f}'.format(epoch+1, val_acc))   

            if val_acc > best_acc:
                best_acc = val_acc
                jt.save(model.state_dict(), save_path + f"CLIP-{get_date_format()}.pkl")   # TAG



def trainer_nofinetune_v1(model, train_loader, val_loader, scheduler, optimizer, EPOCHS, classes_path, ckt_gap, save_path):
    '''
    
    相比于trainer_nofinetune改进的地方：
    将每个类别的4张图像经过image_encoder编码后的特征向量([1,512])多次求平均，作为该类新的特征向量, 防止过拟合，以此提升模型的泛化能力。
    
    model        : 待训练的模型
    train_loader : 训练集的DataLoader
    val_loader   : 验证集的DataLoader
    scheduler    : 学习率衰减策略
    optimizer    : 优化器
    EPOCHS       : 训练的轮数
    classes_path : classes.txt or classes_b.txt文件路径
    ckt_gap      : 验证间隔
    save_path    : 保存模型的路径
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

        IMAGE_FEATURES_1, IMAGE_FEATURES_2 = 0, 0
        N = 1
        for images, label_tokens in train_loader:
            optimizer.zero_grad()
            logits_per_image, logits_per_text, text_features, image_features = model(images, label_tokens)

            if N %2 != 0:
                IMAGE_FEATURES_1 += image_features
                IMAGE_FEATURES_avg = IMAGE_FEATURES_1 / ((N+1) // 2)

            else:
                IMAGE_FEATURES_2 += image_features
                IMAGE_FEATURES_avg = IMAGE_FEATURES_2 / ((N+1) // 2)


            N += 1

            ground_truth = jt.arange(len(images), dtype='int64')
            cur_loss     = (nn.CrossEntropyLoss()(logits_per_image, ground_truth) + nn.CrossEntropyLoss()(logits_per_text, ground_truth))/2

            logits_avg   = IMAGE_FEATURES_avg @ text_features.t()
            avg_loss     = (nn.CrossEntropyLoss()(logits_avg, ground_truth) + nn.CrossEntropyLoss()(logits_avg, ground_truth))/2

            cur_loss = cur_loss + avg_loss
        
            total_loss  += cur_loss
            optimizer.backward(cur_loss)   
            optimizer.step()
        
        logger.info('train epoch:{}     total_loss:{:.4f}'.format(epoch+1, total_loss))

        # 验证开始
        if (epoch+1)%ckt_gap == 0: 
            model.eval()
            val_acc  = 0 
            num      = 0
            with jt.no_grad():
                text_features = get_val_text_features(classes_path, model)
                
                for images, label in val_loader:
                    image_features  =  model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_probs      = (100.0 * image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
                    
                    _, top_labels = text_probs.topk(1)  
                    batch_acc = jt.sum(jt.equal(label.unsqueeze(1), top_labels)).tolist()[0]  # 每一批中正确的个数
                    val_acc  += batch_acc 

            val_acc = val_acc / (val_loader.total_len)  # TAG
            logger.info('val epoch:{} acc:{:.4f}'.format(epoch+1, val_acc))   

            if val_acc > best_acc:
                best_acc = val_acc
                jt.save(model.state_dict(), save_path + f"CLIP-{get_date_format()}-v1.pkl")   # TAG
            
                
                                                            
if __name__   == '__main__':  

    # 配置日志文件 | 设置训练设备 | 设置随机数种子
    logger.add("./Training.log",rotation="500MB", encoding="utf-8", enqueue=True, retention="10 days")
    jt.flags.use_cuda =  1   
    random.seed(args.seed) 
    jt.misc.set_global_seed(args.seed)
    
    args.finetune   = os.getenv('METHOD_NAME')    # TAG 设置微调方法 ['no_finetune', 'no_finetune_v1', 'Tip-Adapter-F', 'cross-modal-Adapter', 'FD-Align', 'Coop']
    load_yaml_params(f'{root_path}\JCLIP\configs/{args.finetune}.yaml', args) 
    net, preprocess = clip.load(args.pretrain_path)
    
    # 加载优化器 | 学习率调整策略 | 数据增强
    optimizer       = get_optimizer(args)
    scheduler       = get_scheduler(args)
    transform_train = get_transform(args)
      
    logger.info(f"本次训练采用的微调方法为：{args.finetune}" if args.finetune != 'no_finetune' else "本次训练未使用微调方法")
    logger.info(f"Prompt :  {generate_prompt('XXX')}")
    
    # 加载数据集
    train_data   = MyDataset(mode='train', is_shuffle=False, transform=transform_train, is_save=False, is_exist_train=args.class_4_path)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

    val_data     = MyDataset(mode='val')
    val_loader   = DataLoader(val_data, batch_size=100, shuffle=False)
    

    # 记录训练参数
    formatted_args = format_args(args)
    logger.info(f"本次训练的参数设置为：\n{formatted_args}")  
    if args.augmentation :
        logger.info(f"使用的数据增强方法：{transform_train.transforms}") 
    logger.info('------开始训练!------')
    
    
    
    if args.finetune == 'no_finetune':
        trainer_nofinetune(model=net,
                           train_loader=train_loader, val_loader=val_loader,
                           scheduler=scheduler, optimizer=optimizer, EPOCHS=args.epoches,
                           classes_path=args.classes_b_path, ckt_gap=args.ckt_gap, save_path=args.save_path)

    elif args.finetune == 'no_finetune_v1':
        trainer_nofinetune_v1(model=net,
                            train_loader=train_loader, val_loader=val_loader,
                            scheduler=scheduler, optimizer=optimizer, EPOCHS=args.epoches,
                            classes_path=args.classes_b_path, ckt_gap=args.ckt_gap, save_path=args.save_path)

    elif args.finetune == 'Tip-Adapter-F':
        from finetune.Tipadapter import Tip_adapter, trainer_Tip_adapter_F
        class_2_idx_dict          = class_2_idx(args.classes_path)
        cache_keys , cache_values = Tip_adapter(root_TrainSet=args.root_TrainSet, 
                                                class_4_path=args.class_4_path,
                                                classes_path=args.classes_path, 
                                                model = net, preprocess=preprocess)
        
        adapter                   = nn.Linear(cache_keys.shape[1], cache_keys.shape[0], bias=False)
        adapter.weight            = nn.Parameter(cache_keys) 
        text_features             = get_val_text_features(args.classes_path, net)  
        
        # tip这里需要训练的是adapter，而不是net
        if args.optimizer == "SGD":
            optimizer = optim.SGD(adapter.parameters(), lr=args.learning_rate, momentum=0.5, weight_decay=0.001)
        elif args.optimizer == "RMSprop":
            optimizer = optim.RMSprop(adapter.parameters(), lr=args.learning_rate, alpha=0.9, eps=1e-8)
        elif args.optimizer == "AdamW":
            optimizer = optim.AdamW(adapter.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)

        trainer_Tip_adapter_F(model=adapter, clip_model=net,
                            train_loader=train_loader, val_loader=val_loader,
                            scheduler=scheduler, optimizer=optimizer, EPOCHS=args.epoches,
                            ALPHA=args.alpha, BETA=args.beta, 
                            cache_values=cache_values, text_features=text_features, 
                            save_path=args.save_path, ckt_gap=args.ckt_gap)

    elif args.finetune == 'cross-modal-Adapter': 
        from  finetune.Cross_modal_Adapter import logitHead, trainer_cross_modal_adapter
        class_2_idx_dict   = class_2_idx(args.classes_b_path)
        net.add_module('cross_logit', logitHead(get_val_text_features(args.classes_b_path, net), _in_features=512, _num_classes=len(class_2_idx_dict)))
        
        trainer_cross_modal_adapter(model=net,
                                    train_loader=train_loader, val_loader=val_loader,
                                    scheduler=scheduler, optimizer=optimizer, EPOCHS=args.epoches,ckt_gap=args.ckt_gap,
                                    classes_path=args.classes_b_path, save_path=args.save_path)
        
      
    elif args.finetune == 'FD-Align':   
        from finetune.FD_Align import Prototype, clip_init, trainer_FD_Align
        class_2_idx_dict = class_2_idx(args.classes_b_path)
        class_prototype, prompt_prototype= Prototype(args.classes_b_path)

        trainer_FD_Align(model=net, clip_init= clip_init,
                                    train_loader=train_loader, val_loader=val_loader,
                                    scheduler=scheduler, optimizer=optimizer, EPOCHS=args.epoches, ckt_gap=args.ckt_gap,
                                    classes_path=args.classes_b_path, save_path=args.save_path,
                                    class_prototype=class_prototype, prompt_prototype=prompt_prototype)

    elif args.finetune == 'Coop':
        from finetune.Coop import CustomCLIP, trainer_Coop
        class_2_idx_dict = class_2_idx(args.classes_b_path)
        net = CustomCLIP(net, list(class_2_idx_dict.keys()))

        trainer_Coop(model=net,
                           train_loader=train_loader, val_loader=val_loader,
                           scheduler=scheduler, optimizer=optimizer, EPOCHS=args.epoches,
                           classes_path=args.classes_b_path, save_path=args.save_path)

    logger.info('------训练结束!------')
                            