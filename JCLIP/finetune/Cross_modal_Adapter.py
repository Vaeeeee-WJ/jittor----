import jittor as jt
import jittor.nn as nn
import numpy as np
from   loguru    import logger
from utils import get_date_format, get_val_text_features



class LogitHead(nn.Module):
    def __init__(self, head, logit_scale=float(np.log(1 / 0.07))):
        super().__init__()
        self.head = head
        self.logit_scale = logit_scale
        
        # Not learnable for simplicity
        self.logit_scale = jt.array([logit_scale])
        # Learnable
        # self.logit_scale = jt.nn.Parameter(jt.ones([1]) * logit_scale)

    def execute(self, x):
        x = jt.normalize(x, dim=1)
        x = self.head(x)
        x = x * self.logit_scale.exp()
        return x

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.2):
        super(Adapter, self).__init__()
        self.residual_ratio = residual_ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU()
        )

    def execute(self, x):
        a = self.fc(x)
        x = self.residual_ratio * a + (1 - self.residual_ratio) * x
        return x
    


def logitHead(text_features, _in_features=512, _num_classes=403):
    adapter            = Adapter(_in_features, residual_ratio=0.2)
    linear_head        = nn.Linear(_in_features, _num_classes, bias=False)
    linear_head.weight = text_features                                         # 使用文本特征初始化线性层权重
    head               = nn.Sequential(adapter, linear_head)
    logithead          = LogitHead(head=head)
    # logithead          = LogitHead(head=linear_head) 
    return logithead


def trainer_cross_modal_adapter(model, train_loader, val_loader, scheduler, optimizer, EPOCHS, ckt_gap, classes_path, save_path):
    
    '''
    model        : 待训练的模型
    train_loader : 训练集的DataLoader
    val_loader   : 验证集的DataLoader
    scheduler    : 学习率衰减策略
    optimizer    : 优化器
    EPOCHS       : 训练的轮数
    ckt_gap      : 验证间隔
    classes_path : classes.txt or classes_b.txt文件路径
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
        for images, label_tokens, label_id in train_loader:
            optimizer.zero_grad()
            
            image_features  = model.encode_image(images)
            text_features   = model.encode_text(label_tokens)
            image_features /= image_features.norm(dim=1, keepdim=True)
            text_features  /=  text_features.norm(dim=1, keepdim=True)

            cross_logits  = model.cross_logit(jt.normalize(jt.concat([image_features, text_features], dim=0)))
            # cross_logits  = model.cross_logit(jt.concat([image_features, text_features], dim=0))

            labels      = jt.concat([label_id, label_id], dim=0).cast(jt.int64)

            cur_loss    = nn.CrossEntropyLoss()(cross_logits*3.0, labels) # ❗❗❗注意：不需要按照伪代码中将cross_logits除以一个常量，loss反而会很难下降,相反乘上一个系数loss下降的更好！
            total_loss += cur_loss 
            optimizer.backward(cur_loss)   
            optimizer.step()
        
        logger.info('train epoch:{}     total_loss:{:.4f}'.format(epoch+1,total_loss))

        # 验证开始
        if (epoch+1)%ckt_gap == 0: 
            model.eval()
            text_features = get_val_text_features(classes_path, model)
            val_acc  = 0 
            num      = 0
            with jt.no_grad():
                
                for images, label in val_loader:
                    image_features  = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_probs      = (100.0*model.cross_logit(image_features)).softmax(dim=-1)

                    _, top_labels = text_probs.topk(1)  
                    batch_acc = jt.sum(jt.equal(label.unsqueeze(1), top_labels)).tolist()[0]  
                    val_acc  += batch_acc 

            val_acc = val_acc / (val_loader.total_len)
            logger.info('val epoch:{} acc:{:.4f}'.format(epoch+1, val_acc))   

            if val_acc > best_acc:
                best_acc = val_acc
                jt.save(model.state_dict(), save_path + f"CLIP-cross_modal_adapter-{get_date_format()}.pkl")

