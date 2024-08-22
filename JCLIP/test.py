import jittor as jt
import jclip as clip
import jittor.nn as nn
import os
from PIL import Image
from tqdm import tqdm
import argparse
import random
import pandas as pd
import numpy as np
from finetune.Tipadapter   import  Tip_adapter
from finetune.cache_module import cache_module
from sklearn.linear_model import LogisticRegression
from utils import generate_prompt, normalize_tensor, count_parameters_in_mb, get_val_text_features, get_date_format
from sklearn.multiclass import OneVsRestClassifier


random.seed(20) 
jt.misc.set_global_seed(20)

def compute_acc_TestSetZ(root_TrainSet, pkl_path, FD_Align_pkl_path, Tip_Adapter_F_pkl_path, cross_modal_pkl_path, classes_path , TestData_path , TestSetZ_label_path, ALPHA =  2.4, BETA = 0.7 ,method_name = None):
    
    """
    root_TrainSet          : 训练集的根目录
    pkl_path               : clip模型权重的路径
    FD_Align_pkl_path      : FD_Align方法训练得到的权重路径
    Tip_Adapter_F_pkl_path : Tip-Adapter-F方法训练得到的权重路径
    cross_modal_pkl_path   : cross_modal_adapter方法训练得到的权重路径
    classes_path           : classes.txt文件的路径
    TestData_path          : 测试数据集的路径     (TestSetZ or TestSetA or TestSetB)
    TestSetZ_label_path    : TestSetZ-label.txt文件路径
    method_name            : 使用的微调方法，默认为None,其余可选择的有 ['Tip-Adapter', 'Tip-Adapter-F', 'Linear_Probe', 'cross_modal_Adapter', 'cross_modal+tip_adapter', 'FD-Align', 'WiSE-FT', 'fusion', 'fusion_2']
    ALPHA , BETA           : Tip-Adapter微调方法中的超参数,默认为(2.4,0.7)
    """
    
    model, preprocess = clip.load(pkl_path)                            # 加载训练过的clip模型

    text_features  = get_val_text_features(classes_path, model=model)  # 类别的文本特征

    num         = 0 # 预测正确的图片数量

    class_4_path= f'{root_TrainSet}/train_4class.txt' 
    df_label    = pd.read_csv(TestSetZ_label_path, delimiter = '\t',encoding ='utf-8',header = None)     # TestSetZ 的标签，用于计算准确率
    df_result   = pd.DataFrame(columns=['img_name', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5'])   # 用于保存在TestSetA上预测的结果
    result_name = f'../results/result{get_date_format()}.txt'                                                       # 保存结果的文件名

    print(f'Test Dataset is: {TestData_path[-9:-1]}')
    imgs = os.listdir(TestData_path)

    def compute_probs(path_of_img :str, model = model, preprocess = preprocess):    
        '''
        path_of_img   : 输入的图片路径
        return        : 两个返回值：经过CLIP模型输出的预测概率 以及 经过image_encoder输出的图片特征 
        '''
        image           = Image.open(path_of_img).convert('RGB')
        image           = preprocess(image).unsqueeze(0)
        # image   = Image_Transform(image, clip_init).choose_best_img().unsqueeze(0)
        image_features  = model.encode_image(image)
        image_features  /= image_features.norm(dim=-1, keepdim=True)
 
        return image_features   

    # 获取所有测试图像的特征
    def get_test_fea(model=model, preprocess=preprocess):
        test_features = []             
        print('loading test data...')
        with jt.no_grad():
            for img in  tqdm(imgs):
                image_features = compute_probs(TestData_path + img, model, preprocess)
                test_features.append(image_features)
        test_features = jt.cat(test_features)
        return test_features
    
    # 获取所有测试图像的标签
    def get_test_label():
        test_label  = df_label.set_index(0).reindex(imgs)[1].tolist() 
        return test_label
    
    test_label  = get_test_label()  
    
    #------------------------------不使用其他微调方法,直接测试------------------------------
    if method_name == None:

        def test(img_fea):
            '''
            img_fea : 图像特征
            return : 返回预测的Top5的类别
            '''
            test_probs    = (100.0 * img_fea @ text_features.transpose(0, 1)).softmax(dim=-1)
            _, top_labels = test_probs.topk(5)          # top5 predictions
            return top_labels

        if 'TestSetA' in TestData_path or 'TestSetB' in TestData_path:
            top5_result = test(get_test_fea())

            df_result['img_name'] = imgs
            df_result[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']] = top5_result
            df_result.to_csv(result_name, sep=' ', index=False, header=False)  # 转成txt结果文件
            return num
        
        else:
            top5_result = test(get_test_fea())
            top1_result = top5_result[:,0]
            num = jt.sum(jt.equal(test_label, top1_result))
            print(f"共{len(df_label)}张图片，预测完成！")
            return num
        
    #------------------------------调用其他finetune方法------------------------------
    #———————————— Tip-Adapter ————————————#
    elif method_name == 'Tip-Adapter':
        
        print(f'Utilized the {method_name} method!')
        cached_keys , cached_values= Tip_adapter(root_TrainSet, class_4_path, classes_path , model , preprocess)

        def tip_test(test_features, alpha=ALPHA, beta=BETA):
            cache_logits = ((-1) * (beta - beta* test_features @ cached_keys.t())).exp() @ cached_values # [3000,374]
            cache_logits = normalize_tensor(cache_logits)
            # cache_logits /= cache_logits.norm(dim=1, keepdim=True)

            text_probs    = (100.0 * test_features @ text_features.transpose(0, 1)).softmax(dim=-1)
            logits        = alpha * cache_logits + text_probs
            # logits = alpha * cache_logits * text_probs

            _, top_labels  = logits.topk(5)
            return top_labels  # 返回预测Top5的类别 [3000,5]
  
        if 'TestSetA' in TestData_path or 'TestSetB' in TestData_path:
            test_features = get_test_fea()
            top5_result   = tip_test(test_features, alpha=2.0, beta=3.0)
            df_result['img_name'] = imgs
            df_result[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']] = top5_result
            df_result.to_csv(result_name, sep=' ', index=False, header=False)  # 转成txt结果文件
            return num
        
        else:
            best_acc = 0
            ALPHA = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6]
            BETA  = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1] 
            test_features = get_test_fea()
            for alpha in ALPHA:
                for beta in BETA:
                    top5_result = tip_test(test_features, alpha, beta)
                    top1_result = top5_result[:,0]
                    num = jt.sum(jt.equal(test_label, top1_result))
                    acc = num/3000
                    if round(float(acc), 4) >  best_acc:
                        best_acc   = round(float(acc), 4)
                        ALPHA_best = alpha
                        BETA_best  = beta
                        print(f'The best acc is {best_acc}, alpha is {ALPHA_best}, beta is {BETA_best}')
            return num

    #———————————— Tip-Adapter-F:经过训练的模型 ————————————#
    elif method_name == 'Tip-Adapter-F': 

        print(f'Utilized the {method_name} method!')
        cached_keys , cached_values= Tip_adapter(root_TrainSet, class_4_path , classes_path , model , preprocess)

        # 加载训练的Tip-adapater-F
        adapter                   = nn.Linear(cached_keys.shape[1], cached_keys.shape[0], bias=False)
        adapter.load_state_dict(jt.load(Tip_Adapter_F_pkl_path))

        def tip_F_test(test_features, alpha=0.4, beta=5.1): 
            affinity     = adapter(test_features)
            cache_logits = ((-1) * (beta - beta*  affinity)).exp() @ cached_values # [3000,374]
            # cache_logits = normalize_tensor(cache_logits)
            # cache_logits /= cache_logits.norm(dim=1, keepdim=True)

            text_probs    = (100.0 * test_features @ text_features.transpose(0, 1)).softmax(dim=-1)
            logits        = alpha * cache_logits + text_probs
            # logits = alpha * cache_logits * text_probs

            _, top_labels  = logits.topk(5)
            return top_labels  # 返回预测Top5的类别 [3000,5]
  
        if 'TestSetA' in TestData_path or 'TestSetB' in TestData_path:
            test_features = get_test_fea()
            top5_result = tip_F_test(test_features, alpha=0.22, beta=4.5)
            df_result['img_name'] = imgs
            df_result[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']] = top5_result
            df_result.to_csv(result_name, sep=' ',  index=False, header=False)  # 转成txt结果文件
            return num
        
        else:
                 
            best_acc = 0
            ALPHA = [0.1,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6]
            BETA  = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1] 
            test_features = get_test_fea()
            for alpha in ALPHA:
                for beta in BETA:
                    
                    top5_result = tip_F_test(test_features, alpha, beta)
                    top1_result = top5_result[:,0]
                    test_label  = get_test_label()
                    num = jt.sum(jt.equal(test_label, top1_result))
                    acc = num/3000
                    if round(float(acc), 4) >  best_acc:
                        best_acc = round(float(acc), 4)
                        ALPHA_best = alpha
                        BETA_best  = beta
                        print(f'The best acc is {best_acc}, alpha is {ALPHA_best}, beta is {BETA_best}')
            return num

        
    #———————————— Linear Probe:线性分类头，冻住主干，仅微调一个线性分类器 ————————————#
    # 📌在TestSetZ上的效果不好,所以并未在TestSetA or B 上进行测试
    elif method_name == 'Linear_Probe':
        
        print(f'Utilized the {method_name} method!')
        
        # 注意：因为要简单训练一下分类器，正好导入tip_adapter中的缓存模型，而不是使用Tip-adapter方法
        train_features , train_labels = Tip_adapter(root_TrainSet, class_4_path , classes_path , model , preprocess)
        train_features                = train_features.numpy()
        train_labels                  = jt.argmax(train_labels, dim=1)[0].numpy().astype('float32')

        if 'TestSetA' in TestData_path:
            ...
        else:
            # 训练分类器
            classifier = LogisticRegression(random_state=0,
                                            C=8.960,
                                            max_iter=3000,
                                            verbose=1)
            classifier.fit(train_features, train_labels)
            
            # 加载所有测试数据
            test_features = []             # 存储所有的测试图片特征
            test_label    = []             # 存储所有的测试图片标签
            print('loading test data...')
            with jt.no_grad():
                for img in  tqdm(imgs):
                    image_features = compute_probs(TestData_path + img)
                    test_features.append(image_features)
                    test_label.append(int(df_label.loc[df_label[0] == img, 1].values[0]))
                    
            test_features = jt.cat(test_features).numpy()
            
            print('start predicting...')
            predictions = classifier.predict_proba(test_features)
            for prediction , label  in zip(predictions , test_label):
                prediction = np.asarray(prediction)
                top5_idx = prediction.argsort()[-1:-6:-1]  # 取前5个预测结果
                output = top5_idx[0]
                if output == label:
                    num += 1    
            return num
    
    elif method_name == 'cross_modal_Adapter':

        print(f'Utilized the {method_name} method!')
        from finetune.Cross_modal_Adapter import logitHead

        model.add_module('cross_logit', logitHead(get_val_text_features(classes_path, model)))
        model.load_state_dict(jt.load(pkl_path))

        model1, preprocess1 = clip.load(cross_modal_pkl_path)

        test_features_clip = get_test_fea(model1, preprocess=preprocess1)
        text_features  = get_val_text_features(classes_path, model=model1) # 类别的文本特征

        def cross_modal_Adapter_test(test_features, alpha=ALPHA, beta=BETA):

            logits1 =(100.0*model.cross_logit(test_features)).softmax(dim=-1)
            logits2 = (100.0 * test_features_clip @ text_features.transpose(0, 1)).softmax(dim=-1)

            logits = alpha * logits2 + logits1*beta

            _, top_labels  = logits.topk(5)
            return top_labels  # 返回预测Top5的类别 
        
        if 'TestSetA' in TestData_path or 'TestSetB' in TestData_path:
            test_features = get_test_fea()
            top5_result = cross_modal_Adapter_test(test_features, alpha=ALPHA, beta=BETA)
            df_result['img_name'] = imgs
            df_result[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']] = top5_result
            df_result.to_csv(result_name, sep=' ', index=False, header=False)  # 转成txt结果文件
            return num
        
        else:
                         
            best_acc = 0
            ALPHA = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6]
            BETA  = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1] 
            test_features = get_test_fea()
            for alpha in ALPHA:
                for beta in BETA:
                    
                    top5_result = cross_modal_Adapter_test(test_features, alpha, beta)
                    top1_result = top5_result[:,0]
                    num = jt.sum(jt.equal(test_label, top1_result))
                    acc = num/3000
                    if round(float(acc), 4) >  best_acc:
                        best_acc = round(float(acc), 4)
                        ALPHA_best = alpha
                        BETA_best  = beta
                        print(f'The best acc is {best_acc}, alpha is {ALPHA_best}, beta is {BETA_best}')
            return num

    #———————————— cross_modal+tip_adapter : 两种微调方法的结合，即跨模态训练+缓存模型微调————————————#
    elif method_name == 'cross_modal+tip_adapter':
 
        print(f'Utilized the {method_name} method!')
        
        train_img_features, train_txt_features,  train_labels = cache_module(root_TrainSet, class_4_path , classes_path , model , preprocess)
        train_img_features, train_txt_features                = train_img_features.numpy(), train_txt_features.numpy()
        # train_labels                                          = jt.argmax(train_labels, dim=1)[0].numpy().astype('float32')
        train_features = np.concatenate((train_img_features, train_txt_features), axis=0)
        train_labels   = np.concatenate((train_labels, train_labels), axis=0)
        # 训练分类器
        classifier = OneVsRestClassifier(LogisticRegression(random_state=0,
                                        C=8.960,
                                        max_iter=6000,
                                        verbose=1))
        classifier.fit(train_features, train_labels)
        
        # 加载所有测试数据
        print('loading test data...')
        test_features = get_test_fea().numpy()
        
        def cmta_test(alpha=0.5, beta=0.1):    # The best acc is 0.7547, alpha is 1.4, beta is 0.9, gamma is 0.1
            # print('start predicting...')
            predictions = classifier.predict_proba(test_features)
            pre_similitys = (100.0 * jt.array(test_features) @ text_features.transpose(0, 1)).softmax(dim=-1)

            x = predictions
            y = pre_similitys.numpy()

            logits = alpha*x + beta*y
            # logits = 0.8*x*  0.23*y 
            top5_result = np.flip(np.argsort(logits, axis=1)[:, -5:], axis=1) # np.array():包含整个数据集top5的预测结果
            return top5_result

        if 'TestSetA' in TestData_path or 'TestSetB' in TestData_path:
            
            top5_result = cmta_test(alpha=7.0, beta=1.2)
            df_result[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']] = top5_result
            df_result['img_name'] = imgs
            df_result.to_csv(result_name, sep=' ', index=False, header=False) 
            
        else:
            best_acc = 0
            ALPHA = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6,6.8,7.0,7.2,7.4,7.6,7.8,8.0,8.2,8.4,8.6,8.8,9.0,9.2,9.4,9.6,9.8,10.0,10.2,10.4,10.6,10.8,11.0,11.2,11.4,11.6,11.8,12.0,12.2,12.4,12.6,12.8,13.0,13.2,13.4,13.6,13.8,14.0,14.2,14.4,14.6,14.8,15.0,15.2]
            BETA  = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0] 
            for alpha in ALPHA:
                for beta in BETA:
                    top5_result = cmta_test(alpha, beta)
                    top1_result = top5_result[:,0]
                    num = jt.sum(jt.equal(np.asarray(test_label), top1_result))
                    acc = num/3000
                    if round(float(acc), 4) >  best_acc:
                        best_acc = round(float(acc), 4)
                        ALPHA_best = alpha
                        BETA_best  = beta
                        print(f'The best acc is {best_acc}, alpha is {ALPHA_best}, beta is {BETA_best}')

            # top1_result = cmta_test()[:, 0]
            # num = jt.sum(jt.equal(np.asarray(test_label), top1_result))
            return num
    

    elif method_name == 'FD-Align':

        print(f'Utilized the {method_name} method!')
        from finetune.FD_Align import Prototype
        class_prototype, prompt_prototype= Prototype(classes_path)
        
        def FD_Align_test(img_fea):
            '''
            img_fea : 测试图像特征
            return  : 返回预测的Top5类别
            '''
            text_probs = (100.0 * img_fea @ class_prototype.transpose(0, 1)).softmax(dim=-1)
            _, top_labels = text_probs.topk(5)
            return top_labels  
        
        if 'TestSetA' in TestData_path or 'TestSetB' in TestData_path:
            top5_result = FD_Align_test(get_test_fea())
            df_result['img_name'] = imgs
            df_result[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']] = top5_result
            df_result.to_csv(result_name, sep=' ', index=False, header=False) 
            return num
        else:
            top5_result = FD_Align_test(get_test_fea())
            top1_result = top5_result[:,0]
            num = jt.sum(jt.equal(test_label, top1_result))
            print(f"共{len(df_label)}张图片，预测完成！")
            return num
    
    elif method_name == 'WiSE-FT':

        print(f'Utilized the {method_name} method!')
        from jclip.model import build_model
        from jclip.clip  import _transform
        
        def wise_test(alpha):
    
            added_dict    = { key: alpha* weights_1.get(key, 0) + (1-alpha) * weights_2.get(key, 0) for key in set(weights_1) | set(weights_2)}
            model_12      = build_model(added_dict)
            preprocess_12 = _transform(model_12.visual.input_resolution)
            model_12.eval()
            text_features = get_val_text_features(classes_path, model_12) 
            image_features= get_test_fea(model=model_12, preprocess=preprocess_12)
            test_probs    = (100.0 * image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
            _, top_labels = test_probs.topk(5)          # top5 predictions
            return top_labels

        
        model_1, preprocess_1 = clip.load(pkl_path)   
        model_2, preprocess_2 = clip.load('another_pkl_path')
        
        weights_1 = model_1.state_dict()
        weights_2 = model_2.state_dict()
        
        if 'TestSetA' in TestData_path or 'TestSetB' in TestData_path:
            top5_result = wise_test(0.2)
            df_result['img_name'] = imgs
            df_result[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']] = top5_result
            df_result.to_csv(result_name, sep=' ', index=False, header=False) 
        
        else:
            best_acc   = 0
            best_alpha = 0
            for ALPHA in np.arange(0.1, 1, 0.01):
                top5_result   = wise_test(ALPHA)
                top1_result = top5_result[: ,0]
                num         = jt.sum(jt.equal(test_label, top1_result))
                acc         = num / 3000
                if round(float(acc), 4) >  best_acc:
                    best_acc   = round(float(acc), 4)
                    best_alpha = ALPHA
                    print(f'The best acc is {best_acc} ; best x is : {best_alpha}')
            print(f"共{len(df_label)}张图片，预测完成！")
            print("Best alpha is : ", best_alpha)
            return num

    
    
    
    
    # 训练的clip模型 + Tip-adapter-F + FD-Align + cross_modal_adapter
    elif method_name == 'fusion':
        print(f'Utilized the {method_name} method!')               
        
        from finetune.FD_Align import Prototype
        class_prototype, prompt_prototype= Prototype(classes_path)
        
        def get_logits_FD_Align(pkl_FD):
            
            model_FD_Align, preprocess_FD_Align = clip.load(pkl_FD)          
            test_features   = get_test_fea(model_FD_Align, preprocess_FD_Align)
            
            text_probs = (100.0 * test_features @ class_prototype.transpose(0, 1)).softmax(dim=-1)
            return text_probs, count_parameters_in_mb(model_FD_Align)  

        def get_logits_Tip_Adapter_F(pkl_TAF, alpha, beta):
            cached_keys , cached_values= Tip_adapter(root_TrainSet, class_4_path , classes_path , model , preprocess)

            # 加载训练的Tip-adapater-F
            adapter       = nn.Linear(cached_keys.shape[1], cached_keys.shape[0], bias=False)
            adapter.load_state_dict(jt.load(pkl_TAF)) 

            test_features = get_test_fea() 
            affinity      = adapter(test_features)
            cache_logits  = ((-1) * (beta - beta*  affinity)).exp() @ cached_values 

            text_probs    = (100.0 * test_features @ text_features.transpose(0, 1)).softmax(dim=-1)
            logits        = alpha * cache_logits + text_probs
            # logits        = cache_logits 
            return logits, count_parameters_in_mb(adapter)
        
        def get_logits_cross_modal(pkl_cross):
            from finetune.Cross_modal_Adapter import logitHead

            model.add_module('cross_logit', logitHead(get_val_text_features(classes_path, model)))
            model.load_state_dict(jt.load(pkl_cross))

            test_features = get_test_fea(model)
            text_probs = (100.0 * model.cross_logit(test_features)).softmax(dim=-1)
            return text_probs, count_parameters_in_mb(model)

        
        def get_logits_clip_train(pkl_clip):
            img_fea       = get_test_fea(model, preprocess)
            test_probs    = (100.0 * img_fea @ text_features.transpose(0, 1)).softmax(dim=-1)
            return test_probs, count_parameters_in_mb(model)

        logits_FD_Align,      parameters_0   = get_logits_FD_Align(FD_Align_pkl_path)
        logits_Tip_Adapter_F, parameters_1   = get_logits_Tip_Adapter_F(Tip_Adapter_F_pkl_path, alpha=1.4, beta=1.3)
        logits_clip, parameters_2            = get_logits_clip_train(pkl_path)
        logits_cross_modal,   parameters_3   = get_logits_cross_modal(cross_modal_pkl_path)

        total_parameters = parameters_0 + parameters_1 + parameters_2 + parameters_3
        print(f'The total number of parameters used by the model is: {total_parameters:.2f} Mb') 
        
        def fusion_test(logits_FD_Align, logits_Tip_Adapter_F, logits_cross_modal, X, Y, Z):
            logits  = X * logits_Tip_Adapter_F + Y * logits_FD_Align + Z * logits_cross_modal
            _, top_labels = logits.topk(5)
            return top_labels
        
        if 'TestSetA' in TestData_path or 'TestSetB' in TestData_path:
            top5_result           = fusion_test(logits_FD_Align, logits_Tip_Adapter_F, logits_cross_modal, X=0.04, Y=0.53, Z=0.43)
            df_result['img_name'] = imgs
            df_result[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']] = top5_result
            df_result.to_csv(result_name, sep=' ', index=False, header=False) 
        else:
            best_acc   = 0
            X_list     = np.arange(0.00, 1.00, 0.02)
            for X in X_list:
                for Y in np.arange(0.00, 1-X, 0.02):
                    Z = round(1-X-Y, 2)
                    top1_result   = fusion_test(logits_FD_Align, logits_clip, logits_cross_modal, X=X, Y=Y, Z=Z)[:,0]
                    num = jt.sum(jt.equal(test_label, top1_result))
                    acc = num / 3000
                    if round(float(acc), 4) >  best_acc:
                        best_acc   = round(float(acc), 4)
                        print(f'The best acc is {best_acc} ; best x is : {X}, best y is : {Y}, best z is : {Z}')
            return num
    
   
   
    # 训练和的clip模型 + 使用Tip-adapter-F 方法微调的clip模型
    elif method_name == 'fusion_2':
        
        print(f'Utilized the {method_name} method!')

        test_features    = get_test_fea()

        def get_clip_result():
            text_probs       = (100.0 * test_features @ text_features.transpose(0, 1)).softmax(dim=-1)
            top5_result      = text_probs.topk(5)[1] 
            return top5_result, count_parameters_in_mb(model)

        def get_Tip_adapater_F_result(pkl_TAF, alpha, beta):
            cached_keys , cached_values= Tip_adapter(root_TrainSet, class_4_path , classes_path , model , preprocess)

            # 加载训练的Tip-adapater-F
            adapter       = nn.Linear(cached_keys.shape[1], cached_keys.shape[0], bias=False)
            adapter.load_state_dict(jt.load(pkl_TAF)) 

            affinity      = adapter(test_features)
            cache_logits  = ((-1) * (beta - beta*  affinity)).exp() @ cached_values 

            text_probs    = (100.0 * test_features @ text_features.transpose(0, 1)).softmax(dim=-1)
            logits        = alpha * cache_logits + text_probs

            top5_result   = logits.topk(5)[1]
            return top5_result, count_parameters_in_mb(adapter)
        
        def result_fusion(top5_result_clip, top5_result_TAF):

            '''
            对于top5_result_clip预测的结果，即训练后的clip模型预测的结果：
            如果预测的类别在训练集上未出现过，则保留；反之，则替换成使用了Tip-adapter-F方法预测的结果
            '''
            index                   = np.where(top5_result_clip < 374) 
            top5_result_clip[index] = top5_result_TAF[index]
            return top5_result_clip
        
        top5_result_clip, parameters_1 = get_clip_result()
        top5_result_TAF,  parameters_2 = get_Tip_adapater_F_result(Tip_Adapter_F_pkl_path, alpha=2.4, beta=0.7)
    
        total_parameters = parameters_1 + parameters_2
        print(f'The total number of parameters used by the model is: {total_parameters:.2f} Mb') 

        top5_result   = result_fusion(top5_result_clip, top5_result_TAF)

        # top1_result   = top5_result[:,0]
        # num = jt.sum(jt.equal(test_label, top1_result))
        # print(num / 3000)

        df_result['img_name'] = imgs
        df_result[['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']] = top5_result
        df_result.to_csv(result_name, sep=' ', index=False, header=False) 

        
        

if __name__ == "__main__":

    jt.flags.use_cuda = 1  

    root_TrainSet = r'E:\Competition1'
    
    pkl_path               = "E:\Competition1\Weights\CLIP-0820.pkl"
    Tip_Adapter_F_pkl_path = "E:\Competition1\Weights\Tip_adapter_F-0820.pkl"
    
    TestData_path = f"{root_TrainSet}\TestSetB/"           # TestSetA | TestSetB | TestSetZ
    label_path    = f"{root_TrainSet}\TestSetZ-label.txt"
    classes_path  = f"{root_TrainSet}\classes_b.txt"

    FD_Align_pkl_path      = ""
    os.environ['ROOT_PATH']= root_TrainSet
    cross_modal_pkl_path   = ""


    method_name   = 'fusion_2'
    num           = compute_acc_TestSetZ(root_TrainSet, pkl_path, FD_Align_pkl_path, Tip_Adapter_F_pkl_path, cross_modal_pkl_path, classes_path , TestData_path ,label_path, method_name = method_name)
    if 'TestSetZ' in TestData_path:
        print(f"该模型在{TestData_path[-9:-1]}数据集上的分类准确率为：{num / 3000:.4f}")
    else:
        print(f'该模型在{TestData_path[-9:-1]}数据集上预测完成！')
