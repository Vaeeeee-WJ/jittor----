from   loguru           import logger
from utils import get_date_format
import jittor           as     jt
import jittor.transform as     transforms
from   jittor           import nn
from   jittor.dataset   import Dataset
from   jittor.dataset   import DataLoader
import jclip            as     clip
from jclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = jt.float32

    def execute(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.cast(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).cast(self.dtype)
        x = x[jt.arange(x.shape[0]), jt.argmax(tokenized_prompts, dim=-1)[0]] @ self.text_projection

        return x
    
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls =  len(classnames) 
        n_ctx =  16  
        ctx_init = ""
        dtype = jt.float16
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)

            with jt.no_grad():
                embedding = clip_model.token_embedding(prompt).cast(dtype)

            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            if False:
                print("Initializing class-specific contexts")
                ctx_vectors = jt.empty((n_cls, n_ctx, ctx_dim), dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = jt.empty((n_ctx, ctx_dim), dtype=dtype)

            nn.init.gauss_(ctx_vectors, mean=0, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)
        # classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = jt.concat([clip.tokenize(p) for p in prompts])

        with jt.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).cast(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = "end"

    def execute(self):
        ctx = self.ctx
        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(0).expand((self.n_cls, -1, -1))

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = jt.concat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = jt.concat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = jt.concat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = jt.concat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = jt.concat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, clip_model ,classnames):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
    
    # 训练CoOp时的前向阶段
    def execute(self, image):
        image = image.cast(self.dtype)  # Jittor 的类型转换
        image_features = self.image_encoder(image)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features  /=  text_features.norm(dim=-1, keepdim=True)

        logit_scale = jt.exp(self.logit_scale)
        logits = logit_scale * image_features @ text_features.t()

        return logits

# class CoOp():
#     def __init__(self, clip_path,  classnames):
#         net, _ = clip.load(clip_path)
#         self.classnames = classnames
#         self.clip_model = net
#     def build_model(self):
#         print("Building custom CLIP")
#         self.model = CustomCLIP(self.clip_model , classnames=self.classnames)

#         # 关闭模型中除了 prompt_learner 之外的所有参数的梯度计算
#         print("Turning off gradients in both the image and the text encoder")
#         for name, param in self.model.named_parameters():
#             if "prompt_learner" not in name:
#                 param.requires_grad_(False)

# net, preprocess = clip.load('F:\jittor_comprtition\Competition1\JCLIP\ViT-B-32.pkl')
# b = PromptLearner('dog', net)


def trainer_Coop(model, train_loader, val_loader, scheduler, optimizer, EPOCHS, classes_path, save_path):

    '''
    model        : 待训练的模型
    train_loader : 训练集的DataLoader
    val_loader   : 验证集的DataLoader
    scheduler    : 学习率衰减策略
    optimizer    : 优化器
    EPOCHS       : 训练的轮数
    classes_path : classes.txt文件路径
    save_path    : 保存模型的路径
    '''
    
    best_acc   = 0
    for epoch in range(EPOCHS): 
        for name,param in model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad = False 

        total_loss = 0   # 总损失
        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(total_loss) 
        else :
            scheduler.step() 
    
        # 开始训练 
        model.train()
        for images, label in train_loader:
            optimizer.zero_grad()
            logits      = model(images)
            cur_loss    = nn.CrossEntropyLoss()(logits*5.0, label) # TAG
            total_loss += cur_loss
            optimizer.backward(cur_loss)   
            optimizer.step()
        
        logger.info('train epoch:{}     total_loss:{:.4f}'.format(epoch+1, total_loss))

        # 验证开始
        if (epoch+1)%10 == 0: 
            model.eval()
            val_acc  = 0 
            num      = 0
            with jt.no_grad():
                
                for images, label in val_loader:
                    image_features    =  model.image_encoder(images)
                    image_features   /= image_features.norm(dim=-1, keepdim=True)
                    
                    prompts           = model.prompt_learner()
                    tokenized_prompts = model.tokenized_prompts
                    text_features     = model.text_encoder(prompts, tokenized_prompts)
                    text_features    /= text_features.norm(dim=-1, keepdim=True)
                    
                    text_probs      = (100.0 * image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
                    
                    _, top_labels = text_probs.topk(1)  
                    batch_acc = jt.sum(jt.equal(label.unsqueeze(1),top_labels)).tolist()[0]  # 每一批中正确的个数
                    val_acc  += batch_acc 

            val_acc = val_acc / 3000
            logger.info('val epoch:{} acc:{:.4f}'.format(epoch+1, val_acc))   

            if val_acc > best_acc:
                best_acc = val_acc
                jt.save(model.state_dict(), save_path + f"CLIP-Coop-{get_date_format()}.pkl")