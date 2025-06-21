import utils

import torch
import os
import logging
from tqdm import tqdm

"""
困惑度评估工具

"""

@torch.no_grad()
def evaluator(model, testenc, dev, args):

    

    model.eval()

    if 'opt' in args.model:
        opt_type = True
        llama_type = False
    elif 'meta' in args.model:
        llama_type = True
        opt_type = False
    else:
        raise ValueError(f'Unknown model {args.model}')
    
    print("===== 评估参数检查 =====")
    model_type = "Llama" if llama_type else "OPT"
    print(f"模型类型: {model_type}")
    print(f"模型序列长度: {model.seqlen}")
    print(f"评估批次大小 args.bsz: {args.bsz}")
    print(f"输入数据形状: {testenc.input_ids.shape}")
    print("=======================")


    use_cache = model.config.use_cache
    model.config.use_cache = False

    if opt_type:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)

    elif llama_type:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)

    layers[0] = layers[0].to(dev)

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)  # (nsamples, seqlen)

    batch_size = args.bsz
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)] # Python 的切片操作会自动处理超出范围的索引，确保不会引发错误
    nbatches = len(input_ids)
    if not nsamples%batch_size == 0:
        nbatches=nbatches-1

    print(f"总样本数 nsamples: {nsamples}")
    print(f"批次数 nbatches: {nbatches}")
    print(f"每批大小: {batch_size}")
    if nbatches > 0:
        print(f"第一批输入形状: {input_ids[0].shape}")
        print(f"最后一批输入形状: {input_ids[-1].shape}")

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    inps = [0] * nbatches
    cache = {'i': 0, 'attention_mask': None}
    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            print(f"模型内部生成的注意力掩码形状: {kwargs['attention_mask'].shape}")
            print(f"当前批次输入形状: {inp.shape}")
            cache['attention_mask'] = kwargs['attention_mask']
            if llama_type:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])

    for i in range(nbatches):
        batch = input_ids[i]
        print(f"处理第{i}批，输入形状: {batch.shape}")
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    if opt_type:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif llama_type:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        position_ids = cache['position_ids']

    torch.cuda.empty_cache()
    outs = [0] * nbatches
    attention_mask = cache['attention_mask']

    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)

        print(f"第{i}层处理，注意力掩码形状: {attention_mask.shape}")

        for j in range(nbatches):
            if opt_type:
                outs[j] = layer(inps[j], attention_mask=attention_mask)[0]
            elif llama_type:
                outs[j] = layer(inps[j], attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if opt_type:
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)

    elif llama_type:
        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)

    model.lm_head = model.lm_head.to(dev)
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction = "none") #如果 reduction='none'，则为loss 的形状 [batch_size, seq_len]
    for i in range(nbatches):
        hidden_states = inps[i]
        if opt_type:
            if model.model.decoder.final_layer_norm is not None:
                hidden_states = model.model.decoder.final_layer_norm(hidden_states)
            if model.model.decoder.project_out is not None:
                hidden_states = model.model.decoder.project_out(hidden_states)
        elif llama_type:
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        # 获取lm_head的输出结果和input_ids做比较。lm_logits除去最后一行，input_ids除去第一行
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels) # shift_logits.permute(0, 2, 1)：将维度调整为 [batch_size, vocab_size, seq_len]，符合 PyTorch 交叉熵损失的输入要求。
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    model.config.use_cache = use_cache
    logging.info(f'\n{args.eval_dataset.upper()} PPL: {ppl.item():.3f}')
    return ppl.item()
