

def get_loaders(eval_dataset, nsamples, seed, seqlen, tokenizer, eval_model=False):
    from datasets import load_dataset
    import random

    if eval_dataset in 'wikitext2':
        if eval_model:
            testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            testenc = tokenizer(
                "\n\n".join(testdata['text']),
                return_tensors='pt',
            )
            print(f"testenc.input_ids.shape:{testenc.input_ids.shape}") #shape为[1,wikitest2_seq]。
            print("Loaded wikitext2 dataset for evaluation.")
            return testenc
        else:
            train_data=load_dataset('wikitext','wikitext-2-raw-v1',split='train')
            trainenc = tokenizer(
                "\n\n".join(train_data['text']),
                return_tensors='pt',
            )
            random.seed(seed)
            train_loader =[]
            for _ in range(nsamples):
                # random.randint计算范围为[a,b],包括a和b
                i=random.randint(0, train_loader.input_ids.shape[0]-seqlen-1)
                input = trainenc.input_ids[:,i,i+seqlen] #切片返回的是视图
                tar = input.clone()
                tar[:,:-1]=-100 # tar[:, :-1] = -100这行代码把目标张量中除了最后一个标记之外的其他标记都赋值为-100。在 PyTorch 的交叉熵损失函数里，-100是一个特殊值，它表示这个位置的标记在计算损失时会被忽略。只考虑对最后一个标记的预测是否准确
                train_loader.append((input,tar))
            
            return testenc



def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "meta-llama/Llama-2-7b-hf"
    print("Loading model...")
    model=AutoModelForCausalLM.from_pretrained(model_name,
                                               torch_dtype=torch.fp16,
                                               device_map="cpu",
                                               low_cpu_mem_usage=True,
                                               )
    print("Loading model success...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading tokenizer success...")
    layers=model.model.layers

    testloader=get_loaders(eval_dataset='wikitext2',nsamples=128,seed=0,seqlen=2048,tokenizer=tokenizer,eval_model=True)

    




if __name__=='main':
    main()