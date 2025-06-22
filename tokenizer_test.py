from transformers import AutoTokenizer
from huggingface_hub import login

if __name__ == '__main__':
    prompt="hello how are you? \n\n hello how are you?"
    model_name='meta-llama/Llama-2-7b-hf'
    tokenizer=AutoTokenizer.from_pretrained(model_name)  
    input_ids=tokenizer.encode(prompt)
    print(input_ids) # [1, 22172, 920, 526, 366, 29973, 29871, 13, 13, 22172, 920, 526, 366, 29973]

    return_pt=tokenizer(prompt,return_tensors='pt') # 不设置返回值类型则默认返回为python list格式
    print(return_pt.input_ids) 
    # tensor([[    1, 22172,   920,   526,   366, 29973, 29871,    13,    13, 22172,
    #      920,   526,   366, 29973]])
    

    # 也可以接受数组
    prompt_list=['hello','how are you?']
    encode_res=tokenizer.encode(prompt_list)
    print(encode_res)
    encode_res=tokenizer(prompt_list)
    print(encode_res.input_ids)