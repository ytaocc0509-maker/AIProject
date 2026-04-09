import tiktoken

# 根据模型名字来获取采用的分词器编码名称
tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
print(tokenizer.name)

# 根据名称得到分词器
# tokenizer2 = tiktoken.get_encoding('cl100k_base')

# 分词得到向量
res1 = tokenizer.encode('China is great!')
res1 = tokenizer.encode('tiktoken is great!')
res2 = tokenizer.encode('这种模型通常用于自动摘要、文章创作、代码生成等任务，其中用户提供部分内容，而模型则帮助完成剩余的文本。')
print(res2)
print(len(res2))

# 根据向量，还原文本
print(tokenizer.decode(res2))

print(res1)

# 把每一个整数还原成一个词条（token）
words = [tokenizer.decode_single_token_bytes(token) for token in res1 ]
print(words)