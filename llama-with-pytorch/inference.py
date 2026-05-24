from typing import Optional

import torch
import json
from pathlib import Path
from sentencepiece import SentencePieceProcessor
import time

from tqdm import tqdm

from model import Transformer, ModelArgs


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int,
              device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, f"checkpoint 文件没有在 {checkpoints_dir} 找到"
            chk_path = checkpoints[0]
            print(f"加载模型文件... {chk_path}")
            checkpoint = torch.load(chk_path, map_location='cpu')
            print(f"checkpoint 文件加载完成, 耗时 {(time.time() - prev_time):.2f}s")
            prev_time = time.time()

        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        with open(Path(checkpoints_dir) / "params.json", 'r') as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        model = Transformer(model_args).to(device)

        if load_model:
            # 从checkpoint中删除 rope.freqs, 因为我们前面通过 precompute_theta_pos_frequencies 函数自己计算了
            del checkpoint["rope.freqs"]
            # strict=True 模型是字典, 键值对形式, 所以参数名字要对上, 如果发现对不上就抛异常
            model.load_state_dict(checkpoint, strict=True)
            print(f"加载 state dict 耗时 {time.time() - prev_time:.2f}s")

        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts: list[str], device: str, do_sample: bool = True, temperature: float = 0.6,
                        top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len

        # 转换每个 prompt 为 token ids
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]

        # 确保 batch size 不要太大, 因为我们指明了 KV cache 的大小
        batch_size = len(prompts)
        assert batch_size <= self.args.max_batch_size, f"batch size 必须小于等于 {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, f"prompt 长度必须小于等于 {self.args.max_seq_len}"

        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id()  # 获取分词器对应填充所使用的pad在词表中的int值
        # 我们会把输入的提示词和模型输出的结果统统放到 tokens 变量中
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # 将 tokens 矩阵中提示词对应的token替换，没有被替换的位置依然是padding token
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # 初始化 eos_reached 对于每个prompt 都是 False, 后面每个 prompt的 eos_reached 都为 True, 就无需再去 generate
        eos_reached = torch.tensor([False] * batch_size, device=device)
        # 计算一个矩阵 prompt_tokens_mask 指明在 tokens 矩阵中哪些位置对应的提示词 token id, 哪些位置是 padding id
        prompt_tokens_mask = tokens != pad_id  # True 如果token是输入的提示词token, 否则就是 False

        # 接下来就可以把准备好的输入交给模型进行推理预测
        for cur_pos in tqdm(range(1, total_len), desc="生成 tokens"):
            with torch.no_grad():
                # 每次传入一个token
                logits = self.model.forward(tokens[:, cur_pos - 1:cur_pos], cur_pos)

            if do_sample:
                # 基于 Top P 的随机采样策略
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # 取当前这一个时刻最大的值对应索引, Greedy Search
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # prompt_tokens_mask 中对应位置是 True, 那么next_token就来自于提示词, 否则就用 LLM 预测出来的 token 作为 next_token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)

            tokens[:, cur_pos] = next_token

            # EOS is reached 仅当我们发现在新生成token位置中包含 EOS token
            # 反过来说就是, 因为提示词对于LLM给出的next_token如果是EOS的话, 我们不关系
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())

            # 对于每一个prompt提示词LLM都已经生成过EOS标识, 就可以停止继续inference
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # 只保留到 EOS token, 如果句子中包含 EOS 的话, 输入的提示词 prompt 不能包含 EOS
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return out_tokens, out_text

    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        # 没有被 top p 选择的 tokens 的概率值 全部设置为 0.0
        probs_sort[mask] = 0.0
        # 重新获得一个加起来是 1 的概率分布
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # 下面进行随机采样
        # 从 top p distribution 中采样一个 token 的 index, 注意此时 index 并不是词典中对于的 token_id
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # 因为一开始概率分布probs进行了排序, 所以采样出来的 token index 并非词典中对应的 Token id
        # 这就是为什么一开始 sort 要接收两个东西的原因, probs_sort 是根据顺序排序的概率值,
        # probs_idx 是排序后概率值对应的原有的词典中的顺序
        next_token = torch.gather(probs_idx, -1, next_token)  # 对应在词典中的 token id
        return next_token


if __name__ == '__main__':
    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "Simply put, the theory of relativity states that ",
        "1+1=2 2+3=4 4+10= "
    ]

    model = LLaMA.build(
        checkpoints_dir='f:/LLaMA/llama-2-7b/',
        tokenizer_path='f:/LLaMA/llama-2-7b/tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    print('ALL OKAY')

    out_tokens, out_texts = (model.text_completion(prompts, device=device, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)
