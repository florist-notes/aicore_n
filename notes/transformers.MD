# Transformers

```python

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

```


<img src="img/trans.png" width=100%> 

```python

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

```

resources: [How Transformers Work](https://towardsdatascience.com/transformers-141e32e69591), [Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html), [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [The Transformer Model](https://machinelearningmastery.com/the-transformer-model/), [Illustrated Guide to Transformers Neural Network: A step by step explanation](https://www.youtube.com/watch?v=4Bdc55j80l8), [NLP Demystified 15: Transformers From Scratch + Pre-training and Transfer Learning With BERT/GPT](https://youtu.be/acxqoltilME), [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://youtu.be/zxQyTK8quyY), [Pytorch Transformers from Scratch (Attention is all you need)](https://youtu.be/U0s0f995w14), [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY), [Transformers from scratch](https://www.youtube.com/playlist?list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4), [Vision Transformers (ViT) Explained + Fine-tuning in Python](https://youtu.be/qU7wO02urYU), [Vision Transformers explained](https://www.youtube.com/playlist?list=PLpZBeKTZRGPMddKHcsJAOIghV8MwzwQV6), [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Paper Explained)](https://www.youtube.com/watch?v=TrdevFK_am4), [MLP-Mixer: An all-MLP Architecture for Vision (Machine Learning Research Paper Explained)](https://www.youtube.com/watch?v=7K4Z8RqjWIk), [Robust Perception with Vision Transformer SegFormer](https://youtu.be/t79E4gq4L-A), [Transformers: a Primer](http://www.columbia.edu/~jsl2239/transformers.html), @github/[transformers](https://github.com/hyunwoongko/transformer), [How to code The Transformer in Pytorch](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec).