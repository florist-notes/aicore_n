# Transformers

In recent years, Transformers have emerged as a groundbreaking architecture in the field of deep learning, revolutionizing natural language processing and extending their capabilities to various domains, including computer vision.

A Transformer model consists of several key components:

   + <b>Multi-Head Self-Attention:</b> This mechanism enables the model to focus on different parts of the input sequence simultaneously, promoting better contextual understanding.

   + <b>Positional Encoding:</b> Transformers lack inherent information about the order of elements in a sequence, so positional encodings are added to input embeddings to convey their positions.

   + <b>Feedforward Neural Networks:</b> After attention mechanisms, a feedforward neural network is applied to capture complex patterns and relationships in the data.

   + <b>Layer Normalization and Residual Connections:</b> These techniques stabilize training and mitigate vanishing gradient issues.

The Transformer architecture follows an encoder-decoder structure but does not rely on recurrence and convolutions in order to generate an output. A transformer is a deep learning architecture that relies on the parallel multi-head attention mechanism.  


The task of the encoder, on the left half of the Transformer architecture, is to map an input sequence to a sequence of continuous representations, which is then fed into a decoder. The decoder, on the right half of the architecture, receives the output of the encoder together with the decoder output at the previous time step to generate an output sequence.

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
## Vision Transformers (ViT)

Vision Transformers extend the Transformer architecture to the domain of computer vision. ViTs have exhibited remarkable performance in image classification, object detection, and segmentation tasks. Their core idea is to convert 2D image data into a 1D sequence that can be processed by the Transformer architecture. 

The Vision Transformer, or ViT, is a model for image classification that employs a Transformer-like architecture over patches of the image. An image is split into fixed-size patches, each of them are then linearly embedded, position embeddings are added, and the resulting sequence of vectors is fed to a standard Transformer encoder.

ViT Architecture:

+ <b>Patch Embeddings:</b> Unlike Convolutional Neural Networks (CNNs), which operate on the entire image grid, ViTs divide images into non-overlapping fixed-size patches. These patches are linearly embedded into a sequence of vectors, preserving spatial relationships.

+ <b>Positional Encodings:</b> To convey positional information, ViTs use positional encodings, similar to language Transformers. These encodings are added to patch embeddings, enabling the model to distinguish between patches based on their positions.

+ <b>Transformer Layers:</b> ViTs consist of multiple Transformer layers, often resembling the original architecture introduced by Vaswani et al. Each layer consists of:

    - <b>Multi-Head Self-Attention:</b> Captures global and local dependencies between patches.
    - <b>Feedforward Neural Networks:</b> Applies nonlinear transformations to patch embeddings.
    - <b>Layer Normalization and Residual Connections:</b> Stabilize training and facilitate gradient flow.

+ <b>Classification Head:</b> After processing patch embeddings through Transformer layers, a classification head (e.g., a linear layer or multi-layer perceptron) is added for final predictions.

### Advantages of Vision Transformers:

+ <b>Parallelization:</b> ViTs enable parallel processing of patches, benefiting from the efficient use of hardware accelerators like GPUs and TPUs.

+ <b>Scalability:</b> ViTs can handle images of varying sizes without the need for resizing or cropping, making them adaptable to diverse applications.

+ <b>Global Context:</b> The self-attention mechanism in Transformers captures long-range dependencies, allowing ViTs to excel at tasks requiring a global understanding of the image.

+ <b>Transfer Learning:</b> Pre-trained ViT models can be fine-tuned on specific vision tasks with limited labeled data, yielding state-of-the-art results.

```py

import torch
import torch.nn as nn
from einops import rearrange

from self_attention_cv import TransformerEncoder


class ViT(nn.Module):
    def __init__(self, *,
                 img_dim,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 dropout=0, transformer=None, classification=True):
        """
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
            classification: creates an extra CLS token
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible'
        self.p = patch_dim
        self.classification = classification
        tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)
        self.dim = dim
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)
        if self.classification:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, dim))
            self.mlp_head = nn.Linear(dim, num_classes)
        else:
            self.pos_emb1D = nn.Parameter(torch.randn(tokens, dim))

        if transformer is None:
            self.transformer = TransformerEncoder(dim, blocks=blocks, heads=heads,
                                                  dim_head=self.dim_head,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer

    def expand_cls_to_batch(self, batch):
        """
        Args:
            batch: batch size
        Returns: cls token expanded to the batch size
        """
        return self.cls_token.expand([batch, -1, -1])

    def forward(self, img, mask=None):
        batch_size = img.shape[0]
        img_patches = rearrange(
            img, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)
        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        if self.classification:
            img_patches = torch.cat(
                (self.expand_cls_to_batch(batch_size), img_patches), dim=1)

        patch_embeddings = self.emb_dropout(img_patches + self.pos_emb1D)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)

        if self.classification:
            # we index only the cls token for classification. nlp tricks :P
            return self.mlp_head(y[:, 0, :])
        else:
            return y

```



resources: [How Transformers Work](https://towardsdatascience.com/transformers-141e32e69591), [Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html), [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [The Transformer Model](https://machinelearningmastery.com/the-transformer-model/), [Illustrated Guide to Transformers Neural Network: A step by step explanation](https://www.youtube.com/watch?v=4Bdc55j80l8), [NLP Demystified 15: Transformers From Scratch + Pre-training and Transfer Learning With BERT/GPT](https://youtu.be/acxqoltilME), [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!](https://youtu.be/zxQyTK8quyY), [Pytorch Transformers from Scratch (Attention is all you need)](https://youtu.be/U0s0f995w14), [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY), [Transformers from scratch](https://www.youtube.com/playlist?list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4), [Vision Transformers (ViT) Explained + Fine-tuning in Python](https://youtu.be/qU7wO02urYU), [Vision Transformers explained](https://www.youtube.com/playlist?list=PLpZBeKTZRGPMddKHcsJAOIghV8MwzwQV6), [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Paper Explained)](https://www.youtube.com/watch?v=TrdevFK_am4), [MLP-Mixer: An all-MLP Architecture for Vision (Machine Learning Research Paper Explained)](https://www.youtube.com/watch?v=7K4Z8RqjWIk), [Robust Perception with Vision Transformer SegFormer](https://youtu.be/t79E4gq4L-A), [Transformers: a Primer](http://www.columbia.edu/~jsl2239/transformers.html), @github/[transformers](https://github.com/hyunwoongko/transformer), [How to code The Transformer in Pytorch](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec), [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).