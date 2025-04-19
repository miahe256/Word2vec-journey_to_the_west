import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out

    def get_word_vector(self, word_idx):
        return self.embeddings(word_idx).detach()

def load_model(model_path):
    """加载训练好的模型和词表"""
    checkpoint = torch.load(model_path)
    vocab = checkpoint['vocab']
    vector_size = checkpoint['vector_size']
    
    # 初始化模型
    model = Word2Vec(len(vocab), vector_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 如果有GPU则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    return model, vocab

def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度"""
    return torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item()

def get_word_vector(model, vocab, word):
    """获取词的向量表示"""
    if word not in vocab:
        raise ValueError(f"词语 '{word}' 不在词表中")
    device = next(model.parameters()).device
    word_idx = torch.tensor(vocab[word], device=device)
    return model.get_word_vector(word_idx)

def find_similar_words(model, vocab, target_word, top_k=5):
    """查找与目标词最相似的K个词"""
    if target_word not in vocab:
        raise ValueError(f"词语 '{target_word}' 不在词表中")
    
    # 获取目标词向量
    target_vector = get_word_vector(model, vocab, target_word)
    
    # 构建反向词表
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    # 计算与所有词的相似度
    similarities = []
    for word in vocab.keys():
        if word != target_word:
            vector = get_word_vector(model, vocab, word)
            similarity = cosine_similarity(target_vector, vector)
            similarities.append((word, similarity))
    
    # 按相似度排序并返回前K个
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def test_word_similarities(model, vocab, words):
    """测试词语之间的相似度"""
    for w1 in words:
        for w2 in words:
            if w1 != w2:
                try:
                    v1 = get_word_vector(model, vocab, w1)
                    v2 = get_word_vector(model, vocab, w2)
                    sim = cosine_similarity(v1, v2)
                    print(f"{w1} 和 {w2} 的相似度: {sim:.4f}")
                except ValueError as e:
                    print(e)

def word_vector_operation(model, vocab, positive_words, negative_words, top_k=5):
    """进行词向量运算，类似于 word1 + word2 - word3
    
    Args:
        model: Word2Vec模型
        vocab: 词表
        positive_words: 加法运算的词列表
        negative_words: 减法运算的词列表
        top_k: 返回最相似词的数量
    
    Returns:
        与结果向量最相似的top_k个词及其相似度
    """
    # 确保所有词都在词表中
    for word in positive_words + negative_words:
        if word not in vocab:
            raise ValueError(f"词语 '{word}' 不在词表中")
    
    # 计算正向词的向量和
    positive_vecs = [get_word_vector(model, vocab, word) for word in positive_words]
    # 计算负向词的向量和
    negative_vecs = [get_word_vector(model, vocab, word) for word in negative_words]
    
    # 计算结果向量：positive_vecs的和 - negative_vecs的和
    result_vector = sum(positive_vecs) - sum(negative_vecs)
    
    # 计算与所有词的相似度
    similarities = []
    for word in vocab.keys():
        # 排除参与运算的词
        if word not in positive_words and word not in negative_words:
            vector = get_word_vector(model, vocab, word)
            similarity = cosine_similarity(result_vector, vector)
            similarities.append((word, similarity))
    
    # 按相似度排序并返回前K个
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

if __name__ == "__main__":
    # 加载模型
    model_path = './models/word2vec_torch.pt'
    model, vocab = load_model(model_path)
    
    # 测试词语相似度
    test_words = ['孙悟空', '红孩儿', '花果山', '取经']
    print("\n测试词语相似度:")
    test_word_similarities(model, vocab, test_words)
    
    # 查找相似词
    print("\n查找相似词:")
    for word in test_words:
        try:
            similar_words = find_similar_words(model, vocab, word, top_k=5)
            print(f"\n与 '{word}' 最相似的词语:")
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        except ValueError as e:
            print(e)
    
    # 测试词向量运算
    print("\n测试词向量运算:")
    try:
        # 计算 孙悟空 + 唐僧 - 大王
        result = word_vector_operation(
            model, 
            vocab,
            positive_words=['孙悟空', '唐僧'],
            negative_words=['大王'],
            top_k=5
        )
        print("\n'孙悟空 + 唐僧 - 大王' 最相似的词语:")
        for word, similarity in result:
            print(f"  {word}: {similarity:.4f}")
    except ValueError as e:
        print(e) 