# Demo Retrieval Dataset

这个目录是一个用于学习文搜图和图搜图 embedding 训练的数据集样例，所有图片都是脚本生成的合成图。

## 设计思路

- 推理阶段的 document 库只有图片，所以 `corpus.jsonl` 只保存图片 document。
- 文搜图训练样本把文本 query 作为 anchor，把相关图片作为 positive。
- 图搜图训练样本把查询图片作为 anchor，把同一 item family 的图库图作为 positive。
- ms-swift 训练 JSONL 的 document 侧只使用 `<image>` 加固定图库编码 instruction，不把 caption 混入模型输入，以贴近推理阶段只建图片 embedding 的设置。
- `qrels.jsonl` 保留一个 query 对多个正例的评测关系；ms-swift 训练 JSONL 则按一条样本一个 positive 的格式展开。
- hard negative 优先来自同类目但不同颜色或不同细节的图片，例如红色背包 query 的 hard negative 是蓝色背包。

## 文件

| 文件 | 用途 |
|---|---|
| `images/corpus/*.png` | 图库图片，共 40 张 |
| `images/query/*.png` | 图搜图查询图片，共 8 张 |
| `corpus.jsonl` | 图库 document 元数据 |
| `queries.jsonl` | 文本 query 和图片 query 的统一格式 |
| `qrels.jsonl` | 评测用 query-document 正例关系 |
| `train_text_to_image_ms_swift.jsonl` | 文搜图 InfoNCE 训练样本 |
| `train_image_to_image_ms_swift.jsonl` | 图搜图 InfoNCE 训练样本 |
| `train_embedding_combined_ms_swift.jsonl` | 文搜图 + 图搜图混合训练样本 |
| `val_*_ms_swift.jsonl` | 验证集样本，按 item family 与 train 隔离 |

## 推荐扩展方式

真实业务里不要只准备“图片 caption”。更好的最小单元是：

```json
{
  "query": "红色帆布双肩包，正面有白色竖条",
  "positives": ["同款/同实体/同语义族图片"],
  "hard_negatives": ["同类目但颜色、图案、型号或语义不同的图片"]
}
```

如果一个 query 有多个等价正例，评测文件可以保留全部正例；训练文件可以把它们展开成多条样本，或在 sampler 里避免同一正例族互相成为 in-batch false negative。
