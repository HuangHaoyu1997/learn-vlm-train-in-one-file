from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "demo_retrieval"
IMAGE_SIZE = 512
RNG = random.Random(20260618)


FAMILIES = [
    {
        "family_id": "red_canvas_backpack",
        "category": "backpack",
        "category_zh": "帆布双肩包",
        "color_zh": "红色",
        "primary": (214, 57, 52),
        "accent": (255, 255, 255),
        "detail_zh": "白色竖条、黑色提手",
        "caption_zh": "红色帆布双肩包，正面有白色竖条和黑色提手",
        "text_queries": [
            "红色帆布双肩包，正面有白色竖条",
            "找红色背包，黑色提手，白色竖条装饰",
        ],
    },
    {
        "family_id": "blue_canvas_backpack",
        "category": "backpack",
        "category_zh": "帆布双肩包",
        "color_zh": "蓝色",
        "primary": (48, 106, 199),
        "accent": (245, 190, 62),
        "detail_zh": "黄色方形贴片、黑色提手",
        "caption_zh": "蓝色帆布双肩包，正面有黄色方形贴片和黑色提手",
        "text_queries": [
            "蓝色帆布双肩包，正面有黄色贴片",
            "找蓝色背包，黑色提手，黄色方形标记",
        ],
    },
    {
        "family_id": "red_running_shoe",
        "category": "shoe",
        "category_zh": "跑鞋",
        "color_zh": "红色",
        "primary": (207, 52, 65),
        "accent": (250, 250, 245),
        "detail_zh": "白色鞋底、三条鞋带",
        "caption_zh": "红色跑鞋，白色鞋底，鞋面有三条浅色鞋带",
        "text_queries": [
            "红色跑鞋，白色鞋底，浅色鞋带",
            "找红色运动鞋，侧面视角，白色厚鞋底",
        ],
    },
    {
        "family_id": "green_running_shoe",
        "category": "shoe",
        "category_zh": "跑鞋",
        "color_zh": "绿色",
        "primary": (46, 148, 106),
        "accent": (31, 41, 55),
        "detail_zh": "深色鞋底、两条鞋带",
        "caption_zh": "绿色跑鞋，深色鞋底，鞋面有两条深色鞋带",
        "text_queries": [
            "绿色跑鞋，深色鞋底，两条鞋带",
            "找绿色运动鞋，侧面轮廓，黑色鞋底",
        ],
    },
    {
        "family_id": "yellow_ceramic_mug",
        "category": "mug",
        "category_zh": "陶瓷马克杯",
        "color_zh": "黄色",
        "primary": (240, 191, 54),
        "accent": (46, 99, 190),
        "detail_zh": "蓝色星形图案、右侧杯柄",
        "caption_zh": "黄色陶瓷马克杯，杯身有蓝色星形图案和右侧杯柄",
        "text_queries": [
            "黄色马克杯，蓝色星形图案，右侧杯柄",
            "找黄色陶瓷杯，杯身有蓝色五角星",
        ],
    },
    {
        "family_id": "white_ceramic_mug",
        "category": "mug",
        "category_zh": "陶瓷马克杯",
        "color_zh": "白色",
        "primary": (245, 245, 235),
        "accent": (210, 54, 88),
        "detail_zh": "红色爱心图案、右侧杯柄",
        "caption_zh": "白色陶瓷马克杯，杯身有红色爱心图案和右侧杯柄",
        "text_queries": [
            "白色马克杯，红色爱心图案，右侧杯柄",
            "找白色陶瓷杯，杯身有红色心形标记",
        ],
    },
    {
        "family_id": "black_headphones",
        "category": "headphones",
        "category_zh": "头戴式耳机",
        "color_zh": "黑色",
        "primary": (35, 38, 47),
        "accent": (218, 68, 72),
        "detail_zh": "红色耳罩内衬、粗头梁",
        "caption_zh": "黑色头戴式耳机，红色耳罩内衬，粗头梁",
        "text_queries": [
            "黑色头戴式耳机，红色耳罩内衬",
            "找黑色耳机，粗头梁，耳罩里面是红色",
        ],
    },
    {
        "family_id": "silver_headphones",
        "category": "headphones",
        "category_zh": "头戴式耳机",
        "color_zh": "银色",
        "primary": (183, 190, 196),
        "accent": (52, 132, 190),
        "detail_zh": "蓝色耳罩内衬、细头梁",
        "caption_zh": "银色头戴式耳机，蓝色耳罩内衬，细头梁",
        "text_queries": [
            "银色头戴式耳机，蓝色耳罩内衬",
            "找银色耳机，细头梁，耳罩里面是蓝色",
        ],
    },
]


BG_COLORS = [
    (247, 241, 230),
    (229, 239, 243),
    (236, 232, 246),
    (239, 244, 229),
    (246, 235, 238),
    (232, 238, 236),
]


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rounded_shadow(canvas: Image.Image, bbox: tuple[int, int, int, int], radius: int) -> None:
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.rounded_rectangle(bbox, radius=radius, fill=(25, 25, 35, 48))
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))
    canvas.alpha_composite(shadow)


def draw_star(draw: ImageDraw.ImageDraw, center: tuple[int, int], radius: int, fill: tuple[int, int, int]) -> None:
    cx, cy = center
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    draw.polygon(points, fill=fill)


def draw_heart(draw: ImageDraw.ImageDraw, center: tuple[int, int], size: int, fill: tuple[int, int, int]) -> None:
    cx, cy = center
    r = size // 4
    draw.ellipse((cx - 2 * r, cy - r, cx, cy + r), fill=fill)
    draw.ellipse((cx, cy - r, cx + 2 * r, cy + r), fill=fill)
    draw.polygon([(cx - 2 * r, cy), (cx + 2 * r, cy), (cx, cy + 2 * r)], fill=fill)


def draw_backpack(draw: ImageDraw.ImageDraw, family: dict, variant: int) -> None:
    p = family["primary"]
    a = family["accent"]
    x0, y0, x1, y1 = 154, 126, 358, 392
    draw.line((198, 126, 198, 82, 314, 82, 314, 126), fill=(25, 28, 34), width=16)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=38, fill=p, outline=(44, 47, 56), width=5)
    draw.rounded_rectangle((178, 164, 334, 245), radius=24, fill=tuple(min(255, c + 24) for c in p), outline=(44, 47, 56), width=3)
    draw.rounded_rectangle((192, 282, 320, 365), radius=22, fill=tuple(max(0, c - 24) for c in p), outline=(44, 47, 56), width=3)
    if family["family_id"].startswith("red"):
        draw.rounded_rectangle((246, 150, 270, 372), radius=10, fill=a)
        draw.line((222, 176, 290, 176), fill=(44, 47, 56), width=4)
    else:
        draw.rounded_rectangle((218, 250, 294, 318), radius=12, fill=a)
        draw.line((184, 206, 328, 206), fill=(44, 47, 56), width=4)
    if variant % 2:
        draw.arc((120, 150, 205, 355), 80, 280, fill=(44, 47, 56), width=10)
        draw.arc((307, 150, 392, 355), -100, 100, fill=(44, 47, 56), width=10)


def draw_shoe(draw: ImageDraw.ImageDraw, family: dict, variant: int) -> None:
    p = family["primary"]
    a = family["accent"]
    upper = [(115, 292), (172, 214), (272, 214), (352, 271), (410, 295), (384, 335), (142, 335)]
    draw.polygon(upper, fill=p)
    draw.line(upper + [upper[0]], fill=(36, 39, 48), width=5, joint="curve")
    draw.rounded_rectangle((112, 319, 414, 364), radius=22, fill=a, outline=(36, 39, 48), width=5)
    draw.arc((190, 196, 314, 310), 200, 350, fill=(36, 39, 48), width=5)
    lace_color = (250, 250, 245) if family["family_id"].startswith("red") else (28, 33, 42)
    for i in range(3 if family["family_id"].startswith("red") else 2):
        x = 220 + i * 34
        draw.line((x, 246, x + 42, 278), fill=lace_color, width=5)
    if variant % 2 == 0:
        draw.ellipse((356, 278, 388, 310), fill=tuple(min(255, c + 25) for c in p), outline=(36, 39, 48), width=3)


def draw_mug(draw: ImageDraw.ImageDraw, family: dict, variant: int) -> None:
    p = family["primary"]
    a = family["accent"]
    draw.ellipse((176, 353, 336, 385), fill=(0, 0, 0, 28))
    draw.rounded_rectangle((162, 146, 326, 365), radius=30, fill=p, outline=(44, 47, 56), width=5)
    draw.ellipse((160, 126, 328, 172), fill=tuple(min(255, c + 16) for c in p), outline=(44, 47, 56), width=5)
    draw.arc((294, 198, 410, 316), -72, 78, fill=(44, 47, 56), width=25)
    draw.arc((303, 214, 388, 300), -70, 78, fill=BG_COLORS[variant % len(BG_COLORS)], width=22)
    if family["family_id"].startswith("yellow"):
        draw_star(draw, (245, 255), 43, a)
    else:
        draw_heart(draw, (245, 252), 56, a)


def draw_headphones(draw: ImageDraw.ImageDraw, family: dict, variant: int) -> None:
    p = family["primary"]
    a = family["accent"]
    band_width = 22 if family["family_id"].startswith("black") else 13
    draw.arc((145, 102, 367, 348), 200, 340, fill=p, width=band_width)
    draw.arc((164, 126, 348, 330), 200, 340, fill=(44, 47, 56), width=4)
    draw.rounded_rectangle((116, 242, 196, 358), radius=28, fill=p, outline=(44, 47, 56), width=5)
    draw.rounded_rectangle((316, 242, 396, 358), radius=28, fill=p, outline=(44, 47, 56), width=5)
    draw.rounded_rectangle((134, 264, 180, 342), radius=22, fill=a)
    draw.rounded_rectangle((332, 264, 378, 342), radius=22, fill=a)
    if variant % 2:
        draw.line((196, 298, 316, 298), fill=(44, 47, 56), width=5)


DRAWERS = {
    "backpack": draw_backpack,
    "shoe": draw_shoe,
    "mug": draw_mug,
    "headphones": draw_headphones,
}


def create_image(family: dict, output_path: Path, variant: int, query: bool = False) -> None:
    bg = BG_COLORS[(variant + len(family["family_id"])) % len(BG_COLORS)]
    canvas = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), bg + (255,))
    draw_bg = ImageDraw.Draw(canvas, "RGBA")

    for i in range(0, IMAGE_SIZE, 42):
        draw_bg.line((i, 0, i - 130, IMAGE_SIZE), fill=(255, 255, 255, 52), width=2)
    draw_bg.rounded_rectangle((48, 54, 464, 450), radius=36, outline=(44, 47, 56, 45), width=2)

    object_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    rounded_shadow(object_layer, (118, 100, 408, 398), 45)
    draw_obj = ImageDraw.Draw(object_layer, "RGBA")
    DRAWERS[family["category"]](draw_obj, family, variant)

    angle = [-4, 3, 0, 5, -2, 7][variant % 6]
    if query:
        angle += 7
    object_layer = object_layer.rotate(angle, resample=Image.Resampling.BICUBIC, center=(256, 256))
    canvas.alpha_composite(object_layer)

    draw = ImageDraw.Draw(canvas, "RGBA")
    label = family["family_id"].replace("_", " ").upper()
    badge = "QUERY" if query else f"V{variant + 1}"
    draw.rounded_rectangle((58, 410, 454, 468), radius=14, fill=(255, 255, 255, 190), outline=(44, 47, 56, 55), width=2)
    draw.text((78, 419), label[:28], font=font(18, bold=True), fill=(31, 35, 45))
    draw.text((78, 445), badge, font=font(15), fill=(80, 84, 96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, quality=95)


def sibling_hard_negatives(family: dict, corpus_by_family: dict[str, list[dict]]) -> list[str]:
    same_category = [
        other
        for other in FAMILIES
        if other["category"] == family["category"] and other["family_id"] != family["family_id"]
    ]
    candidates: list[str] = []
    for other in same_category:
        candidates.extend(doc["doc_id"] for doc in corpus_by_family[other["family_id"]][:3])

    same_color = [
        other
        for other in FAMILIES
        if other["color_zh"] == family["color_zh"] and other["family_id"] != family["family_id"]
    ]
    for other in same_color:
        candidates.extend(doc["doc_id"] for doc in corpus_by_family[other["family_id"]][:2])

    if len(candidates) < 3:
        for other in FAMILIES:
            if other["family_id"] != family["family_id"]:
                candidates.extend(doc["doc_id"] for doc in corpus_by_family[other["family_id"]][:1])

    return list(dict.fromkeys(candidates))[:3]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def ms_swift_record(query: dict, positive_doc: dict, negative_docs: list[dict]) -> dict:
    if query["modality"] == "text":
        messages = [
            {
                "role": "user",
                "content": f"{query['instruction']}：{query['text']}",
            }
        ]
        images = None
    else:
        messages = [
            {
                "role": "user",
                "content": f"<image>{query['instruction']}。请找同一物品或同一视觉语义族的图片。",
            }
        ]
        images = [query["image"]]

    record = {
        "id": query["query_id"],
        "messages": messages,
        "positive_messages": [
            [
                {
                    "role": "user",
                    "content": "<image>为图库图片编码，用于文搜图和图搜图召回。",
                }
            ]
        ],
        "positive_images": [[positive_doc["image"]]],
        "negative_messages": [
            [
                {
                    "role": "user",
                    "content": "<image>为图库图片编码，用于文搜图和图搜图召回。",
                }
            ]
            for doc in negative_docs
        ],
        "negative_images": [[doc["image"]] for doc in negative_docs],
        "meta": {
            "task": query["task"],
            "query_id": query["query_id"],
            "positive_doc_id": positive_doc["doc_id"],
            "negative_doc_ids": [doc["doc_id"] for doc in negative_docs],
            "family_id": query["family_id"],
            "split": query["split"],
        },
    }
    if images is not None:
        record["images"] = images
    return record


def build_dataset() -> None:
    corpus: list[dict] = []
    corpus_by_family: dict[str, list[dict]] = {}
    queries: list[dict] = []
    qrels: list[dict] = []

    for family_index, family in enumerate(FAMILIES):
        split = "train" if family_index < 6 else "val"
        family_docs: list[dict] = []
        for variant in range(5):
            image_path = OUT_DIR / "images" / "corpus" / f"{family['family_id']}_v{variant + 1}.png"
            create_image(family, image_path, variant)
            doc = {
                "doc_id": f"doc_{family['family_id']}_v{variant + 1}",
                "image": rel(image_path),
                "family_id": family["family_id"],
                "category": family["category"],
                "category_zh": family["category_zh"],
                "color_zh": family["color_zh"],
                "caption_zh": family["caption_zh"],
                "detail_zh": family["detail_zh"],
            }
            family_docs.append(doc)
            corpus.append(doc)
        corpus_by_family[family["family_id"]] = family_docs

        query_image_path = OUT_DIR / "images" / "query" / f"{family['family_id']}_query.png"
        create_image(family, query_image_path, 11 + family_index, query=True)

        for text_index, text in enumerate(family["text_queries"], start=1):
            qid = f"tq_{family['family_id']}_{text_index:02d}"
            queries.append(
                {
                    "query_id": qid,
                    "task": "text_to_image",
                    "modality": "text",
                    "split": split,
                    "instruction": "为文搜图检索编码这条文本查询",
                    "text": text,
                    "image": None,
                    "family_id": family["family_id"],
                    "positive_doc_ids": [doc["doc_id"] for doc in family_docs],
                    "hard_negative_doc_ids": [],
                }
            )

        qid = f"iq_{family['family_id']}_01"
        queries.append(
            {
                "query_id": qid,
                "task": "image_to_image",
                "modality": "image",
                "split": split,
                "instruction": "为图搜图检索编码这张查询图片",
                "text": None,
                "image": rel(query_image_path),
                "family_id": family["family_id"],
                "positive_doc_ids": [doc["doc_id"] for doc in family_docs],
                "hard_negative_doc_ids": [],
            }
        )

    doc_by_id = {doc["doc_id"]: doc for doc in corpus}

    for query in queries:
        family = next(f for f in FAMILIES if f["family_id"] == query["family_id"])
        query["hard_negative_doc_ids"] = sibling_hard_negatives(family, corpus_by_family)
        for rank, doc_id in enumerate(query["positive_doc_ids"], start=1):
            qrels.append(
                {
                    "query_id": query["query_id"],
                    "doc_id": doc_id,
                    "relevance": 1,
                    "match_type": "same_item_family",
                    "positive_rank_hint": rank,
                }
            )

    text_records: list[dict] = []
    image_records: list[dict] = []
    for query in queries:
        positives = [doc_by_id[doc_id] for doc_id in query["positive_doc_ids"]]
        negatives = [doc_by_id[doc_id] for doc_id in query["hard_negative_doc_ids"][:2]]
        pos = positives[0 if query["modality"] == "text" else 2]
        record = ms_swift_record(query, pos, negatives)
        if query["modality"] == "text":
            text_records.append(record)
        else:
            image_records.append(record)

    train_text = [row for row in text_records if row["meta"]["split"] == "train"]
    val_text = [row for row in text_records if row["meta"]["split"] == "val"]
    train_image = [row for row in image_records if row["meta"]["split"] == "train"]
    val_image = [row for row in image_records if row["meta"]["split"] == "val"]

    write_jsonl(OUT_DIR / "corpus.jsonl", corpus)
    write_jsonl(OUT_DIR / "queries.jsonl", queries)
    write_jsonl(OUT_DIR / "qrels.jsonl", qrels)
    write_jsonl(OUT_DIR / "train_text_to_image_ms_swift.jsonl", train_text)
    write_jsonl(OUT_DIR / "val_text_to_image_ms_swift.jsonl", val_text)
    write_jsonl(OUT_DIR / "train_image_to_image_ms_swift.jsonl", train_image)
    write_jsonl(OUT_DIR / "val_image_to_image_ms_swift.jsonl", val_image)
    write_jsonl(OUT_DIR / "train_embedding_combined_ms_swift.jsonl", train_text + train_image)
    write_jsonl(OUT_DIR / "val_embedding_combined_ms_swift.jsonl", val_text + val_image)

    manifest = {
        "name": "demo_retrieval",
        "purpose": "Tiny text-to-image and image-to-image retrieval training dataset.",
        "generated_by": rel(Path(__file__).resolve()),
        "counts": {
            "corpus_images": len(corpus),
            "query_images": len([q for q in queries if q["modality"] == "image"]),
            "queries": len(queries),
            "qrels": len(qrels),
            "train_records": len(train_text) + len(train_image),
            "val_records": len(val_text) + len(val_image),
        },
        "splits": {
            "train_families": [f["family_id"] for f in FAMILIES[:6]],
            "val_families": [f["family_id"] for f in FAMILIES[6:]],
        },
        "path_convention": "Image paths are relative to the repository root.",
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (OUT_DIR / "README.md").write_text(dataset_readme(manifest), encoding="utf-8", newline="\n")


def dataset_readme(manifest: dict) -> str:
    return f"""# Demo Retrieval Dataset

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
| `images/corpus/*.png` | 图库图片，共 {manifest["counts"]["corpus_images"]} 张 |
| `images/query/*.png` | 图搜图查询图片，共 {manifest["counts"]["query_images"]} 张 |
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
{{
  "query": "红色帆布双肩包，正面有白色竖条",
  "positives": ["同款/同实体/同语义族图片"],
  "hard_negatives": ["同类目但颜色、图案、型号或语义不同的图片"]
}}
```

如果一个 query 有多个等价正例，评测文件可以保留全部正例；训练文件可以把它们展开成多条样本，或在 sampler 里避免同一正例族互相成为 in-batch false negative。
"""


if __name__ == "__main__":
    build_dataset()
    print(f"Wrote dataset to {OUT_DIR}")
