import pandas as pd
import numpy as np
import random
import transformers
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import SequentialSampler, RandomSampler
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

"""
Задача: выделить для каждого текста вступительную часть( то есть ту часть,
которую можно убрать без ущерба для понимания). Размеченных данных нет.
Есть только тексты и метка - спам или не спам.
Идея решения задачи заключается в том, что я обучаю модель Bert
для классификации текстов (спам или не спам). После смотрю на
attention score для [CLS] токена (который и отвечает за классификацию)
для каждого слова. В моём понимании attention score для [CLS] токена
можно проинтерпретировать как важность слова для понимания текста.
То есть если с самого начала предложения идут маловажные слова (то есть
с маленьким attention score), то их можно закинуть во вступление.
"""

df = pd.read_csv("/kaggle/input/email-data/emailTextsI.csv")
# Кодируем категориальный признак (чтобы можно было закидывать в модель)
df["label"] = df["label"].apply(lambda x: 1 if x == "Spam" else 0)
print("a")
print(df.head())
# В датасете есть некоторые строчки с NaN текстами. Их пока не будем учитывать
df_train = df[~df["text"].isna()]

sentences = df_train.text.values
labels = df_train.label.values
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

# Токенизируем текст
input_ids = []  # Здесь будем хранить id токенов каждого текста
attention_masks = []  # Здесь будем хранить маски внимания для каждого текста

for sent in tqdm(sentences):
    #   (1) Токенизация текста.
    #   (2) ставим [CLS] токен в начало.
    #   (3) ставим [SEP] токен в конец.
    #   (4) Ставим в соответствие токенов их id.
    #   (5) Заполняем пустое пространство [PAD] токенами до максимальной длины
    #       (это нужно, чтобы каждый токенизированный текст имел одинаковую длину)
    #   (6) Создаём маску для [PAD] токенов
    #       (чтобы настоящие слова и токен [CLS] не обращали внимание на пустоту)
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )

    input_ids.append(encoded_dict["input_ids"])

    attention_masks.append(encoded_dict["attention_mask"])

# Конвертируем всё в тензор
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Создаём датасет из id токенов, масок внимания и соответствующих меток о спаме
dataset = TensorDataset(input_ids, attention_masks, labels)

# 90% выборки - тренировочный датасет
# 10% выборки - валидационный датасет
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Будем обучать модель батчами размером 32
batch_size = 32

# Создаём тренировочный даталоадер
train_dataloader = DataLoader(
    train_dataset,
    # Перемешаем строчки для случайности
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size,
)

# # Создаём валидационный даталоадер
validation_dataloader = DataLoader(
    val_dataset,
    # Здесь нет необходимости перемешивать строчки
    sampler=SequentialSampler(val_dataset),
    batch_size=batch_size,
)

# Загрузим модель Bert
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=2,
    # Для нашей задачи важно, чтобы модель выдавала attention score
    output_attentions=True,
    output_hidden_states=False,
)

model.cuda()

# Реализуем transfer learning: заморозим все веса,
# кроме последнего слоя энкодера и всего, что после него
for param in model.parameters():
    param.requires_grad = False

for param in model.bert.encoder.layer[11].parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

for param in model.bert.pooler.parameters():
    param.requires_grad = True

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)


# Функция для подсчёта метрики accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


seed_val = 42
device = "cuda"
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
epochs = 1
training_stats = []
# Цикл обучения модели
for epoch_i in range(0, epochs):
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="epoch: {epoch_i}")):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        res = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )
        loss, logits = res.loss, res.logits
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    print("")
    print("Running Validation...")

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            res = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            loss, logits = res.loss, res.logits

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    training_stats.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Valid. Accur.": avg_val_accuracy,
        }
    )

print("")
print("Training complete!")


# Создадим даталоадер уже для всех имеющихся данных
batch_size = 16

dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

model.eval()
# в attns будем хранить attention score для каждого текста
# Причём только для [CLS] токена
attns = torch.tensor([])
with torch.no_grad():
    for inp_ids, attn_mask, _ in tqdm(dataloader):
        inputs = inp_ids.to("cuda")
        masks = attn_mask.to("cuda")
        out = model(inputs, masks)
        out.logits = out.logits.to("cpu")
        attns = torch.concat((attns, out.attentions[11][:, :, 0].to("cpu")))
        del out

# Возьмём средний по головам attention score
attns_mean = attns.mean(dim=1)

intros = []  # Здесь храним вступления для каждого текста
for i in tqdm(range(attns_mean.shape[0])):
    for j in range(1, attns_mean.shape[1]):
        # Если суммарный attention score перевалил за 0.1
        if attns_mean[i, :j].sum() > 0.1:
            if j == 1:
                # Если это произошло на [CLS] токене,
                # то будем считать, что вступления нет
                intros.append(np.nan)
            else:
                # Токен может быть частью слова, поэтому ошибочно будет
                # просто декодировать всё но нужного индекса
                # Если токены являются частью одного слова, то
                # они соединяются '###'.
                while tokenizer.decode(input_ids[i, j]).find("#") == 1:
                    # Идём до следующего слова (именно слова, а не токена)
                    j += 1
                intros.append(tokenizer.decode(input_ids[i, :j])[6:])
            break

df_train.loc[:, "introduction"] = intros
submission = df.copy()
# Помним, что в начале мы удалили строки с NaN текстами
# Их нужно вернуть
indexes = submission[
    (submission["introduction"].isna()) & (~submission["text"].isna())
].index
submission.loc[indexes, "introduction"] = df_train.loc[indexes, "introduction"]
submission[['id', 'introductions']].to_csv("introductions_vk.csv", index=False)

"""
Для улучшения результатов можно более точно настроить обучение модели
(Подобрать learning rate, поставить scheduler и так далее),
поставить побольше эпох (Здесь я обучал всего на одной эпохе, так как времени было не так много),
поиграться с порогом attention score, после которого мы откидываем левую часть во вступление.
Так же можно попробовать посмотреть модели для решения NER задач, которые, возможно,
лучше справятся с этой задачей. Идея с attention score мне показалась интересной, поэтому
я решил попробовать её реализовать.
"""
