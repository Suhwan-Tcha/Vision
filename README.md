# DL Project

# **Image Classification via EfficientNet**

### ■ Factors Considered in Model Development

- Model Variations
- Data Augmentation Techniques
- Batch Size Optimization
- Optimizer Selection
- LR Scheduling
- Regularization
- Fine Tuning
- Number of Epochs

### 1. Model Variations

Considering the computational cost and memory

EfficientNet B0 ~ B3 → B2 or B3

### 2. Data Augmentation Techniques

- test1 [RandomRotation, RandomResizedCrop, RandomAffine, ColorJitter]

```
  transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),

    #data augmentaton 적용
    transforms.RandomRotation(30),# 30도 이내의 랜덤 회전
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 80%~100% 크기에서 랜덤 자름
    transforms.RandomAffine(degrees=0, shear=20),  # 랜덤 기울이기
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # 색상 변화
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

![image.png](image.png)

![image.png](56b4a63d-e826-40f3-abe9-87c9ade15c4f.png)

- test2 [RandomRotation(lower rotation slope), RandomResizedCrop, ColorJitter(lower range)]

```
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),

    #data augmentaton 적용
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 80%~100% 크기에서 랜덤 자름
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # 색상 변화
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

![image.png](image%201.png)

![image.png](image%202.png)

- test3 [Cutmix Image]

![image.png](image%203.png)

- test4 [Cutmix Image(Reduced proportion of mixed images)]

![image.png](image%204.png)

- test5 [Original Image + Cutmix Image]

![image.png](image%205.png)

- test6 [MixUP]

![image.png](image%206.png)

Test 2에서 가장 좋은 학습률을 보임

CutMix 이미지 사용 시 학습률이 매우 저조했는데, 이는 Oxford-IIIT Pet Dataset이 주로 고양이와 개의 이미지로 구성되어 있어, CutMix가 클래스 간 경계를 모호하게 만들어 모델 학습을 저해한 것으로 판단됨

### 3. Batch Size Optimization

32 → 64 → 128

64로 결정

### 4. Optimizer Selection

- test1

```
optimizer = optim.Adam(model._fc.parameters(), lr=0.001)
```

- test2

```
optimizer = optim.AdamW(model._fc.parameters(), lr=0.001, weight_decay = 0.05)
```

- test3

```
optimizer = optim.AdamW(model._fc.parameters(), lr=0.001, weight_decay = 0.01)
```

- test4

```
optimizer = optim.RMSprop(model._fc.parameters(), lr=0.0005, weight_decay=0.01, momentum=0.9, alpha=0.95) 
```

- test5

```
optimizer = optim.SGD(model._fc.parameters(), lr=0.001, momentum=0.9, weight_decay=0.05)
```

RMSprop으로 결정

### 5. LR Scheduling

- test1

```jsx
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```

- test2

```jsx
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

- test3

```jsx
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
```

- test4

```jsx
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
```

StepLR로 결정

### 6. Regularization

- Adding Regularization to the last layer of EfficientNet

```
model._fc = nn.Sequential(
    nn.Dropout(p=0.2),  
    nn.Linear(model._fc.in_features, num_classes)
)
```

```
model._fc = nn.Sequential(
    nn.Dropout(p=0.3),  
    nn.Linear(model._fc.in_features, num_classes)
)
```

```
model._fc = nn.Sequential(
    nn.Dropout(p=0.5),  # Dropout 추가
    nn.Linear(model._fc.in_features, num_classes)
)
```

- Setting weight_decay in optimizer

```jsx
optimizer = optim.RMSprop(model._fc.parameters(), lr=0.0005, weight_decay=0.001, momentum=0.9, alpha=0.95) 
```

```jsx
optimizer = optim.RMSprop(model._fc.parameters(), lr=0.0005, weight_decay=0.01, momentum=0.9, alpha=0.95) 
```

```jsx
optimizer = optim.RMSprop(model._fc.parameters(), lr=0.0005, weight_decay=0.1, momentum=0.9, alpha=0.95) 
```

dropout = 0.3, weight_decay = 0.01로 결정

### 7. Fine Tuning

- test1 (모든 파라미터가 다시 학습되도록 설정)

```jsx
for param in model.parameters():
    param.requires_grad = True
```

- test2 (모든 파라미터 고정하고 마지막 Layer만 학습하게 설정)

```jsx
for param in model.parameters():
    param.requires_grad = False  # 모든 레이어의 가중치를 고정

# 예를 들어, 마지막 분류 레이어만 학습 가능하게 설정
for param in model._fc.parameters():
    param.requires_grad = True  # 마지막 분류 레이어의 가중치만 학습
```

test1로 결정

### 8. Number of Epochs

5, 10, 20 → 10으로 결정

<aside>
💡

**Final Selection Model**

- EfficientNet b2
- Data Augmentation : RandomRotation + RandomResizedCrop + ColorJitter
- Batch Size : 64
- optimizer : RMSprop(model._fc.parameters(), lr=0.0005, weight_decay=0.01, momentum=0.9, alpha=0.95)
- Schedular : StepLR(optimizer, step_size=7, gamma=0.1)
- Regularization : dropout = 0.3, weight_decay = 0.01
- Fine-tuned with all parameters being learned
- epoch= 10

![image.png](image%207.png)

</aside>