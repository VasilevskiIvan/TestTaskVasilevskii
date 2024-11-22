import asyncio
import logging
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Router
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

logging.basicConfig(level=logging.INFO)

load_dotenv()
bot = Bot(os.getenv('TOKEN'))

storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()

# Пути к весам моделей
resnet_model_path = "resnet34_w.pth"
mycnn_model_path = "MyCNN_w1(2).pth"
myresnet_model_path = "MyResNet_w.pth"

# Классы
class_names = ['гусь', 'индюк', 'курица', 'петух', 'страус', 'утка', 'цыпленок']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация трансформации для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Инициализация бота
@router.message(Command("start"))
async def start_handler(message: types.Message):
    markup = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text="Собственная CNN")],
            [types.KeyboardButton(text="Дообученный ResNet")],
            [types.KeyboardButton(text="Собственный аналог ResNet")],
        ], resize_keyboard=True
    )
    await message.reply("Привет! Выберите модель для классификации:", reply_markup=markup)

# Выбор модели
@router.message(lambda message: message.text in ["Собственная CNN", "Дообученный ResNet", "Собственный аналог ResNet"])
async def model_selection_handler(message: types.Message):
    global model, transform
    selected_model = message.text

    if selected_model == "Собственная CNN":
        # Для MyCNN
        model = MyCNN(len(class_names))
        model.load_state_dict(torch.load(mycnn_model_path, map_location=device))
        model = model.to(device)
        await message.reply("Вы выбрали модель: Собственная CNN. "
                            "Она имеет точность около 0.95, однако подвержена некоторому влиянию фона и птицы, \
желательно, должны располагаться достаточно близко")

    elif selected_model == "Дообученный ResNet":
        model = models.resnet34(pretrained=False)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, len(class_names))
        model.load_state_dict(torch.load(resnet_model_path, map_location=device))
        model = model.to(device)
        model.eval()
        await message.reply("Вы выбрали модель: Дообученный ResNet. "
                            "Она имеет точность около 1 и в действительно классифицирует изображения почти идеально")

    elif selected_model == "Собственный аналог ResNet":
        model = MyResNet(len(class_names))
        model.load_state_dict(torch.load(myresnet_model_path, map_location=device))
        model = model.to(device)
        model.eval()
        await message.reply("Вы выбрали модель: Собственный аналог ResNet. "
                            "Она имеет точность около 0.95, однако подвержена некоторому влиянию фона и может путать \
некоторые схожие виды птиц друг с другом (например, курицу и петуха или гуся и утку)")

    markup = types.ReplyKeyboardRemove()
    await message.reply("Теперь отправьте мне фото для классификации.", reply_markup=markup)

# Обработчик фотографий
@router.message(lambda message: message.photo)
async def photo_handler(message: types.Message):
    photo = message.photo[-1]
    file = await bot.download(photo.file_id)

    image = Image.open(file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]

    await message.reply(f"Птица на данной фотографии относится к классу: *{predicted_class}*", parse_mode="Markdown")

# Вспомогательные классы для CNN и ResNet
class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MyCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.34),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

class MyResNet(nn.Module):
    def __init__(self, num_classes):
        super(MyResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [ResidualBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

dp.include_router(router)


async def main():
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

try:
    asyncio.run(main())
except RuntimeError:
    import nest_asyncio

    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
