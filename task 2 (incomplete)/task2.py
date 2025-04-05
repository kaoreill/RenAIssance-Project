import os
import unicodedata
import editdistance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    logging,
    AutoConfig
)
from transformers import AutoTokenizer

logging.set_verbosity_error()

# Configuration
CONFIG = {
    "data_root": "./data",
    "layout_model_weights": "./layout_model.pth",
    "batch_size": 4,
    "ocr_learning_rate": 5e-5,
    "num_epochs": 10,
    "max_text_length": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_size": (384, 384),
    "trocr_model": "microsoft/trocr-small-handwritten"
}

class LayoutSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = deeplabv3_resnet50(weights='DEFAULT')  # Updated weights parameter
        self.base_model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        
    def forward(self, x):
        return self.model(x)['out']

class OCRDataset(Dataset):
    def __init__(self, root_dir, processor, layout_model=None, max_text_length=100):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.transcription_dir = os.path.join(root_dir, "transcriptions")
        self.processor = processor
        self.layout_model = layout_model
        self.max_text_length = max_text_length
        self.image_names = sorted([f for f in os.listdir(self.image_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Character vocabulary
        self.char2idx, self.idx2char = self._build_vocab()

    def _build_vocab(self):
        chars = set()
        for img_name in self.image_names:
            txt_path = os.path.join(self.transcription_dir, f"{os.path.splitext(img_name)[0]}.txt")
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = self._normalize_text(f.read())
                    chars.update(text)
            except FileNotFoundError:
                continue
        
        vocab = ['<pad>', '<unk>'] + sorted(chars)
        return {c:i for i,c in enumerate(vocab)}, {i:c for i,c in enumerate(vocab)}

    def _normalize_text(self, text):
        text = text.lower()
        replacements = {'v':'u', 'ſ':'s', 'ç':'z', '’':"'", '¯':'n'}
        for old, new in replacements.items():
            text = text.replace(old, new)
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        txt_path = os.path.join(self.transcription_dir, f"{os.path.splitext(img_name)[0]}.txt")
        
        # Process image
        image = Image.open(img_path).convert('RGB')
        if self.layout_model:
            with torch.no_grad():
                layout_input = transforms.Compose([
                    transforms.Resize(CONFIG['image_size']),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(image).unsqueeze(0).to(CONFIG['device'])
                layout_mask = torch.sigmoid(self.layout_model(layout_input))
                image = Image.fromarray((np.array(image) * (layout_mask.squeeze().cpu().numpy() > 0.5)[..., np.newaxis]))
        
        # Process text
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = self._normalize_text(f.read().strip())
        
        # Encode and process
        labels = [self.char2idx.get(c, 1) for c in text[:self.max_text_length]]
        labels += [0] * (self.max_text_length - len(labels))
        
        return {
            'pixel_values': self.processor(image, return_tensors="pt").pixel_values.squeeze(),
            'labels': torch.LongTensor(labels),
            'text_length': torch.tensor(len(text))
        }

class OCRSystem:
    def __init__(self):
        # Initialize layout model
        self.layout_model = self._init_layout_model()
        
        # Load processor with explicit tokenizer configuration
        self.processor = TrOCRProcessor.from_pretrained(
            CONFIG['trocr_model'],
            use_fast=False,  # Disable fast tokenizer
            tokenizer=AutoTokenizer.from_pretrained(
                CONFIG['trocr_model'],
                use_fast=False,
                tokenizer_type="xlmtokenizer"  # Force specific tokenizer type
            )
        )
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
            CONFIG['trocr_model']
        ).to(CONFIG['device'])

    def _init_layout_model(self):
        model = LayoutSegmentationModel().to(CONFIG['device'])
        try:
            model.load_state_dict(torch.load(CONFIG['layout_model_weights'], map_location=CONFIG['device']))
        except FileNotFoundError:
            print("Using randomly initialized layout model")
        return model.eval()

    def train(self):
        dataset = OCRDataset(CONFIG['data_root'], self.processor, self.layout_model)
        dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        
        optimizer = optim.AdamW(self.ocr_model.parameters(), lr=CONFIG['ocr_learning_rate'])
        
        for epoch in range(CONFIG['num_epochs']):
            self.ocr_model.train()
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = self.ocr_model(
                    pixel_values=batch['pixel_values'].to(CONFIG['device']),
                    labels=batch['labels'].to(CONFIG['device'])
                )
                outputs.loss.backward()
                optimizer.step()
                total_loss += outputs.loss.item()
            
            print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} - Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    ocr_system = OCRSystem()
    ocr_system.train()