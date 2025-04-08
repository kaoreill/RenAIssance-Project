import os
import unicodedata
import editdistance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class OCRDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_text_length=80):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.transcription_dir = os.path.join(root_dir, "transcriptions")
        self.transform = transform
        self.image_names = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.max_text_length = max_text_length
        self.valid_chars = {'ñ', '¯'}  # Special characters to handle
        self.char2idx = {'<blank>': 0, '<pad>': 1, '<unk>': 2}
        self._build_vocab()

    def _get_text_filename(self, img_filename):
        """Convert image filename to corresponding text filename"""
        base = os.path.splitext(img_filename)[0]  # Removes .jpg/.png
        return f"{base}.txt"

    def _normalize_text(self, text):
        """Apply historical text normalization rules"""
        # Character replacements
        text = text.lower()
        text = text.replace('v', 'u').replace('ſ', 's').replace('ç', 'z')
        text = self._expand_macrons(text)
        text = self._remove_accents(text)
        
        # Handle macrons
        text = self._expand_macrons(text)
        
        # Remove unwanted accents
        text = self._remove_accents(text)
        
        text = text.replace("’", "'")  # Normalize apostrophes
        # Handle hyphens
        if text.endswith('-'):
            text = text[:-1]
            
        return text

    def _expand_macrons(self, text):
        """Expand macron characters according to rules"""
        expanded = []
        text_chars = list(text)
        while text_chars:
            char = text_chars.pop(0)
            if char == '¯':
                if expanded and expanded[-1] == 'q':
                    expanded.append('ue')
                else:
                    expanded.append('n')
            else:
                expanded.append(char)
        return ''.join(expanded)

    def _remove_accents(self, text):
        """Remove accents except ñ and handled macrons"""
        cleaned = []
        for c in text:
            if c == 'ñ':
                cleaned.append(c)
            else:
                nfkd = unicodedata.normalize('NFKD', c)
                cleaned.append(nfkd[0])
        return ''.join(cleaned)

    def _build_vocab(self):
        """Build vocabulary from normalized text in transcriptions"""
        all_text = []  # MUST BE INITIALIZED HERE
        
        for img_name in self.image_names:
            txt_filename = self._get_text_filename(img_name)
            txt_path = os.path.join(self.transcription_dir, txt_filename)
            
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    normalized_text = self._normalize_text(text)
                    all_text.extend(list(normalized_text))
            except FileNotFoundError:
                continue

        # Create vocabulary with special tokens
        unique_chars = {'<blank>', '<pad>', '<unk>'}
        unique_chars.update(set(all_text))  # Now works because all_text exists
        
        # Create mapping dictionaries
        self.char2idx = {'<blank>': 0, '<pad>': 1, '<unk>': 2}
        idx = 3
        for char in sorted(set(all_text)):
            if char not in self.char2idx:
                self.char2idx[char] = idx
                idx += 1
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_filename = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        txt_path = os.path.join(self.transcription_dir, self._get_text_filename(img_filename))

        try:
            # Load image
            image = Image.open(img_path).convert("L")  # Convert to grayscale

            # Load and normalize text
            with open(txt_path, 'r', encoding='utf-8') as f:
                raw_text = f.read().strip()
            
            # Apply normalization pipeline
            normalized_text = self._normalize_text(raw_text)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing paired files:\nImage: {img_path}\nText: {txt_path}"
            )

        # Convert text to indices
        text_indices = [
            self.char2idx.get(c, self.char2idx['<unk>'])
            for c in normalized_text
        ]
        
        # Pad/truncate to max length
        text_length = len(text_indices)
        if text_length > self.max_text_length:
            text_indices = text_indices[:self.max_text_length]
            text_length = self.max_text_length
        else:
            padding = [self.char2idx['<pad>']] * (self.max_text_length - text_length)
            text_indices += padding

        # Apply image transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        return {
            'image': image,
            'text': torch.tensor(text_indices).long(),
            'length': torch.tensor(text_length).long()
        }
import torch
import torch.nn as nn

class OCRModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, nhead=8):
        super(OCRModel, self).__init__()

        # CNN Backbone (modified for proper dimension flow)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
        )

        # Calculate the output features after the CNN layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 128, 512)  # Match input size
            dummy_out = self.cnn(dummy)
            dummy_out = dummy_out.permute(0, 3, 2, 1)  # [B, W, H, C]
            features = dummy_out.size(2) * dummy_out.size(3)  # H*C

        # Linear layer for dimension reduction
        self.linear = nn.Linear(features, 256)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Output layer that maps LSTM output to vocab size
        self.output = nn.Linear(hidden_size * 2, vocab_size)
        self.blank_bias = 1  # Bias for the blank token

    def forward(self, x):
        # CNN Feature extraction
        x = self.cnn(x)  # [B, C, H, W]
        
        # Reshape for LSTM
        B, C, H, W = x.size()
        x = x.permute(0, 3, 2, 1)  # [B, W, H, C]
        x = x.reshape(B, W, H * C)  # [B, W, H*C]
        
        # Linear dimension reduction
        x = self.linear(x)  # Now correctly sized [B, W, 256]
        
        # LSTM processing
        x, _ = self.lstm(x)  # LSTM output [B, T, hidden_size*2]
        
        # Final output projection (mapping LSTM output to vocab space)
        logits = self.output(x)  # [B, T, vocab_size]
        
        # Apply bias to blank token logits (adjust blank token probability)
        logits[:, :, 0] = logits[:, :, 0] - self.blank_bias  # Subtract bias from blank token logits

        print(f"Model output shape: {logits.shape}")  # Should be (B, T, vocab_size)
        
        return logits


import torch
import torch.nn as nn

def compute_loss(preds, targets, target_lengths):
    # Permute preds to match the shape (T, N, C)
    preds = preds.permute(1, 0, 2)  # [B, T, C] -> [T, B, C]
    batch_size = preds.size(0)
    
    input_lengths = torch.full((batch_size,), preds.size(0), dtype=torch.long)  # Batch size
    loss = nn.CTCLoss(blank=0)(preds, targets, input_lengths, target_lengths)
    
    # Apply weight adjustment to the loss based on target lengths
    weights = 1.0 + 3.0 * (target_lengths.float() / max(target_lengths))  # Adjust multiplier as needed
    loss = (loss * weights).mean()

    return loss


def compute_loss_with_blank_penalty(preds, targets, target_lengths, lambda_reg=0.5):
    # Permute preds to match the shape (T, N, C)
    preds = preds.permute(1, 0, 2)  # [B, T, C] -> [T, B, C]
    
    # Compute standard CTC loss
    input_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long)  # Batch size
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)(preds, targets, input_lengths, target_lengths)
    
    # Convert log-probabilities to probabilities and compute mean probability for blank token
    blank_prob = preds.exp()[:, :, 0].mean()  # Get probability for the blank token (index 0)
    
    # Penalize high blank probabilities (you may need to tune lambda_reg)
    reg_loss = lambda_reg * blank_prob
    
    # Return the combined CTC loss with blank penalty
    return ctc_loss + reg_loss


def decode_raw_predictions(preds, idx2char):
    """
    Decodes predictions by simply taking the argmax at each time step,
    without removing blank tokens or repeated characters.
    """
    _, max_indices = torch.max(preds, dim=-1)
    batch_texts = []
    for indices in max_indices:
        chars = [idx2char.get(idx.item(), '<unk>') for idx in indices]
        batch_texts.append(''.join(chars))
    return batch_texts

def decode_predictions(preds, idx2char):
    _, max_indices = torch.max(preds, dim=-1)
    batch_texts = []
    for indices in max_indices:
        text = []
        previous_idx = None
        for idx in indices:
            idx = idx.item()
            if idx == 0:  # <blank>
                previous_idx = idx
                continue
            if idx == previous_idx:
                continue
            char = idx2char.get(idx, '<unk>')
            if char != '<pad>':
                text.append(char)
            previous_idx = idx
        batch_texts.append(''.join(text))
    return batch_texts


def cer_score(true_texts, pred_texts):
    total_errors = 0
    total_chars = 0
    for true, pred in zip(true_texts, pred_texts):
        total_errors += editdistance.eval(true, pred)
        total_chars += len(true)
    return total_errors / total_chars if total_chars > 0 else 0

def decode_targets(targets, idx2char):
    """Directly convert target indices to text without max operation"""
    batch_texts = []
    for target in targets:
        chars = [idx2char.get(idx.item(), '<unk>') for idx in target]
        # Remove padding
        text = ''.join([c for c in chars if c != '<pad>'])
        batch_texts.append(text)
    return batch_texts

def greedy_decode(preds, idx2char, blank_idx=0):
    """
    Greedily decodes the predictions by taking the argmax of logits, 
    removing repeated characters, and stripping out blanks.
    
    Args:
        preds (Tensor): The output logits from the model (B, T, C)
        blank_idx (int): The index of the blank token in the vocab.
        
    Returns:
        decoded_texts (List[str]): The list of decoded strings for each sample in the batch.
    """
    batch_size = preds.size(0)
    decoded_texts = []
    
    for b in range(batch_size):
        # Get the most probable token at each time step
        pred = torch.argmax(preds[b], dim=-1)  # (T,)
        
        # Remove the blank tokens (index 0) and strip repeated characters
        decoded = []
        prev_char = -1
        
        for char in pred:
            if char.item() == blank_idx:
                continue  # Skip blanks
            if char.item() != prev_char:  # Remove repeated characters
                decoded.append(char.item())
            prev_char = char.item()
        
        # Convert to string using the idx2char mapping
        decoded_text = ''.join([idx2char[i] for i in decoded])
        decoded_texts.append(decoded_text)
    
    return decoded_texts


def train_ocr(model, dataloader, optimizer, scheduler, num_epochs=10, device="cuda"):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        all_preds = []
        all_targets = []
        blank_probs_all = []

        # Loop over the DataLoader (which yields dictionaries)
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['text'].to(device)
            target_lengths = batch['length'].to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            loss = compute_loss_with_blank_penalty(preds, targets, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()

            # Greedy Decoding
            output = model(images)  # (B, T, Vocab)
            log_probs = output.log_softmax(2)  # assuming you're using nn.CTCLoss

            # Apply greedy decoding (strip repeats and blanks)
            decoded = greedy_decode(log_probs, idx2char=dataloader.dataset.idx2char, blank_idx=0)
            print("Greedy Decoded Sample (first in batch):")
            print(decoded[0])

            # Compute average blank probability for debugging
            probs = preds.exp()  # Convert log-softmax to probabilities
            blank_probs = probs[:, :, 0]  # blank token is at index 0
            blank_probs_all.append(blank_probs.mean().item())
            
            # Decode predictions and targets for CER calculation
            with torch.no_grad():
                processed_pred_texts = decode_predictions(preds.cpu(), dataloader.dataset.idx2char)
                true_texts = decode_targets(targets.cpu(), dataloader.dataset.idx2char)
                all_preds.extend(processed_pred_texts)
                all_targets.extend(true_texts)

        avg_loss = epoch_loss / len(dataloader)
        avg_blank_prob = sum(blank_probs_all) / len(blank_probs_all)
        cer = cer_score(all_targets, all_preds)
        
        # Update scheduler based on CER
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(cer)
        else:
            scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, CER: {cer:.4f}, Avg Blank Prob: {avg_blank_prob:.4f}")

        # Print sample predictions every 5 epochs for further debugging
        if epoch % 5 == 0:
            with torch.no_grad():
                # Using the last batch's predictions as samples
                sample_preds = preds[:3].cpu()  # take first 3 examples from the last batch
                raw_pred_texts = decode_raw_predictions(sample_preds, dataloader.dataset.idx2char)
                processed_pred_texts = decode_predictions(sample_preds, dataloader.dataset.idx2char)
                sample_true_texts = decode_targets(targets[:3].cpu(), dataloader.dataset.idx2char)
                print("\nSample Predictions:")
                for i in range(len(sample_true_texts)):
                    print(f"True: {sample_true_texts[i]}")
                    print(f"Processed Pred: {processed_pred_texts[i]}")
                    print(f"Raw Pred: {raw_pred_texts[i]}\n")

def main():
    root_dir = "./data"
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

   
    transform = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = OCRDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, pin_memory=True if device=='cuda' else False)
    
    print("Vocabulary:", dataset.char2idx)
    print("Vocab size:", len(dataset.char2idx))


    model = OCRModel(vocab_size=len(dataset.char2idx)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,  # Maximum learning rate
        steps_per_epoch=len(dataloader),
        epochs=20
    )
    


    train_ocr(model, dataloader, optimizer, scheduler, num_epochs=num_epochs, device=device)

if __name__ == "__main__":
    main()