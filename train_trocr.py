import torch
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Define custom dataset
class HandwritingDataset(Dataset):
    def __init__(self, image_dir, annotation_file, processor):
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx]
        image_path = os.path.join(self.image_dir, data["image"])
        image = Image.open(image_path).convert("RGB")
        text = data["text"]

        # Tokenize text & process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(text, padding="max_length", max_length=128, return_tensors="pt").input_ids.squeeze()
        
        return {"pixel_values": pixel_values, "labels": labels}

# Load processor & model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Dataset paths
image_dir = "path_to_images"
annotation_file = "annotations.json"

# Create dataset and dataloader
dataset = HandwritingDataset(image_dir, annotation_file, processor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_trained",
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs"
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train model
trainer.train()

# Save model & processor
model.save_pretrained("./trocr_trained")
processor.save_pretrained("./trocr_trained")

