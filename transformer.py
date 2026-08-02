import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm

# ===============================
# Configuration
# ===============================

DATASET_PATH = "dataset"

IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42
NUM_EPOCHS = 5

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# ===============================
# Image Processor
# ===============================

processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

def to_rgb(img):
    # Must be a top-level named function, not a lambda -- Windows
    # DataLoader workers use pickling to receive the transform, and
    # lambdas cannot be pickled on Windows (this is fine on Linux/Mac,
    # which is why it didn't show up until num_workers>0 on Windows).
    return img.convert("RGB")


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(to_rgb),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=processor.image_mean,
        std=processor.image_std
    )
])


def main():
    # ===============================
    # Sanity check the dataset path
    # ===============================

    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(
            f"DATASET_PATH '{DATASET_PATH}' does not exist relative to "
            f"the current working directory ({os.getcwd()}). "
            f"Fix the path or run the script from the right folder."
        )

    # ===============================
    # Load Dataset
    # ===============================

    dataset = datasets.ImageFolder(
        root=DATASET_PATH,
        transform=transform
    )

    print("Classes :", dataset.classes)
    print("Total Images :", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError(
            f"ImageFolder found 0 images under '{DATASET_PATH}'. "
            f"Expected a structure like dataset/<class_name>/*.jpg"
        )

    if len(dataset.classes) != 2:
        print(
            f"WARNING: model is configured for num_labels=2 but found "
            f"{len(dataset.classes)} classes: {dataset.classes}"
        )

    # ===============================
    # Train / Validation / Test Split
    # ===============================

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    print("Training :", len(train_dataset))
    print("Validation :", len(val_dataset))
    print("Testing :", len(test_dataset))

    # Fail loudly instead of silently doing 0 iterations later.
    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training split is empty. Your dataset is too small for an "
            "80/10/10 split, or the split ratios need adjusting."
        )
    if len(val_dataset) == 0:
        raise RuntimeError(
            "Validation split is empty. Add more images or adjust the "
            "split ratios."
        )

    # ===============================
    # DataLoaders
    # ===============================

    # num_workers>0 lets image loading/resizing happen in parallel
    # worker processes instead of blocking the main training loop.
    # This matters a lot with 40k images -- num_workers=0 was likely why
    # the first epoch appeared to hang with no output.
    NUM_WORKERS = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0
    )

    print("Data Loaders Created Successfully")

    # =====================================
    # Load Pretrained Vision Transformer
    # =====================================

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    model.to(device)

    # Freeze the pretrained backbone
    for param in model.vit.parameters():
        param.requires_grad = False

    print("Training classifier only...")

    # ===============================
    # Loss Function
    # ===============================

    criterion = nn.CrossEntropyLoss()

    # ===============================
    # Optimizer
    # ===============================

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=3e-5,
        weight_decay=0.01
    )

    # ===============================
    # Learning Rate Scheduler
    # ===============================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2
    )

    best_accuracy = 0

    # ===============================
    # Training Loop (with per-epoch validation)
    # ===============================

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            leave=True
        )

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(pixel_values=images).logits

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

            # live per-batch feedback so it never looks "stuck"
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100*correct/total:.2f}%"
            )

        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Loss: {avg_train_loss:.4f} "
            f"Accuracy: {train_acc:.2f}%"
        )

        # ---- Validation (now runs every epoch) ----
        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = outputs.max(1)

                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"           Val Loss: {avg_val_loss:.4f} Val Accuracy: {val_acc:.2f}%")

        # Feed the scheduler -- this was never called before.
        scheduler.step(avg_val_loss)

        # Save the best checkpoint instead of only the last one.
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), "vit_microcrack.pth")
            print(f"           New best model saved (val acc {val_acc:.2f}%)")

    print(f"\nBest Validation Accuracy: {best_accuracy:.2f}%")
    print("Training complete.")


if __name__ == "__main__":
    main()