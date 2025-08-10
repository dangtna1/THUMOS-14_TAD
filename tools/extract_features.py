import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_i3d import InceptionI3d
import torch.nn.functional as F


# ---- 1. Load frames from RGB directory ----
def load_rgb_frames_from_folder(folder_path, resize=(224, 224)):
    frame_names = sorted(os.listdir(folder_path))
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),  # Converts to [C, H, W] in [0, 1]
        ]
    )

    frames = []
    for fname in frame_names:
        img_path = os.path.join(folder_path, fname)
        image = Image.open(img_path).convert("RGB")
        image = transform(image)
        frames.append(image)

    # Pad if not enough frames
    # while len(frames) < max_frames:
    #     frames.append(frames[-1].clone())

    frames = torch.stack(frames, dim=1)  # [C, T, H, W]
    return frames.unsqueeze(0)  # [1, C, T, H, W]


# ---- 2. Extract features using I3D ----
def extract_i3d_features(rgb_tensor, model_path):
    model = InceptionI3d(157, in_channels=3)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.replace_logits(157)  # for Charades: CHANGE?
    model.eval()

    with torch.no_grad():
        feats = model.extract_features(rgb_tensor)  # [1, 1024, T', 7, 7]
        print("Raw shape:", feats.shape)

        # Optional: spatial average pooling to get [T', 1024]
        feats = F.adaptive_avg_pool3d(feats, (None, 1, 1))  # [1, 1024, T', 1, 1]
        feats = feats.squeeze(-1).squeeze(-1).permute(2, 0, 1).squeeze(1)  # [T', 1024]
        print("Pooled shape:", feats.shape)

    return feats.cpu().numpy()


# ---- 3. Run on one example ----
if __name__ == "__main__":
    rgb_folder = "data/charades/CharadesEgo_v1_rgb/CJRJWEGO"  # <- a folder of .jpg or .png frames
    i3d_weights = "models/rgb_charades.pt"

    rgb_tensor = load_rgb_frames_from_folder(rgb_folder)
    features = extract_i3d_features(rgb_tensor, i3d_weights)

    # Save if needed
    np.save("CJRJWEGO.npy", features)
