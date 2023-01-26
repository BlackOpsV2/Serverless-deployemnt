from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from captum.robust import FGSM
from PIL import Image

IMG_SIZE: Tuple[int, int] = (224, 224)
MEAN: List[float] = [0.485, 0.456, 0.406]
STD: List[float] = [0.229, 0.224, 0.225]

augs: Dict = {
    "gaussian_noise": A.GaussNoise(always_apply=True, mean=MEAN),
    "random_brightness": A.RandomBrightness(always_apply=True, limit=0.7),
    "pixel_dropout": A.CoarseDropout(
        max_holes=8,
        max_height=128,
        max_width=128,
        min_holes=8,
        min_height=128,
        min_width=128,
        always_apply=True,
        fill_value=MEAN,
    ),
}
apply_fgsm = True

transform: T.Compose = T.Compose(
    [T.Resize(IMG_SIZE), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
)
inv_transform: T.Compose = T.Compose(
    [
        T.Normalize(
            mean=(-1 * np.array(MEAN) / np.array(STD)).tolist(),
            std=(1 / np.array(STD)).tolist(),
        ),
    ]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
outdir = Path("images/robust")


def inference(model: torch.nn.Module, img_tensor: torch.Tensor) -> Tuple[str, float, int]:
    with torch.no_grad():
        logits = model(img_tensor)
        preds = F.softmax(logits, dim=-1)

    prediction_score, pred_label_idx = torch.topk(preds, 1)
    predicted_label = model.idx_to_class[pred_label_idx.item()]

    return predicted_label, prediction_score.squeeze().item(), pred_label_idx.item()


def save_plot(img1: np.ndarray, img2: np.ndarray, suptitle: str, title1: str, title2: str, savepath: Path):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis("off")
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis("off")
    plt.title(title2)

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(savepath)


def main(model_path: str, images: str) -> None:
    images = Path(images)
    sources = [images] if images.is_file() else list(images.glob("*"))
    print("Total images:: ", len(sources), "\n-> ", sources)

    model = torch.jit.load(model_path)
    model = model.to(device)
    model.eval()

    fgsm = FGSM(model, lower_bound=-1, upper_bound=1)
    
    for image_path in sources:
        print("=====:: ", image_path)
        out_path = outdir / image_path.stem
        out_path.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGB")
        transformed_img = transform(img).unsqueeze(0).to(device)
        pred_label, score, idx = inference(model, transformed_img)
        print(pred_label, score)
        
        for aug_name, aug in augs.items():            
            augmented_image = aug(image=np.array(img))["image"]
            transformed_aug_img = transform(Image.fromarray(augmented_image)).unsqueeze(0).to(device)

            aug_pred_label, aug_score, _ = inference(model, transformed_aug_img)
            print("\t:: ", aug_name, "|", aug_pred_label, aug_score)
            
            save_plot(np.array(img),  augmented_image, aug_name, title1="Original:\n%s | %s" % (pred_label, "%.4f" % score), 
                      title2="augmented:\n%s | %s" % (aug_pred_label, "%.4f" % aug_score), savepath=out_path / f"{aug_name}.png")
        
        if apply_fgsm:
            transformed_aug_img = fgsm.perturb(
                transformed_img, epsilon=0.16, target=idx
            )

            augmented_image = inv_transform(transformed_aug_img).squeeze().permute(1, 2, 0).detach().cpu().numpy()
            aug_pred_label, aug_score, _ = inference(model, transformed_aug_img)
            print("\t:: FGSM ", '|', aug_pred_label, aug_score)

            save_plot(np.array(img),  augmented_image, "FGSM", title1="Original:\n%s | %s" % (pred_label, "%.4f" % score), 
                      title2="augmented:\n%s | %s" % (aug_pred_label, "%.4f" % aug_score), savepath=out_path / "fgsm.png")


if __name__ == "__main__":
    main("serverless/model.scripted.pt", "images/examples")
