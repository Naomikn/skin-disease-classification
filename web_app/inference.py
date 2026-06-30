import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import timm

# ── Class names — must match training folder sort order exactly ────────────────
LABELS = [
    "Eczema",
    "Warts / Molluscum",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Benign Keratosis",
    "Psoriasis / Lichen Planus",
    "Seborrheic Keratoses",
    "Tinea / Candidiasis",
]

NUM_CLASSES = len(LABELS)

# ── Risk classification ────────────────────────────────────────────────────────
HIGH_RISK   = ["Melanoma", "Basal Cell Carcinoma"]
MEDIUM_RISK = ["Eczema", "Atopic Dermatitis", "Psoriasis / Lichen Planus"]

def get_risk_level(class_name):
    if class_name in HIGH_RISK:   return "HIGH"
    if class_name in MEDIUM_RISK: return "MEDIUM"
    return "LOW"

def get_recommendation(class_name, confidence):
    risk = get_risk_level(class_name)
    if risk == "HIGH":
        return "Urgent dermatology referral recommended."
    if risk == "MEDIUM" and confidence > 80.0:
        return "Routine dermatology consultation recommended."
    if confidence < 50.0:
        return "Low confidence prediction. In-person assessment recommended."
    return "Monitor symptoms and consult a GP if condition persists."

# ── Model architecture ─────────────────────────────────────────────────────────
SWIN_CANDIDATES = [
    "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_small_patch4_window7_224.ms_in1k",
    "swin_small_patch4_window7_224",
]
CONVNEXT_CANDIDATES = [
    "convnext_small.fb_in22k_ft_in1k",
    "convnext_small.fb_in1k",
    "convnext_small",
]

def resolve_model_name(candidates):
    for name in candidates:
        if timm.is_model(name):
            return name
    return candidates[-1]

SWIN_MODEL_NAME     = resolve_model_name(SWIN_CANDIDATES)
CONVNEXT_MODEL_NAME = resolve_model_name(CONVNEXT_CANDIDATES)


class AttentionalFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.Sigmoid()
        )
    def forward(self, f_conv, f_swin):
        alpha = self.gate(torch.cat([f_conv, f_swin], dim=1))
        return alpha * f_conv + (1 - alpha) * f_swin


class ConvNeXtSwinHybrid(nn.Module):
    def __init__(self, num_classes, dropout=0.3, feat_dim=768):
        super().__init__()
        self.convnext = timm.create_model(
            CONVNEXT_MODEL_NAME, pretrained=False,
            num_classes=0, global_pool="avg"
        )
        self.swin = timm.create_model(
            SWIN_MODEL_NAME, pretrained=False,
            num_classes=0, global_pool="avg"
        )
        self.convnext.eval(); self.swin.eval()
        with torch.no_grad():
            dummy  = torch.zeros(1, 3, 224, 224)
            conv_d = self.convnext(dummy).shape[1]
            swin_d = self.swin(dummy).shape[1]
        self.conv_proj = (nn.Identity() if conv_d == feat_dim
                          else nn.Linear(conv_d, feat_dim))
        self.swin_proj = (nn.Identity() if swin_d == feat_dim
                          else nn.Linear(swin_d, feat_dim))
        self.aff = AttentionalFusion(feat_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, 512), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(512, 256), nn.GELU(),
            nn.Dropout(dropout / 4), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        f_conv = self.conv_proj(self.convnext(x))
        f_swin = self.swin_proj(self.swin(x))
        return self.classifier(self.aff(f_conv, f_swin))


# ── Image preprocessing ────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Load model once at startup ─────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "best_model_hybrid.pth"

print(f"Loading model on {DEVICE}...")
MODEL = ConvNeXtSwinHybrid(num_classes=NUM_CLASSES).to(DEVICE)
MODEL.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
MODEL.eval()
print("✅ Model loaded and ready.")


# ── Prediction function ────────────────────────────────────────────────────────
def predict_image(image_bytes: bytes):
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(MODEL(tensor), dim=1)[0].cpu().numpy()

    preds_pct = (probs * 100.0).astype(float)
    idx       = int(np.argmax(preds_pct))
    order     = np.argsort(-preds_pct)

    top_class  = LABELS[idx]
    confidence = float(preds_pct[idx])

    top_5 = [
        {
            "class":      LABELS[int(i)],
            "confidence": round(float(preds_pct[int(i)]), 2),
            "risk":       get_risk_level(LABELS[int(i)])
        }
        for i in order[:5]
    ]

    return {
        "predicted_class":    top_class,
        "confidence_percent": round(confidence, 2),
        "risk_level":         get_risk_level(top_class),
        "recommendation":     get_recommendation(top_class, confidence),
        "top_5":              top_5,
        "disclaimer": (
            "This tool is for clinical decision support only and does not "
            "constitute a medical diagnosis. All predictions must be reviewed "
            "by a qualified healthcare professional before any clinical action."
        )
    }