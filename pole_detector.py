import numpy as np
from ultralytics import YOLO
from tkinter import Tk, filedialog
from PIL import Image
import torch
import cv2
from transformers import DPTImageProcessor, DPTForDepthEstimation, CLIPProcessor, CLIPModel
device = "cpu"

def getTextSize(text):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    return text_size

def putText(img, text, org):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

class PoleModel:
    def __init__(self):
        self.depth_feat = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)
        self.depth_model.eval()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        print("Models loaded")

    def train(self):
        model = YOLO("y11.pt", task="detect")
        model.train(
            data = "data.yaml",
            epochs = 5,
            imgsz = 256,
            batch = 4,
            device = 'cpu',
            lr0 = 1e-3,
            optimizer= "Adam",
            patience = 2,
            augment = False,
            pretrained = True,
            exist_ok = True,
            resume=True,
        )

    def detect(self, image):
        model = YOLO("y8.pt")
        res = model.predict(image)
        return res[0]
    
    def classify_material(self, img_crop):
        candidate_labels = ["concrete", "metal", "wood", "plastic", "unknown"]
        inputs = self.clip_processor(img_crop, text=candidate_labels, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = self.clip_model(**inputs)
            logits = out.logits_per_image
            probs = logits.softmax(dim=1).cpu().numpy()[0]
            results = [(label,float(prob)) for label,prob in zip(candidate_labels,probs)]
            results = sorted(results, key=lambda x: x[1], reverse=True)
        return results
    
    def get_depthMap(self, img):
        h,w = img.height, img.width
        inputs = self.depth_feat(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
        depth = outputs.predicted_depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h,w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = depth.cpu().numpy().astype(np.float32)
        depth_map = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        #cv2.imshow("DP", depth_map)
        return depth_map
    
    def pipeline(self, source):
        source = Image.open(source).convert("RGB")
        detected = self.detect(source)
        img = detected.plot()
        bboxes = detected.boxes.xyxy.tolist()
        depth_map = self.get_depthMap(source)
        results = []
        for bbox in bboxes:
            x1,y1,x2,y2 = bbox
            cx,cy = int((x1+x2)/2), int(y1)+30
            cropped = source.crop((x1,y1,x2,y2))

            mats = self.classify_material(cropped)
            text = f"Material: {mats[0][0]}"
            tw,th = getTextSize(text)
            putText(img, text, (cx - (tw//2), cy + (th//2)))
            text = f"Probability: {mats[0][1]*100:0.2f}%"
            tw,th = getTextSize(text)
            putText(img, text, (cx - (tw//2), cy + (th//2)+30))
            
            height = self.estimate_pole_height(depth_map, bbox)
            if height and height["estimated_height"]:
                text = f"Height: {height['estimated_height']:0.2f}m"
                tw,th = getTextSize(text)
                putText(img, text, (cx - (tw//2), cy + (th//2)+80))
                text = f"Confidence: {height['confidence']*100:0.2f}%"
                tw,th = getTextSize(text)
                putText(img, text, (cx - (tw//2), cy + (th//2)+110))
            results.append({"bbox_xyxy":bbox, "material":mats[0], "material_probabilities":mats, "height":height})
        return img, results
    
    def pipeline_stream(self, source):
        source = Image.open(source).convert("RGB")
        detected = self.detect(source)
        img = detected.plot()
        yield({"stage":"detection", "image":img.copy()})
        bboxes = detected.boxes.xyxy.tolist()
        depth_map = self.get_depthMap(source)
        for bbox in bboxes:
            x1,y1,x2,y2 = bbox
            cx,cy = int((x1+x2)/2), int(y1)+30
            cropped = source.crop((x1,y1,x2,y2))

            mats = self.classify_material(cropped)
            text = f"Material: {mats[0][0]}"
            tw,th = getTextSize(text)
            putText(img, text, (cx - (tw//2), cy + (th//2)))
            text = f"Probability: {mats[0][1]*100:0.2f}%"
            tw,th = getTextSize(text)
            putText(img, text, (cx - (tw//2), cy + (th//2)+30))
            yield {"stage":"classify", "image":img.copy()}
            
            height = self.estimate_pole_height(depth_map, bbox)
            if height and height["estimated_height"]:
                text = f"Height: {height['estimated_height']:0.2f}m"
                tw,th = getTextSize(text)
                putText(img, text, (cx - (tw//2), cy + (th//2)+80))
                text = f"Confidence: {height['confidence']*100:0.2f}%"
                tw,th = getTextSize(text)
                putText(img, text, (cx - (tw//2), cy + (th//2)+110))
            result = {"bbox_xyxy":bbox, "material":mats[0], "material_probabilities":mats, "height":height}
            yield({"stage":"result", "image":img.copy(), "info":result})

    def run(self, source):
        img,res = self.pipeline(source)
        print(res)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def estimate_pole_height(self, depth_map, bbox, fov_deg=90, pitch_deg=5, cam_height_m=3):
        H, W = depth_map.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1: return None

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)

        cy = H / 2.0
        fov_rad = np.deg2rad(fov_deg)
        fy = (H / 2.0) / np.tan(fov_rad / 2.0)

        cx = int((x1 + x2) / 2)
        strip = max(1, int((x2 - x1) * 0.05))
        xs = np.arange(max(cx - strip, 0), min(cx + strip + 1, W))

        top_ys = np.arange(y1, min(y1 + 5, y2))
        bot_ys = np.arange(max(y2 - 5, y1), y2)

        if len(top_ys) < 3 or len(bot_ys) < 3: return None

        top_depths = depth_map[np.ix_(top_ys, xs)].flatten()
        bot_depths = depth_map[np.ix_(bot_ys, xs)].flatten()

        top_depths = top_depths[np.isfinite(top_depths)]
        bot_depths = bot_depths[np.isfinite(bot_depths)]

        if len(top_depths) < 10 or len(bot_depths) < 10: return None

        depth_top = np.median(top_depths)
        depth_bottom = np.median(bot_depths)

        var_top = np.std(top_depths) / (depth_top + 1e-6)
        var_bot = np.std(bot_depths) / (depth_bottom + 1e-6)
        depth_stability = np.exp(-(var_top + var_bot))

        top_y = np.mean(top_ys)
        bottom_y = np.mean(bot_ys)

        theta_top = np.arctan((top_y - cy) / fy)
        theta_bottom = np.arctan((bottom_y - cy) / fy)

        Y_top = depth_top * np.tan(theta_top)
        Y_bottom = depth_bottom * np.tan(theta_bottom)

        if abs(Y_bottom) < 1e-3: return None

        scale = (-cam_height_m) / Y_bottom
        height = (Y_top - Y_bottom) * scale

        if height <= 0 or height > 50: return None

        bbox_height = y2 - y1
        bbox_width = x2 - x1
        aspect_ratio = bbox_height / (bbox_width + 1e-6)
        vertical_conf = np.clip((aspect_ratio - 2) / 6, 0, 1)

        confidence = depth_stability * vertical_conf
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return {"estimated_height":float(height), "confidence":confidence}

if __name__ == "__main__":
    model = PoleModel()
    #model.train()
    Tk().withdraw()
    f = filedialog.askopenfilename(filetypes=[("Image", [".jpg",".png",".jpeg"])] )
    model.run(f)